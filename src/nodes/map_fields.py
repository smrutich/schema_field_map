"""
Step 10 — Node: map_fields

LLM calls: 5–10 (batched).

Primary reasoning node. For each source field with retrieved candidates,
asks the LLM to select the best destination match (or null) and to fill
the `notes` field with any required value-level transformation logic in
the same call. There is no separate derive_transformations pass.

Batching strategy:
    - Group up to _BATCH_SIZE source fields per call when:
        * They belong to the same source table, AND
        * Their candidate sets are disjoint (no shared destination paths)
    - Overlapping candidate sets are processed individually to avoid the
      LLM implicitly ranking competing claims.
"""

from __future__ import annotations

import logging
from itertools import groupby
from typing import Any

from src.clients import LLMManager
from src.models import CandidateMatch, FieldMapping, FieldMappingProposal, SourceField
from src.prompts import MAP_FIELDS_SYSTEM_PROMPT, MAP_FIELDS_USER_TEMPLATE
from src.state import PipelineState

logger = logging.getLogger(__name__)

# Maximum fields per batched LLM call
_BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_source_field_block(
    sf: SourceField,
    candidates: list[CandidateMatch],
    dest_lookup: dict[str, dict[str, Any]],
) -> str:
    """Format one source field + its candidates for the prompt."""
    lines = [f"Source field: {sf.table_name}.{sf.field_name}"]
    lines.append(f"  Type: {sf.sql_type}")
    lines.append(f"  Nullable: {sf.nullable}")
    if sf.constraints:
        lines.append(f"  Constraints: {', '.join(sf.constraints)}")
    if sf.comment:
        lines.append(f"  Comment: {sf.comment}")

    lines.append("  Candidates:")
    for i, c in enumerate(candidates, 1):
        info = dest_lookup.get(c.destination_field, {})
        bson = info.get("bson_type", "unknown")
        line = (
            f"    {i}. {c.destination_field} "
            f"(type: {bson}, "
            f"hybrid_score: {c.hybrid_score:.4f}, "
            f"confidence_prior: {c.retrieval_confidence_prior})"
        )
        if info.get("comment"):
            line += f"\n       Comment: {info['comment']}"
        lines.append(line)

    return "\n".join(lines)


def _candidate_paths(candidates: list[CandidateMatch]) -> set[str]:
    return {c.destination_field for c in candidates}


def _build_batches(
    items: list[tuple[str, SourceField, list[CandidateMatch]]],
) -> list[list[tuple[str, SourceField, list[CandidateMatch]]]]:
    """Group fields into batches: same table + disjoint candidate sets, max _BATCH_SIZE."""
    sorted_items = sorted(items, key=lambda x: x[1].table_name)
    batches: list[list[tuple[str, SourceField, list[CandidateMatch]]]] = []

    for _, group_iter in groupby(sorted_items, key=lambda x: x[1].table_name):
        current: list[tuple[str, SourceField, list[CandidateMatch]]] = []
        current_paths: set[str] = set()
        for item in group_iter:
            paths = _candidate_paths(item[2])
            if current and (
                len(current) >= _BATCH_SIZE or paths & current_paths
            ):
                batches.append(current)
                current = []
                current_paths = set()
            current.append(item)
            current_paths |= paths
        if current:
            batches.append(current)

    return batches


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def map_fields(
    state: PipelineState,
    llm_manager: LLMManager | None = None,
) -> PipelineState:
    """LangGraph node: select best destination + transformation notes per source field.

    LLM calls: 5–10 (batched).

    Reads from state:
        - source_fields, destination_fields, candidate_matches

    Writes to state:
        - field_mappings: list[FieldMapping]
    """
    if llm_manager is None:
        llm_manager = LLMManager()

    source_fields = state["source_fields"]
    destination_fields = state["destination_fields"]
    candidate_matches = state["candidate_matches"]
    errors = state.get("errors", [])

    # Lookups
    dest_lookup: dict[str, dict[str, Any]] = {
        df.path: {
            "bson_type": df.bson_type,
            "comment": df.comment,
            "collection_name": df.collection_name,
        }
        for df in destination_fields
    }
    source_lookup: dict[str, SourceField] = {
        f"{sf.table_name}.{sf.field_name}": sf for sf in source_fields
    }

    # Partition into mapped vs. no-candidates
    fields_with_candidates: list[tuple[str, SourceField, list[CandidateMatch]]] = []
    unmapped: list[str] = []
    for key, candidates in candidate_matches.items():
        if candidates:
            fields_with_candidates.append((key, source_lookup[key], candidates))
        else:
            unmapped.append(key)

    batches = _build_batches(fields_with_candidates)
    logger.info(
        f"map_fields: {len(fields_with_candidates)} fields in "
        f"{len(batches)} batches; {len(unmapped)} unmapped"
    )

    field_mappings: list[FieldMapping] = []

    for batch_idx, batch in enumerate(batches, 1):
        blocks = [
            _format_source_field_block(sf, candidates, dest_lookup)
            for _, sf, candidates in batch
        ]
        user_prompt = MAP_FIELDS_USER_TEMPLATE.format(
            source_fields_block="\n\n---\n\n".join(blocks)
        )

        response_model = (
            FieldMappingProposal if len(batch) == 1 else list[FieldMappingProposal]
        )
        logger.info(f"  Batch {batch_idx}/{len(batches)}: {len(batch)} field(s)")

        try:
            result = llm_manager.call_mapping(
                messages=[
                    {"role": "system", "content": MAP_FIELDS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
            )
            if isinstance(result, FieldMappingProposal):
                result = [result]
            # Promote each LLM proposal to a finalized FieldMapping with
            # relationship_validated=False (only validate_relationships sets True).
            field_mappings.extend(
                FieldMapping(**proposal.model_dump()) for proposal in result
            )
        except Exception as e:
            logger.error(f"  Batch {batch_idx} failed: {e}")
            for key, _sf, _ in batch:
                errors.append({"node": "map_fields", "field": key, "error": str(e)})

    state["field_mappings"] = field_mappings
    state["errors"] = errors

    logger.info(
        f"map_fields complete: {len(field_mappings)} mappings produced, "
        f"{len(unmapped)} fields had no candidates"
    )

    return state
