"""
Step 10 & 11 — Node: map_fields (includes derive_transformations)

LLM calls: 5–10 for mapping (batched) + 1–2 for transformations

The primary reasoning node. For each source field with candidates,
asks the LLM to select the best destination match or indicate no match.
Then derives transformation rules for confirmed mappings.

Batching Strategy:
    - Group up to 5 source fields per call when they belong to the same table
      AND their candidate sets are disjoint (no shared destination paths).
    - Overlapping candidate sets are processed individually to avoid
      degraded quality from implicit ranking.

Transformation Strategy:
    - Trivial transforms (simple type passthrough) are assigned rule-based (NONE)
    - Non-trivial transforms are batched into 1–2 LLM calls
"""

from __future__ import annotations

import logging
import re
from itertools import groupby
from typing import Any

from src.clients import LLMManager
from src.models import CandidateMatch, FieldMapping, SourceField, TransformationRule
from src.prompts import (
    DERIVE_TRANSFORMATIONS_SYSTEM_PROMPT,
    DERIVE_TRANSFORMATIONS_USER_TEMPLATE,
    MAP_FIELDS_SYSTEM_PROMPT,
    MAP_FIELDS_USER_TEMPLATE,
)
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
) -> str:
    """Format a single source field + its candidates for the prompt."""
    lines = [f"Source field: {sf.table_name}.{sf.field_name}"]
    lines.append(f"  Type: {sf.sql_type}")
    lines.append(f"  Nullable: {sf.nullable}")
    if sf.constraints:
        lines.append(f"  Constraints: {', '.join(sf.constraints)}")
    if sf.comment:
        lines.append(f"  Comment: {sf.comment}")

    lines.append("  Candidates:")
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"    {i}. {c.destination_field} "
            f"(type: lookup from schema, "
            f"hybrid_score: {c.hybrid_score:.4f}, "
            f"confidence_prior: {c.retrieval_confidence_prior})"
        )

    return "\n".join(lines)


def _format_source_field_block_with_dest_info(
    sf: SourceField,
    candidates: list[CandidateMatch],
    dest_lookup: dict[str, Any],
) -> str:
    """Format a single source field + its candidates with destination type info."""
    lines = [f"Source field: {sf.table_name}.{sf.field_name}"]
    lines.append(f"  Type: {sf.sql_type}")
    lines.append(f"  Nullable: {sf.nullable}")
    if sf.constraints:
        lines.append(f"  Constraints: {', '.join(sf.constraints)}")
    if sf.comment:
        lines.append(f"  Comment: {sf.comment}")

    lines.append("  Candidates:")
    for i, c in enumerate(candidates, 1):
        dest_info = dest_lookup.get(c.destination_field, {})
        bson_type = dest_info.get("bson_type", "unknown")
        dest_comment = dest_info.get("comment", "")
        candidate_line = (
            f"    {i}. {c.destination_field} "
            f"(type: {bson_type}, "
            f"hybrid_score: {c.hybrid_score:.4f}, "
            f"confidence_prior: {c.retrieval_confidence_prior})"
        )
        if dest_comment:
            candidate_line += f"\n       Comment: {dest_comment}"
        lines.append(candidate_line)

    return "\n".join(lines)


def _get_candidate_paths(candidates: list[CandidateMatch]) -> set[str]:
    """Extract all destination paths from a candidate list."""
    return {c.destination_field for c in candidates}


def _build_batches(
    fields_with_candidates: list[tuple[str, SourceField, list[CandidateMatch]]],
) -> list[list[tuple[str, SourceField, list[CandidateMatch]]]]:
    """Group fields into batches for LLM calls.

    Rules:
    - Same table only
    - Disjoint candidate sets within a batch
    - Max _BATCH_SIZE per batch
    """
    # Sort by table
    sorted_items = sorted(fields_with_candidates, key=lambda x: x[1].table_name)

    batches: list[list[tuple[str, SourceField, list[CandidateMatch]]]] = []

    for _, table_group_iter in groupby(sorted_items, key=lambda x: x[1].table_name):
        table_items = list(table_group_iter)

        current_batch: list[tuple[str, SourceField, list[CandidateMatch]]] = []
        current_paths: set[str] = set()

        for item in table_items:
            key, sf, candidates = item
            item_paths = _get_candidate_paths(candidates)

            # Check for overlap with current batch
            if current_batch and (
                len(current_batch) >= _BATCH_SIZE
                or item_paths & current_paths  # overlap detected
            ):
                # Flush current batch
                batches.append(current_batch)
                current_batch = []
                current_paths = set()

            current_batch.append(item)
            current_paths |= item_paths

        # Flush remaining
        if current_batch:
            batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


def map_fields(
    state: PipelineState,
    llm_manager: LLMManager | None = None,
) -> PipelineState:
    """LangGraph node: Map source fields to destination fields using LLM.

    For each source field with candidates, asks the LLM to select the best
    match. Fields are batched by table with disjoint candidate sets.
    Fields with no candidates are marked as unmapped.

    LLM calls: 5–10 (batched)

    Reads from state:
        - source_fields: list[SourceField]
        - destination_fields: list[DestinationField]
        - candidate_matches: dict[str, list[CandidateMatch]]

    Writes to state:
        - field_mappings: list[FieldMapping]
    """
    if llm_manager is None:
        llm_manager = LLMManager()

    source_fields = state["source_fields"]
    destination_fields = state["destination_fields"]
    candidate_matches = state["candidate_matches"]
    errors = state.get("errors", [])

    # Build destination field lookup for type info in prompts
    dest_lookup: dict[str, dict[str, Any]] = {}
    for df in destination_fields:
        dest_lookup[df.path] = {
            "bson_type": df.bson_type,
            "comment": df.comment,
            "collection_name": df.collection_name,
        }

    # Build source field lookup
    source_lookup: dict[str, SourceField] = {}
    for sf in source_fields:
        source_lookup[f"{sf.table_name}.{sf.field_name}"] = sf

    # Separate fields with candidates from those without
    fields_with_candidates: list[tuple[str, SourceField, list[CandidateMatch]]] = []
    fields_without_candidates: list[str] = []

    for key, candidates in candidate_matches.items():
        if candidates:
            sf = source_lookup[key]
            fields_with_candidates.append((key, sf, candidates))
        else:
            fields_without_candidates.append(key)

    # Build batches
    batches = _build_batches(fields_with_candidates)

    logger.info(
        f"map_fields: {len(fields_with_candidates)} fields to map in "
        f"{len(batches)} batches, {len(fields_without_candidates)} unmapped"
    )

    field_mappings: list[FieldMapping] = []

    for batch_idx, batch in enumerate(batches):
        # Build the prompt for this batch
        source_blocks = []
        for key, sf, candidates in batch:
            block = _format_source_field_block_with_dest_info(
                sf, candidates, dest_lookup
            )
            source_blocks.append(block)

        source_fields_block = "\n\n---\n\n".join(source_blocks)
        user_prompt = MAP_FIELDS_USER_TEMPLATE.format(
            source_fields_block=source_fields_block
        )

        # Choose response model based on batch size
        if len(batch) == 1:
            response_model = FieldMapping
        else:
            response_model = list[FieldMapping]

        logger.info(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch)} fields")

        try:
            result = llm_manager.call_mapping(
                messages=[
                    {"role": "system", "content": MAP_FIELDS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
            )

            # Normalize to list
            if isinstance(result, FieldMapping):
                result = [result]

            field_mappings.extend(result)

        except Exception as e:
            logger.error(f"  Batch {batch_idx + 1} failed: {e}")
            # Mark all fields in this batch as failed
            for key, sf, _ in batch:
                errors.append({
                    "node": "map_fields",
                    "field": key,
                    "error": str(e),
                })

    # --- Derive transformations for confirmed mappings ---
    confirmed_mappings = [fm for fm in field_mappings if fm.destination_field is not None]

    transformation_rules = _derive_transformations(
        confirmed_mappings, source_lookup, dest_lookup, llm_manager, errors
    )

    # --- Write to state ---
    state["field_mappings"] = field_mappings
    state["transformation_rules"] = transformation_rules
    state["errors"] = errors

    logger.info(
        f"map_fields complete: {len(field_mappings)} mappings produced, "
        f"{len(fields_without_candidates)} fields had no candidates, "
        f"{len(transformation_rules)} transformation rules derived"
    )

    return state


# ---------------------------------------------------------------------------
# Transformation Derivation (Step 11 — integrated)
# ---------------------------------------------------------------------------

# Patterns for detecting value codes in comments (e.g., "A=Active, I=Inactive")
_VALUE_CODE_PATTERN = re.compile(r"[A-Z0-9]\s*[=:]\s*\w+", re.IGNORECASE)

# Trivial type mappings that don't need LLM (direct passthrough)
_TRIVIAL_TRANSFORMS = {
    ("VARCHAR", "String"),
    ("CHAR", "String"),
    ("TEXT", "String"),
    ("INT", "Number"),
    ("DECIMAL", "Number"),
    ("FLOAT", "Number"),
    ("DOUBLE", "Number"),
}


def _needs_llm_transform(
    fm: FieldMapping,
    source_lookup: dict[str, SourceField],
    dest_lookup: dict[str, dict[str, Any]],
) -> bool:
    """Determine if a mapping needs LLM to derive transformation logic.

    Uses heuristics based on type patterns and metadata — no hardcoded field names.
    """
    # Find the source field (try with common table prefix patterns)
    sf = None
    for key, field in source_lookup.items():
        if field.field_name == fm.source_field:
            sf = field
            break

    if sf is None:
        return False

    dest_info = dest_lookup.get(fm.destination_field or "", {})
    dest_type = dest_info.get("bson_type", "")

    # Check for value code patterns in comments (e.g., "A=Active, I=Inactive")
    if sf.comment and _VALUE_CODE_PATTERN.search(sf.comment):
        return True

    # Primary key going to ObjectId → ID_STRATEGY needed
    if "PRIMARY KEY" in sf.constraints and dest_type == "ObjectId":
        return True

    # Integer/TINYINT going to Boolean → TYPE_CAST needed
    if "TINYINT" in sf.sql_type.upper() and dest_type == "Boolean":
        return True

    # DATETIME going to ISODate → FORMAT_CHANGE needed
    if "DATETIME" in sf.sql_type.upper() and dest_type == "ISODate":
        return True
    if "DATE" == sf.sql_type.upper() and dest_type == "ISODate":
        return True

    # FK reference going to ObjectId (cross-entity) → ID_STRATEGY
    fk_constraints = [c for c in sf.constraints if c.startswith("FK ->")]
    if fk_constraints and dest_type == "ObjectId":
        return True

    # Check if source and dest base types are trivially compatible
    source_base = sf.sql_type.split("(")[0].upper()
    if (source_base, dest_type) in _TRIVIAL_TRANSFORMS:
        return False

    # Default: if types differ meaningfully, ask LLM
    return True


def _format_field_pair(
    fm: FieldMapping,
    source_lookup: dict[str, SourceField],
    dest_lookup: dict[str, dict[str, Any]],
) -> str:
    """Format a confirmed mapping pair for the transformation prompt."""
    sf = None
    for key, field in source_lookup.items():
        if field.field_name == fm.source_field:
            sf = field
            break

    lines = [f"- {fm.source_field} ({fm.type_transform})"]
    if sf:
        if sf.constraints:
            lines.append(f"  Constraints: {', '.join(sf.constraints)}")
        if sf.comment:
            lines.append(f"  Source comment: {sf.comment}")

    dest_info = dest_lookup.get(fm.destination_field or "", {})
    if dest_info.get("comment"):
        lines.append(f"  Destination comment: {dest_info['comment']}")

    return "\n".join(lines)


def _derive_transformations(
    confirmed_mappings: list[FieldMapping],
    source_lookup: dict[str, SourceField],
    dest_lookup: dict[str, dict[str, Any]],
    llm_manager: LLMManager,
    errors: list[dict[str, Any]],
) -> list[TransformationRule]:
    """Derive transformation rules for confirmed mappings.

    - Trivial transforms: assigned NONE rule-based (no LLM)
    - Non-trivial transforms: batched into LLM calls
    """
    trivial_rules: list[TransformationRule] = []
    needs_llm: list[FieldMapping] = []

    for fm in confirmed_mappings:
        if _needs_llm_transform(fm, source_lookup, dest_lookup):
            needs_llm.append(fm)
        else:
            trivial_rules.append(
                TransformationRule(
                    source_field=fm.source_field,
                    destination_field=fm.destination_field or "",
                    transform_type="NONE",
                    transform_logic=None,
                    example=None,
                )
            )

    logger.info(
        f"Transformations: {len(trivial_rules)} trivial (NONE), "
        f"{len(needs_llm)} need LLM extraction"
    )

    if not needs_llm:
        return trivial_rules

    # Batch non-trivial mappings into LLM calls (up to 10 per call)
    llm_rules: list[TransformationRule] = []
    batch_size = 10

    for i in range(0, len(needs_llm), batch_size):
        batch = needs_llm[i : i + batch_size]

        field_pairs_block = "\n\n".join(
            _format_field_pair(fm, source_lookup, dest_lookup) for fm in batch
        )

        user_prompt = DERIVE_TRANSFORMATIONS_USER_TEMPLATE.format(
            field_pairs_block=field_pairs_block
        )

        logger.info(
            f"  Transformation LLM call: {len(batch)} pairs"
        )

        try:
            if len(batch) == 1:
                response_model = TransformationRule
            else:
                response_model = list[TransformationRule]

            result = llm_manager.call_mapping(
                messages=[
                    {"role": "system", "content": DERIVE_TRANSFORMATIONS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
            )

            if isinstance(result, TransformationRule):
                result = [result]

            llm_rules.extend(result)

        except Exception as e:
            logger.error(f"  Transformation extraction failed: {e}")
            for fm in batch:
                errors.append({
                    "node": "derive_transformations",
                    "field": fm.source_field,
                    "error": str(e),
                })

    return trivial_rules + llm_rules
