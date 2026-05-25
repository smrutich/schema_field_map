"""
Step 6 — Node: build_semantic_profiles

LLM calls: 2–3 batched calls (one per source table to minimize calls)

Purpose:
    Generate enriched business meaning for each field independently,
    without any cross-schema exposure. The profile output is then merged
    back into each field's text_repr before embedding.
"""

from __future__ import annotations

import logging
from itertools import groupby
from operator import attrgetter

from src.clients import LLMManager
from src.models import SemanticProfile, SourceField, build_source_text_repr
from src.prompts import SEMANTIC_PROFILE_SYSTEM_PROMPT, SEMANTIC_PROFILE_USER_TEMPLATE
from src.state import PipelineState

logger = logging.getLogger(__name__)


def _build_fields_block(fields: list[SourceField]) -> str:
    """Format source fields for the LLM prompt."""
    blocks = []
    for sf in fields:
        parts = [f"- {sf.field_name} ({sf.sql_type})"]
        if sf.constraints:
            parts.append(f"  Constraints: {', '.join(sf.constraints)}")
        if sf.comment:
            parts.append(f"  Comment: {sf.comment}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def build_semantic_profiles(
    state: PipelineState,
    llm_manager: LLMManager | None = None,
) -> PipelineState:
    """LangGraph node: Generate semantic profiles for all source fields.

    Makes one LLM call per source table (batched). Merges profile keywords
    and business_meaning back into each field's text_repr for improved
    embedding quality.

    LLM calls: 2–3 (one per source table)

    Reads from state:
        - source_fields: list[SourceField]

    Writes to state:
        - semantic_profiles: dict[str, SemanticProfile]   (keyed by "table.field")
        - source_fields: list[SourceField]   (text_repr enriched in-place)
    """
    if llm_manager is None:
        llm_manager = LLMManager()

    source_fields = state["source_fields"]
    errors = state.get("errors", [])
    semantic_profiles: dict[str, SemanticProfile] = {}

    # Group fields by table for batched LLM calls
    sorted_fields = sorted(source_fields, key=attrgetter("table_name"))

    for table_name, fields_iter in groupby(sorted_fields, key=attrgetter("table_name")):
        fields = list(fields_iter)
        user_prompt = SEMANTIC_PROFILE_USER_TEMPLATE.format(
            table_name=table_name,
            fields_block=_build_fields_block(fields),
        )
        logger.info(f"Profiling {len(fields)} fields from table '{table_name}'")

        try:
            profiles = llm_manager.call_routing(
                messages=[
                    {"role": "system", "content": SEMANTIC_PROFILE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=list[SemanticProfile],
            )
        except Exception as e:
            logger.error(f"Failed to profile table '{table_name}': {e}")
            for sf in fields:
                errors.append({
                    "node": "build_semantic_profiles",
                    "field": f"{sf.table_name}.{sf.field_name}",
                    "error": str(e),
                })
            continue

        profile_by_name = {p.field_name: p for p in profiles}
        for sf in fields:
            if sf.field_name in profile_by_name:
                semantic_profiles[f"{sf.table_name}.{sf.field_name}"] = profile_by_name[sf.field_name]
            else:
                logger.warning(
                    f"No profile returned for {sf.table_name}.{sf.field_name} — skipping enrichment"
                )
                errors.append({
                    "node": "build_semantic_profiles",
                    "field": f"{sf.table_name}.{sf.field_name}",
                    "error": "LLM did not return a profile for this field",
                })

    # Enrich text_repr with profile data via model_copy
    enriched: list[SourceField] = []
    for sf in source_fields:
        profile = semantic_profiles.get(f"{sf.table_name}.{sf.field_name}")
        if profile is None:
            enriched.append(sf)
            continue
        new_text_repr = build_source_text_repr(
            table_name=sf.table_name,
            field_name=sf.field_name,
            sql_type=sf.sql_type,
            nullable=sf.nullable,
            constraints=sf.constraints,
            comment=sf.comment,
            profile=profile,
        )
        enriched.append(sf.model_copy(update={"text_repr": new_text_repr}))

    state["source_fields"] = enriched
    state["semantic_profiles"] = semantic_profiles
    state["errors"] = errors

    logger.info(
        f"Semantic profiling complete: {len(semantic_profiles)}/{len(source_fields)} fields profiled"
    )
    return state
