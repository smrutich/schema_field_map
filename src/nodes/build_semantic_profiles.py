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
from typing import Any

from src.clients import LLMManager
from src.models import SemanticProfile, SourceField, TableProfileResponse
from src.prompts import SEMANTIC_PROFILE_SYSTEM_PROMPT, SEMANTIC_PROFILE_USER_TEMPLATE
from src.state import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_fields_block(fields: list[SourceField]) -> str:
    """Format source fields for the LLM prompt."""
    lines = []
    for sf in fields:
        parts = [f"- {sf.field_name} ({sf.sql_type})"]
        if sf.constraints:
            parts.append(f"  Constraints: {', '.join(sf.constraints)}")
        if sf.comment:
            parts.append(f"  Comment: {sf.comment}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


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
        - semantic_profiles: dict[str, SemanticProfile]
        - source_fields: list[SourceField] (updated with enriched text_repr)
    """
    if llm_manager is None:
        llm_manager = LLMManager()

    source_fields = state["source_fields"]
    errors = state.get("errors", [])
    semantic_profiles: dict[str, SemanticProfile] = {}

    # Group fields by table for batched LLM calls
    sorted_fields = sorted(source_fields, key=attrgetter("table_name"))
    table_groups = groupby(sorted_fields, key=attrgetter("table_name"))

    for table_name, fields_iter in table_groups:
        fields = list(fields_iter)
        fields_block = _build_fields_block(fields)

        user_prompt = SEMANTIC_PROFILE_USER_TEMPLATE.format(
            table_name=table_name,
            fields_block=fields_block,
        )

        logger.info(f"Profiling {len(fields)} fields from table '{table_name}'")

        try:
            response = llm_manager.call_routing(
                messages=[
                    {"role": "system", "content": SEMANTIC_PROFILE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=TableProfileResponse,
            )

            # Map profiles by field_name for lookup
            profile_map = {p.field_name: p for p in response.profiles}

            for sf in fields:
                key = f"{sf.table_name}.{sf.field_name}"
                if sf.field_name in profile_map:
                    p = profile_map[sf.field_name]
                    profile = SemanticProfile(
                        entity=p.entity,
                        concept=p.concept,
                        business_meaning=p.business_meaning,
                        keywords=p.keywords,
                    )
                    semantic_profiles[key] = profile
                else:
                    logger.warning(
                        f"No profile returned for {key} — using empty profile"
                    )
                    errors.append({
                        "node": "build_semantic_profiles",
                        "field": key,
                        "error": "LLM did not return a profile for this field",
                    })

        except Exception as e:
            logger.error(f"Failed to profile table '{table_name}': {e}")
            for sf in fields:
                key = f"{sf.table_name}.{sf.field_name}"
                errors.append({
                    "node": "build_semantic_profiles",
                    "field": key,
                    "error": str(e),
                })

    # --- Enrich source field text_repr with profile data ---
    enriched_fields: list[SourceField] = []
    for sf in source_fields:
        key = f"{sf.table_name}.{sf.field_name}"
        profile = semantic_profiles.get(key)

        if profile:
            # Rebuild text_repr with semantic enrichment
            enriched_text = (
                f"table:{sf.table_name} | field:{sf.field_name} | "
                f"type:{sf.sql_type} | nullable:{sf.nullable}"
            )
            if sf.constraints:
                enriched_text += f" | constraints:{', '.join(sf.constraints)}"
            if sf.comment:
                enriched_text += f" | comment:{sf.comment}"
            enriched_text += f" | meaning:{profile.business_meaning}"
            enriched_text += f" | keywords:{', '.join(profile.keywords)}"

            sf = SourceField(
                table_name=sf.table_name,
                field_name=sf.field_name,
                sql_type=sf.sql_type,
                nullable=sf.nullable,
                constraints=sf.constraints,
                comment=sf.comment,
                text_repr=enriched_text,
            )

        enriched_fields.append(sf)

    # --- Write to state ---
    state["source_fields"] = enriched_fields
    state["semantic_profiles"] = semantic_profiles
    state["errors"] = errors

    logger.info(
        f"Semantic profiling complete: {len(semantic_profiles)}/{len(source_fields)} fields profiled"
    )

    return state
