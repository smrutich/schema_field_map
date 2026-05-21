"""
Step 8 — Node: route_tables

LLM calls: 1

Purpose:
    Establish which source table maps to which destination collection
    before any field-level work begins. This scoping step restricts
    candidate retrieval to the correct collection.

Constraint Compliance:
    Only table/collection names appear in the prompt — no field definitions
    from either schema are exposed together.
"""

from __future__ import annotations

import logging
from typing import Any

from src.clients import LLMManager
from src.models import TableRoutingDecision
from src.prompts import ROUTE_TABLES_SYSTEM_PROMPT, ROUTE_TABLES_USER_TEMPLATE
from src.state import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_source_tables_block(state: PipelineState) -> str:
    """Build a summary of source tables (name + field count) for the prompt."""
    table_counts: dict[str, int] = {}
    for sf in state["source_fields"]:
        table_counts[sf.table_name] = table_counts.get(sf.table_name, 0) + 1

    lines = []
    for table, count in table_counts.items():
        lines.append(f"- {table} ({count} fields)")
    return "\n".join(lines)


def _build_destination_collections_block(state: PipelineState) -> str:
    """Build a summary of destination collections (name + field count) for the prompt."""
    coll_counts: dict[str, int] = {}
    for df in state["destination_fields"]:
        coll_counts[df.collection_name] = coll_counts.get(df.collection_name, 0) + 1

    lines = []
    for coll, count in coll_counts.items():
        lines.append(f"- {coll} ({count} fields)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------


def route_tables(
    state: PipelineState,
    llm_manager: LLMManager | None = None,
) -> PipelineState:
    """LangGraph node: Route source tables to destination collections.

    Makes a single LLM call with table/collection names only (no fields).
    Stores routing decisions in state for use by retrieve_candidates.

    LLM calls: 1

    Reads from state:
        - source_fields: list[SourceField] (for table names)
        - destination_fields: list[DestinationField] (for collection names)

    Writes to state:
        - table_routing: list[TableRoutingDecision]
    """
    if llm_manager is None:
        llm_manager = LLMManager()

    errors = state.get("errors", [])

    source_tables_block = _build_source_tables_block(state)
    destination_collections_block = _build_destination_collections_block(state)

    user_prompt = ROUTE_TABLES_USER_TEMPLATE.format(
        source_tables_block=source_tables_block,
        destination_collections_block=destination_collections_block,
    )

    logger.info("Routing source tables to destination collections")

    try:
        routing_decisions = llm_manager.call_routing(
            messages=[
                {"role": "system", "content": ROUTE_TABLES_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=list[TableRoutingDecision],
        )

        state["table_routing"] = routing_decisions

        for rd in routing_decisions:
            logger.info(
                f"  {rd.source_table} → {rd.destination_collection} "
                f"(confidence: {rd.confidence})"
            )

    except Exception as e:
        logger.error(f"Table routing failed: {e}")
        errors.append({
            "node": "route_tables",
            "field": None,
            "error": str(e),
        })
        # Fallback: empty routing means retrieve_candidates won't filter by collection
        state["table_routing"] = []

    state["errors"] = errors
    return state
