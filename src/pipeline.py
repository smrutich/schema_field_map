"""
Schema Field Mapper — LangGraph Pipeline Definition

Wires all pipeline nodes into a stateful LangGraph graph with:
- Linear edge sequence for the main pipeline
- Conditional edge after assemble_output for low-confidence review
- Conditional edge after map_fields for total failure detection
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.clients import LLMManager
from src.nodes.assemble_output import assemble_output
from src.nodes.build_semantic_profiles import build_semantic_profiles
from src.nodes.convert_raw_schema import parse_schemas
from src.nodes.embed_destination_fields import embed_destination_fields
from src.nodes.embedding_manager import EmbeddingManager
from src.nodes.flag_for_review import flag_for_review, should_flag_for_review
from src.nodes.map_fields import map_fields
from src.nodes.retrieve_candidates import retrieve_candidates
from src.nodes.route_tables import route_tables
from src.nodes.validate_relationships import validate_relationships
from src.state import PipelineState

logger = logging.getLogger(__name__)


def _check_mapping_success(state: PipelineState) -> str:
    """Conditional edge after map_fields: check if any mappings were produced."""
    field_mappings = state.get("field_mappings", [])
    if field_mappings:
        return "validate_relationships"
    else:
        return "error_handler"


def _check_low_confidence(state: PipelineState) -> str:
    """Conditional edge after assemble_output: route to review if needed."""
    if should_flag_for_review(state):
        return "flag_for_review"
    else:
        return END


def _error_handler(state: PipelineState) -> PipelineState:
    """Terminal node for total mapping failure."""
    errors = state.get("errors", [])
    logger.error(
        f"Pipeline failed: no field mappings produced. "
        f"{len(errors)} errors recorded."
    )
    return state


def build_pipeline(
    llm_manager: LLMManager | None = None,
    embedding_manager: EmbeddingManager | None = None,
) -> StateGraph:
    """Build and return the compiled LangGraph pipeline.

    Args:
        llm_manager: Optional shared LLM client (created if not provided)
        embedding_manager: Optional shared embedding model (created if not provided)

    Returns:
        Compiled StateGraph ready to invoke
    """
    if llm_manager is None:
        llm_manager = LLMManager()
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()

    # --- Define node wrappers that inject shared resources ---

    def _parse_schemas(state: PipelineState) -> PipelineState:
        return parse_schemas(state)

    def _build_semantic_profiles(state: PipelineState) -> PipelineState:
        return build_semantic_profiles(state, llm_manager=llm_manager)

    def _embed_destination_fields(state: PipelineState) -> PipelineState:
        return embed_destination_fields(state, embedding_manager=embedding_manager)

    def _route_tables(state: PipelineState) -> PipelineState:
        return route_tables(state, llm_manager=llm_manager)

    def _retrieve_candidates(state: PipelineState) -> PipelineState:
        return retrieve_candidates(state, embedding_manager=embedding_manager)

    def _map_fields(state: PipelineState) -> PipelineState:
        return map_fields(state, llm_manager=llm_manager)

    def _validate_relationships(state: PipelineState) -> PipelineState:
        return validate_relationships(state)

    def _assemble_output(state: PipelineState) -> PipelineState:
        return assemble_output(state)

    def _flag_for_review(state: PipelineState) -> PipelineState:
        return flag_for_review(state)

    # --- Build the graph ---
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("parse_schemas", _parse_schemas)
    graph.add_node("build_semantic_profiles", _build_semantic_profiles)
    graph.add_node("embed_destination_fields", _embed_destination_fields)
    graph.add_node("route_tables", _route_tables)
    graph.add_node("retrieve_candidates", _retrieve_candidates)
    graph.add_node("map_fields", _map_fields)
    graph.add_node("validate_relationships", _validate_relationships)
    graph.add_node("assemble_output", _assemble_output)
    graph.add_node("flag_for_review", _flag_for_review)
    graph.add_node("error_handler", _error_handler)

    # --- Define edges ---

    # Entry point
    graph.set_entry_point("parse_schemas")

    # Linear sequence: parse -> profile -> embed -> route -> retrieve -> map
    graph.add_edge("parse_schemas", "build_semantic_profiles")
    graph.add_edge("build_semantic_profiles", "embed_destination_fields")
    graph.add_edge("embed_destination_fields", "route_tables")
    graph.add_edge("route_tables", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "map_fields")

    # Conditional: after map_fields, check if mappings exist
    graph.add_conditional_edges(
        "map_fields",
        _check_mapping_success,
        {
            "validate_relationships": "validate_relationships",
            "error_handler": "error_handler",
        },
    )

    # Linear: validate -> assemble
    graph.add_edge("validate_relationships", "assemble_output")

    # Conditional: after assemble_output, check for low-confidence flags
    graph.add_conditional_edges(
        "assemble_output",
        _check_low_confidence,
        {
            "flag_for_review": "flag_for_review",
            END: END,
        },
    )

    # Terminal edges
    graph.add_edge("flag_for_review", END)
    graph.add_edge("error_handler", END)

    return graph.compile()


def run_pipeline(
    llm_manager: LLMManager | None = None,
    embedding_manager: EmbeddingManager | None = None,
) -> PipelineState:
    """Build and execute the full pipeline, returning the final state.

    Args:
        llm_manager: Optional shared LLM client
        embedding_manager: Optional shared embedding model

    Returns:
        Final PipelineState with all results
    """
    pipeline = build_pipeline(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
    )

    logger.info("Starting Schema Field Mapper pipeline")

    # Execute with empty initial state
    initial_state: PipelineState = {}
    final_state = pipeline.invoke(initial_state)

    logger.info("Pipeline execution complete")
    return final_state
