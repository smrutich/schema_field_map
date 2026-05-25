"""
Schema Field Mapper — LangGraph Pipeline Definition

Wires all pipeline nodes into a stateful LangGraph graph with:
- Linear edge sequence for the main pipeline
- Conditional edge after assemble_output for low-confidence review
"""

from __future__ import annotations

import logging
from functools import partial

from langgraph.graph import END, StateGraph

from src.clients import LLMManager
from src.nodes.assemble_output import assemble_output
from src.nodes.build_semantic_profiles import build_semantic_profiles
from src.nodes.embed_destination_fields import embed_destination_fields
from src.nodes.embedding_manager import EmbeddingManager
from src.nodes.flag_for_review import flag_for_review, should_flag_for_review
from src.nodes.map_fields import map_fields
from src.nodes.parse_schemas import parse_schemas
from src.nodes.retrieve_candidates import retrieve_candidates
from src.nodes.route_tables import route_tables
from src.nodes.validate_relationships import validate_relationships
from src.state import PipelineState

logger = logging.getLogger(__name__)


def _check_low_confidence(state: PipelineState) -> str:
    """Conditional edge after assemble_output: route to review if any low-confidence mapping."""
    return "flag_for_review" if should_flag_for_review(state) else END


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

    graph = StateGraph(PipelineState)

    # Register nodes — partial() injects shared resources without per-node closures.
    graph.add_node("parse_schemas", parse_schemas)
    graph.add_node("build_semantic_profiles", partial(build_semantic_profiles, llm_manager=llm_manager))
    graph.add_node("embed_destination_fields", partial(embed_destination_fields, embedding_manager=embedding_manager))
    graph.add_node("route_tables", partial(route_tables, llm_manager=llm_manager))
    graph.add_node("retrieve_candidates", partial(retrieve_candidates, embedding_manager=embedding_manager))
    graph.add_node("map_fields", partial(map_fields, llm_manager=llm_manager))
    graph.add_node("validate_relationships", validate_relationships)
    graph.add_node("assemble_output", assemble_output)
    graph.add_node("flag_for_review", flag_for_review)

    # Linear sequence
    graph.set_entry_point("parse_schemas")
    graph.add_edge("parse_schemas", "build_semantic_profiles")
    graph.add_edge("build_semantic_profiles", "embed_destination_fields")
    graph.add_edge("embed_destination_fields", "route_tables")
    graph.add_edge("route_tables", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "map_fields")
    graph.add_edge("map_fields", "validate_relationships")
    graph.add_edge("validate_relationships", "assemble_output")

    # Conditional fan-out after assembly
    graph.add_conditional_edges(
        "assemble_output",
        _check_low_confidence,
        {"flag_for_review": "flag_for_review", END: END},
    )
    graph.add_edge("flag_for_review", END)

    return graph.compile()


def run_pipeline(
    llm_manager: LLMManager | None = None,
    embedding_manager: EmbeddingManager | None = None,
) -> PipelineState:
    """Build and execute the full pipeline, returning the final state."""
    pipeline = build_pipeline(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
    )

    logger.info("Starting Schema Field Mapper pipeline")
    final_state = pipeline.invoke({})
    logger.info("Pipeline execution complete")
    return final_state
