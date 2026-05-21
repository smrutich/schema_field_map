"""
Step 9 — Node: retrieve_candidates

LLM calls: None

Process (per source field):
    1. Embed the source field's enriched text_repr on-demand
    2. Compute hybrid score (0.8 * cosine + 0.2 * lexical) against destinations
    3. Filter to destination collection assigned by routing node
    4. Take top-3 above threshold (0.40)
    5. Assign retrieval_confidence_prior (HIGH/MEDIUM/LOW)
    6. If zero candidates pass → flag as unmapped

State Output:
    candidate_matches: dict keyed by "table_name.field_name" -> list[CandidateMatch]
"""

from __future__ import annotations

import logging

from src.models import CandidateMatch
from src.nodes.embedding_manager import EmbeddingManager
from src.state import PipelineState

logger = logging.getLogger(__name__)


def retrieve_candidates(
    state: PipelineState,
    embedding_manager: EmbeddingManager | None = None,
    top_k: int = 3,
    threshold: float = 0.40,
) -> PipelineState:
    """LangGraph node: Retrieve top candidate matches for each source field.

    Uses hybrid scoring (embedding + lexical) filtered by table routing.
    No LLM calls — purely embedding similarity and string matching.

    Reads from state:
        - source_fields: list[SourceField]
        - destination_fields: list[DestinationField]
        - destination_embeddings: NDArray[np.float32]
        - destination_field_index: list[str]
        - table_routing: list[TableRoutingDecision]

    Writes to state:
        - candidate_matches: dict[str, list[CandidateMatch]]
    """
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()
        # Re-embed destinations if manager is fresh (matrix not loaded)
        text_reprs = [df.text_repr for df in state["destination_fields"]]
        paths = [df.path for df in state["destination_fields"]]
        embedding_manager.embed_destination_fields(text_reprs, paths)
    elif embedding_manager._destination_matrix is None:
        # Manager provided but destinations not yet embedded
        text_reprs = [df.text_repr for df in state["destination_fields"]]
        paths = [df.path for df in state["destination_fields"]]
        embedding_manager.embed_destination_fields(text_reprs, paths)

    source_fields = state["source_fields"]
    destination_fields = state["destination_fields"]
    table_routing = state.get("table_routing", [])
    errors = state.get("errors", [])

    # Build routing lookup: source_table -> destination_collection
    routing_map: dict[str, str] = {}
    for rd in table_routing:
        routing_map[rd.source_table] = rd.destination_collection

    # Build collection lookup: destination_path -> collection_name
    collection_names = [df.collection_name for df in destination_fields]

    candidate_matches: dict[str, list[CandidateMatch]] = {}
    unmapped_count = 0

    for sf in source_fields:
        key = f"{sf.table_name}.{sf.field_name}"

        # Determine collection filter from routing
        collection_filter = routing_map.get(sf.table_name)

        # Retrieve candidates using EmbeddingManager
        raw_candidates = embedding_manager.retrieve_candidates(
            source_text_repr=sf.text_repr,
            source_field_name=sf.field_name,
            collection_filter=collection_filter,
            collection_names=collection_names,
            top_k=top_k,
            threshold=threshold,
        )

        if raw_candidates:
            # Convert to CandidateMatch Pydantic models
            matches = []
            for c in raw_candidates:
                emb_sim = c["embedding_similarity"]
                lex_sim = c["lexical_similarity"]
                # Recompute hybrid_score from rounded components to satisfy validator
                h_score = round(0.8 * emb_sim + 0.2 * lex_sim, 4)
                match = CandidateMatch(
                    destination_field=c["destination_field"],
                    embedding_similarity=emb_sim,
                    lexical_similarity=lex_sim,
                    hybrid_score=h_score,
                    retrieval_confidence_prior=c["retrieval_confidence_prior"],
                )
                matches.append(match)
            candidate_matches[key] = matches
        else:
            # No candidates above threshold — flag as unmapped
            candidate_matches[key] = []
            unmapped_count += 1
            logger.debug(f"No candidates for {key} (below threshold {threshold})")

    state["candidate_matches"] = candidate_matches

    matched_count = len(candidate_matches) - unmapped_count
    logger.info(
        f"Candidate retrieval complete: "
        f"{matched_count}/{len(source_fields)} fields have candidates, "
        f"{unmapped_count} unmapped"
    )

    state["errors"] = errors
    return state
