"""
Step 7 — Node: embed_destination_fields

LLM calls: None

Responsibilities:
    - Take all destination_fields from state
    - Embed each field's text_repr using the local embedding model
    - Store the numpy matrix in state as destination_embeddings
    - Store the parallel list of destination paths as destination_field_index

Timing: Runs after build_semantic_profiles but before source field embedding.
Destination fields use raw text_repr (MongoDB names are already descriptive).
"""

from __future__ import annotations

import logging

from src.nodes.embedding_manager import EmbeddingManager
from src.state import PipelineState

logger = logging.getLogger(__name__)


def embed_destination_fields(
    state: PipelineState,
    embedding_manager: EmbeddingManager | None = None,
) -> PipelineState:
    """LangGraph node: Embed all destination field text_repr strings.

    Builds a pre-computed embedding matrix for all destination fields.
    Source fields are embedded on-demand during candidate retrieval (Step 9).

    LLM calls: None

    Reads from state:
        - destination_fields: list[DestinationField]

    Writes to state:
        - destination_embeddings: NDArray[np.float32]
        - destination_field_index: list[str]
    """
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()

    destination_fields = state["destination_fields"]

    text_reprs = [df.text_repr for df in destination_fields]
    paths = [df.path for df in destination_fields]

    logger.info(f"Embedding {len(destination_fields)} destination fields")

    matrix = embedding_manager.embed_destination_fields(text_reprs, paths)

    state["destination_embeddings"] = matrix
    state["destination_field_index"] = paths

    logger.info(f"Destination embeddings stored | shape={matrix.shape}")

    return state
