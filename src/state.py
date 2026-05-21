"""
Step 4 — LangGraph Pipeline State

The state object is the single source of truth passed between all nodes.
Defined as a TypedDict with all fields typed against the Pydantic models.

LangGraph requires TypedDict (not Pydantic BaseModel) for graph state.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from src.models import (
    CandidateMatch,
    DestinationField,
    FieldMapping,
    FinalOutput,
    SemanticProfile,
    SourceField,
    TableMapping,
    TableRoutingDecision,
    TransformationRule,
)


class PipelineState(TypedDict, total=False):
    """Single source of truth passed between all pipeline nodes.

    All fields are optional (total=False) so nodes can progressively
    populate the state as the pipeline executes.
    """

    # --- Parsed schema fields ---
    source_fields: list[SourceField]
    destination_fields: list[DestinationField]

    # --- Semantic profiling output ---
    # Keyed by "table_name.field_name", value is SemanticProfile
    semantic_profiles: dict[str, SemanticProfile]

    # --- Embedding state ---
    # Numpy matrix of embedded destination text_repr strings
    destination_embeddings: NDArray[np.float32]
    # Destination paths in same order as embedding matrix rows
    destination_field_index: list[str]

    # --- Table routing ---
    table_routing: list[TableRoutingDecision]

    # --- Candidate retrieval ---
    # Keyed by "table_name.field_name", value is list of top-k candidates
    candidate_matches: dict[str, list[CandidateMatch]]

    # --- Field mapping (grows across map_fields execution) ---
    field_mappings: list[FieldMapping]

    # --- Transformation extraction ---
    transformation_rules: list[TransformationRule]

    # --- Assembly ---
    table_mappings: list[TableMapping]
    final_output: FinalOutput | None

    # --- Error tracking ---
    # Pipeline never halts on individual field failure; errors are collected here
    errors: list[dict[str, Any]]
