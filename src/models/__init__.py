"""Pydantic data contracts for the Schema Field Mapper pipeline."""

from src.models.schemas import (
    CandidateMatch,
    DestinationField,
    FieldMapping,
    FieldMappingProposal,
    FinalOutput,
    SemanticProfile,
    SourceField,
    TableMapping,
    TableRoutingDecision,
    build_destination_text_repr,
    build_source_text_repr,
)

__all__ = [
    "CandidateMatch",
    "DestinationField",
    "FieldMapping",
    "FieldMappingProposal",
    "FinalOutput",
    "SemanticProfile",
    "SourceField",
    "TableMapping",
    "TableRoutingDecision",
    "build_destination_text_repr",
    "build_source_text_repr",
]
