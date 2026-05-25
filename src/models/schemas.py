"""
Schema Field Mapper — Pydantic v2 Data Contracts

All pipeline node inputs and outputs are typed against these models.
Instructor uses these as response_model targets for structured LLM output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 1.1 SourceField
# ---------------------------------------------------------------------------


class SourceField(BaseModel):
    """Represents a single parsed field from the MySQL schema."""

    table_name: str
    field_name: str
    sql_type: str  # Full type string, e.g. "DECIMAL(12,2)"
    nullable: bool
    constraints: list[str] = Field(default_factory=list)
    comment: Optional[str] = None
    text_repr: str = ""

    @model_validator(mode="after")
    def compute_text_repr(self) -> "SourceField":
        """Build text_repr from all fields for embedding."""
        if not self.text_repr:
            parts = [
                f"table:{self.table_name}",
                f"field:{self.field_name}",
                f"type:{self.sql_type}",
                f"nullable:{self.nullable}",
            ]
            if self.constraints:
                parts.append(f"constraints:{', '.join(self.constraints)}")
            if self.comment:
                parts.append(f"comment:{self.comment}")
            self.text_repr = " | ".join(parts)
        return self


# ---------------------------------------------------------------------------
# 1.2 DestinationField
# ---------------------------------------------------------------------------


class DestinationField(BaseModel):
    """Represents a single flattened field from the MongoDB schema."""

    collection_name: str
    path: str  # Dot-notation path, e.g. "fullName.firstName"
    bson_type: str
    comment: Optional[str] = None
    ref_target: Optional[str] = None  # e.g. "employees._id", "departments._id"
    text_repr: str = ""

    @model_validator(mode="after")
    def compute_text_repr(self) -> "DestinationField":
        """Build text_repr from all fields for embedding."""
        if not self.text_repr:
            parts = [
                f"collection:{self.collection_name}",
                f"path:{self.path}",
                f"type:{self.bson_type}",
            ]
            if self.ref_target:
                parts.append(f"ref:{self.ref_target}")
            if self.comment:
                parts.append(f"comment:{self.comment}")
            self.text_repr = " | ".join(parts)
        return self


# ---------------------------------------------------------------------------
# 1.3 SemanticProfile
# ---------------------------------------------------------------------------


class SemanticProfile(BaseModel):
    """LLM-generated semantic enrichment for a single field."""

    entity: str  # Top-level business entity, e.g. "employee"
    concept: str  # What the field represents, e.g. "identifier", "status"
    business_meaning: str  # One plain English sentence
    keywords: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1.4 CandidateMatch
# ---------------------------------------------------------------------------


class CandidateMatch(BaseModel):
    """Output of the retrieval stage — one per candidate per source field."""

    destination_field: str  # Dot-notation path of the candidate
    embedding_similarity: float = Field(ge=0.0, le=1.0)
    lexical_similarity: float = Field(ge=0.0, le=1.0)
    hybrid_score: float = Field(ge=0.0, le=1.0)
    retrieval_confidence_prior: str  # "HIGH", "MEDIUM", or "LOW"

    @field_validator("retrieval_confidence_prior")
    @classmethod
    def validate_confidence_prior(cls, v: str) -> str:
        allowed = {"HIGH", "MEDIUM", "LOW"}
        if v not in allowed:
            raise ValueError(f"retrieval_confidence_prior must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_hybrid_score(self) -> "CandidateMatch":
        """Enforce hybrid_score ≈ 0.8 * embedding + 0.2 * lexical (within rounding tolerance)."""
        expected = 0.8 * self.embedding_similarity + 0.2 * self.lexical_similarity
        if abs(self.hybrid_score - expected) > 1e-3:
            raise ValueError(
                f"hybrid_score ({self.hybrid_score}) must equal "
                f"0.8 * embedding_similarity ({self.embedding_similarity}) + "
                f"0.2 * lexical_similarity ({self.lexical_similarity}) = {expected:.6f}"
            )
        return self


# ---------------------------------------------------------------------------
# 1.5 FieldMapping
# ---------------------------------------------------------------------------


class FieldMapping(BaseModel):
    """Core output unit — one per source field. Instructor enforces this."""

    source_field: str
    destination_field: Optional[str] = None
    type_transform: str  # Pattern: "SOURCE_TYPE -> DEST_TYPE"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=5, max_length=500)
    notes: Optional[str] = None
    relationship_validated: bool = False

    @field_validator("destination_field")
    @classmethod
    def validate_destination_field(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            import re

            if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", v):
                raise ValueError(
                    f"destination_field '{v}' must match pattern "
                    "^[A-Za-z_][A-Za-z0-9_.]*$ or be null"
                )
        return v

    @field_validator("type_transform")
    @classmethod
    def validate_type_transform(cls, v: str) -> str:
        import re

        if not re.match(r".+\s->\s.+", v):
            raise ValueError(
                f"type_transform '{v}' must match pattern 'SOURCE_TYPE -> DEST_TYPE'"
            )
        return v


# ---------------------------------------------------------------------------
# 1.6 TableRoutingDecision
# ---------------------------------------------------------------------------


class TableRoutingDecision(BaseModel):
    """Output of the routing LLM call."""

    source_table: str
    destination_collection: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=5, max_length=500)


# ---------------------------------------------------------------------------
# 1.7 TransformationRule
# ---------------------------------------------------------------------------


class TransformationRule(BaseModel):
    """Output of the transformation extraction node."""

    source_field: str
    destination_field: str
    transform_type: str  # VALUE_MAP, TYPE_CAST, ID_STRATEGY, FORMAT_CHANGE, NONE
    transform_logic: Optional[str] = None
    example: Optional[str] = None  # Before/after value pair

    @field_validator("transform_type")
    @classmethod
    def validate_transform_type(cls, v: str) -> str:
        allowed = {"VALUE_MAP", "TYPE_CAST", "ID_STRATEGY", "FORMAT_CHANGE", "NONE"}
        if v not in allowed:
            raise ValueError(f"transform_type must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# 1.8 TableMapping
# ---------------------------------------------------------------------------


class TableMapping(BaseModel):
    """Aggregated output per source table → destination collection pair."""

    source_table: str
    destination_collection: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    field_mappings: list[FieldMapping] = Field(default_factory=list)
    unmapped_source_fields: list[str] = Field(default_factory=list)
    unmapped_destination_fields: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1.9 FinalOutput
# ---------------------------------------------------------------------------


class FinalOutput(BaseModel):
    """Top-level output document."""

    mapping_version: str = "1.0"
    source: str = "legacy_hrm (MySQL)"
    destination: str = "people_platform (MongoDB)"
    generated_at: datetime
    tables: list[TableMapping] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Response Models (used by Instructor as response_model)
# ---------------------------------------------------------------------------


class FieldProfile(BaseModel):
    """A single field's semantic profile — used in batched LLM response."""

    field_name: str = Field(description="The exact field name being profiled")
    entity: str = Field(description="Top-level business entity, e.g. employee, department")
    concept: str = Field(description="What this field represents, e.g. identifier, status, timestamp")
    business_meaning: str = Field(description="One plain English sentence explaining the field's purpose")
    keywords: list[str] = Field(
        description="List of synonyms and related terms useful for semantic matching"
    )


class TableProfileResponse(BaseModel):
    """Batched LLM response: semantic profiles for all fields in a table."""

    profiles: list[FieldProfile]
