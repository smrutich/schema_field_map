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
# text_repr builders — single source of truth for embedding-ready strings.
# Used by model_validators on construction AND by callers who need to rebuild
# after a model_copy() update (which does NOT re-fire model_validators).
# ---------------------------------------------------------------------------


def build_source_text_repr(
    *,
    table_name: str,
    field_name: str,
    sql_type: str,
    nullable: bool,
    constraints: list[str],
    comment: str | None,
    profile: "SemanticProfile | None" = None,
) -> str:
    """Build the embedding-ready string for a source field.

    Pass `profile` to enrich with LLM-derived business meaning + keywords.
    """
    parts = [
        f"table:{table_name}",
        f"field:{field_name}",
        f"type:{sql_type}",
        f"nullable:{nullable}",
    ]
    if constraints:
        parts.append(f"constraints:{', '.join(constraints)}")
    if comment:
        parts.append(f"comment:{comment}")
    if profile is not None:
        parts.append(f"meaning:{profile.business_meaning}")
        if profile.keywords:
            parts.append(f"keywords:{', '.join(profile.keywords)}")
    return " | ".join(parts)


def build_destination_text_repr(
    *,
    collection_name: str,
    path: str,
    bson_type: str,
    ref_target: str | None,
    comment: str | None,
) -> str:
    """Build the embedding-ready string for a destination field."""
    parts = [
        f"collection:{collection_name}",
        f"path:{path}",
        f"type:{bson_type}",
    ]
    if ref_target:
        parts.append(f"ref:{ref_target}")
    if comment:
        parts.append(f"comment:{comment}")
    return " | ".join(parts)


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
    def _default_text_repr(self) -> "SourceField":
        if not self.text_repr:
            self.text_repr = build_source_text_repr(
                table_name=self.table_name,
                field_name=self.field_name,
                sql_type=self.sql_type,
                nullable=self.nullable,
                constraints=self.constraints,
                comment=self.comment,
            )
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
    def _default_text_repr(self) -> "DestinationField":
        if not self.text_repr:
            self.text_repr = build_destination_text_repr(
                collection_name=self.collection_name,
                path=self.path,
                bson_type=self.bson_type,
                ref_target=self.ref_target,
                comment=self.comment,
            )
        return self


# ---------------------------------------------------------------------------
# 1.3 SemanticProfile
# (Used as both the in-memory enrichment model AND the LLM response model;
# Instructor returns list[SemanticProfile] directly — no separate wrapper.)
# ---------------------------------------------------------------------------


class SemanticProfile(BaseModel):
    """LLM-generated semantic enrichment for a single field."""

    field_name: str = Field(description="The exact field name being profiled")
    entity: str = Field(description="Top-level business entity, e.g. employee, department")
    concept: str = Field(description="What this field represents, e.g. identifier, status, timestamp")
    business_meaning: str = Field(description="One plain English sentence explaining the field's purpose")
    keywords: list[str] = Field(
        default_factory=list,
        description="Synonyms and related terms useful for semantic matching",
    )


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


# ---------------------------------------------------------------------------
# 1.5 FieldMapping
#
# Split into two layers:
#   - FieldMappingProposal: what the LLM produces (no relationship_validated)
#   - FieldMapping: what the pipeline state stores (adds relationship_validated,
#     which is set ONLY by the rule-based validate_relationships node)
#
# Excluding `relationship_validated` from the LLM-facing schema prevents the
# model from hallucinating True on non-FK fields.
# ---------------------------------------------------------------------------


class FieldMappingProposal(BaseModel):
    """LLM response for one source field — does NOT include relationship_validated.

    Instructor uses this as the response_model in `map_fields`. The LLM can
    decide source→destination, type transform, confidence, reasoning, and notes
    — but it cannot influence FK validation status.
    """

    source_field: str
    destination_field: Optional[str] = None
    type_transform: str  # Pattern: "SOURCE_TYPE -> DEST_TYPE"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=5, max_length=500)
    notes: Optional[str] = None

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


class FieldMapping(FieldMappingProposal):
    """Finalized mapping — adds the rule-based relationship_validated flag.

    `relationship_validated` is set to True ONLY by validate_relationships
    when the FK metadata + table routing align with the destination schema.
    The LLM never sees this field.
    """

    relationship_validated: bool = False


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
# 1.7 TableMapping
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
# 1.8 FinalOutput
# ---------------------------------------------------------------------------


class FinalOutput(BaseModel):
    """Top-level output document."""

    mapping_version: str = "1.0"
    source: str = "legacy_hrm (MySQL)"
    destination: str = "people_platform (MongoDB)"
    generated_at: datetime
    tables: list[TableMapping] = Field(default_factory=list)
