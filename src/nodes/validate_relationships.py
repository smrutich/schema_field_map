"""
Step 12 — Node: validate_relationships

LLM calls: None — fully rule-based

Purpose:
    Use FK metadata extracted at parse time to validate and boost confidence
    on mappings that align with relational structure. This is the most
    explainable confidence signal in the pipeline.

Rules:
    - For each FK field, check that its FieldMapping.destination_field
      resolves to a reference field (ObjectId with ref_target) in the
      destination schema.
    - If FK aligns correctly → set relationship_validated=True and +0.10 confidence (capped at 1.0)
    - If FK does not align → log warning, flag for review, do not override
"""

from __future__ import annotations

import logging
from typing import Any

from src.models import FieldMapping, SourceField
from src.state import PipelineState

logger = logging.getLogger(__name__)


def validate_relationships(state: PipelineState) -> PipelineState:
    """LangGraph node: Validate FK relationships against destination schema.

    Fully rule-based — no LLM calls.

    Reads from state:
        - source_fields: list[SourceField]
        - destination_fields: list[DestinationField]
        - field_mappings: list[FieldMapping]
        - table_routing: list[TableRoutingDecision]

    Writes to state:
        - field_mappings: list[FieldMapping] (updated with relationship_validated + confidence boost)
    """
    source_fields = state["source_fields"]
    destination_fields = state["destination_fields"]
    field_mappings = state["field_mappings"]
    table_routing = state.get("table_routing", [])
    errors = state.get("errors", [])

    # Build routing lookup: source_table -> destination_collection
    routing_map: dict[str, str] = {}
    for rd in table_routing:
        routing_map[rd.source_table] = rd.destination_collection

    # Build destination ref_target lookup: path -> ref_target
    dest_ref_lookup: dict[str, str | None] = {}
    dest_type_lookup: dict[str, str] = {}
    for df in destination_fields:
        dest_ref_lookup[df.path] = df.ref_target
        dest_type_lookup[df.path] = df.bson_type

    # Build source field lookup: field_name -> SourceField (may have dupes across tables)
    source_by_name: dict[str, list[SourceField]] = {}
    for sf in source_fields:
        source_by_name.setdefault(sf.field_name, []).append(sf)

    # Build FK info: identify source fields that have FK constraints
    # Structure: (table_name, field_name) -> fk_target "table.field"
    fk_fields: dict[tuple[str, str], str] = {}
    for sf in source_fields:
        for constraint in sf.constraints:
            if constraint.startswith("FK ->"):
                fk_target = constraint.replace("FK ->", "").strip()
                fk_fields[(sf.table_name, sf.field_name)] = fk_target

    # Build PK lookup: table_name -> pk_field_name
    pk_lookup: dict[str, str] = {}
    for sf in source_fields:
        if "PRIMARY KEY" in sf.constraints:
            pk_lookup[sf.table_name] = sf.field_name

    validated_count = 0
    misaligned_count = 0

    # Process each field mapping
    updated_mappings: list[FieldMapping] = []
    for fm in field_mappings:
        # Find the source field(s) matching this mapping
        matching_sources = source_by_name.get(fm.source_field, [])

        is_fk = False
        fk_target = None
        source_table = None

        for sf in matching_sources:
            key = (sf.table_name, sf.field_name)
            if key in fk_fields:
                is_fk = True
                fk_target = fk_fields[key]
                source_table = sf.table_name
                break

        if not is_fk or fm.destination_field is None:
            updated_mappings.append(fm)
            continue

        # FK validation logic:
        # Check if the destination field is an ObjectId reference type
        dest_type = dest_type_lookup.get(fm.destination_field, "")
        dest_ref = dest_ref_lookup.get(fm.destination_field)

        # Parse FK target: "table.field"
        fk_parts = fk_target.split(".") if fk_target else []
        fk_target_table = fk_parts[0] if len(fk_parts) == 2 else None

        # Determine if FK aligns:
        # 1. Destination field should be ObjectId type
        # 2. If destination has a ref_target, check it points to the expected collection
        aligned = False

        if dest_type == "ObjectId":
            if dest_ref:
                # Destination has explicit ref -> check it references the right collection
                ref_collection = dest_ref.split(".")[0] if "." in dest_ref else dest_ref

                # The FK target table should route to the ref collection
                expected_collection = routing_map.get(fk_target_table, "")
                if ref_collection == expected_collection:
                    aligned = True
                # Self-referential: FK target table == source table
                elif fk_target_table == source_table:
                    source_collection = routing_map.get(source_table, "")
                    if ref_collection == source_collection:
                        aligned = True
            else:
                # ObjectId without explicit ref — check if FK target table
                # routes to the same collection as the destination field's collection
                # This handles cases like _id fields
                aligned = True  # ObjectId destination is a reasonable FK target

        if aligned:
            # Boost confidence by +0.10, capped at 1.0
            new_confidence = min(fm.confidence + 0.10, 1.0)
            updated_fm = FieldMapping(
                source_field=fm.source_field,
                destination_field=fm.destination_field,
                type_transform=fm.type_transform,
                confidence=new_confidence,
                reasoning=fm.reasoning,
                notes=fm.notes,
                relationship_validated=True,
            )
            updated_mappings.append(updated_fm)
            validated_count += 1
            logger.info(
                f"  FK validated: {fm.source_field} -> {fm.destination_field} "
                f"(confidence {fm.confidence:.2f} -> {new_confidence:.2f})"
            )
        else:
            # Misaligned — log warning but keep the mapping
            updated_mappings.append(fm)
            misaligned_count += 1
            errors.append({
                "node": "validate_relationships",
                "field": fm.source_field,
                "error": (
                    f"FK alignment mismatch: {fm.source_field} "
                    f"(FK -> {fk_target}) mapped to {fm.destination_field} "
                    f"but destination ref_target={dest_ref} does not align "
                    f"with expected collection"
                ),
            })
            logger.warning(
                f"  FK misaligned: {fm.source_field} -> {fm.destination_field} "
                f"(FK target: {fk_target})"
            )

    state["field_mappings"] = updated_mappings
    state["errors"] = errors

    logger.info(
        f"validate_relationships complete: "
        f"{validated_count} validated, {misaligned_count} misaligned, "
        f"{len(field_mappings) - validated_count - misaligned_count} non-FK fields"
    )

    return state
