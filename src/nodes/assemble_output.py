"""
Step 13 — Node: assemble_output

LLM calls: None

Responsibilities:
    1. Group FieldMapping objects by source table (using routing decisions)
    2. Compute TableMapping.confidence as mean of child confidences
    3. Identify unmapped_source_fields and unmapped_destination_fields
    4. Build FinalOutput with ISO 8601 generated_at timestamp
    5. Serialize to schema_mapping_output.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import FieldMapping, FinalOutput, TableMapping
from src.state import PipelineState

logger = logging.getLogger(__name__)


def assemble_output(state: PipelineState) -> PipelineState:
    """LangGraph node: Assemble final output from all pipeline results.

    LLM calls: None

    Reads from state:
        - field_mappings: list[FieldMapping]
        - table_routing: list[TableRoutingDecision]
        - source_fields: list[SourceField]
        - destination_fields: list[DestinationField]

    Writes to state:
        - table_mappings: list[TableMapping]
        - final_output: FinalOutput
    """
    field_mappings = state["field_mappings"]
    table_routing = state.get("table_routing", [])
    source_fields = state["source_fields"]
    destination_fields = state["destination_fields"]
    errors = state.get("errors", [])

    # --- 1. Group by source table using routing ---
    routing_map: dict[str, str] = {}
    routing_reasoning: dict[str, str] = {}
    for rd in table_routing:
        routing_map[rd.source_table] = rd.destination_collection
        routing_reasoning[rd.source_table] = rd.reasoning

    # Build source field -> table lookup
    # Handle both "field_name" and "table.field_name" formats from LLM
    field_to_table: dict[str, str] = {}
    for sf in source_fields:
        field_to_table[sf.field_name] = sf.table_name
        field_to_table[f"{sf.table_name}.{sf.field_name}"] = sf.table_name

    # Group mappings by source table
    table_groups: dict[str, list[FieldMapping]] = {}
    for fm in field_mappings:
        table = field_to_table.get(fm.source_field, "unknown")
        table_groups.setdefault(table, []).append(fm)

    # --- 3-5. Build TableMapping for each table ---
    # Collect all source field names per table
    source_fields_by_table: dict[str, set[str]] = {}
    for sf in source_fields:
        source_fields_by_table.setdefault(sf.table_name, set()).add(sf.field_name)

    # Collect all destination paths per collection
    dest_fields_by_collection: dict[str, set[str]] = {}
    for df in destination_fields:
        dest_fields_by_collection.setdefault(df.collection_name, set()).add(df.path)

    table_mappings: list[TableMapping] = []

    for source_table, dest_collection in routing_map.items():
        mappings = table_groups.get(source_table, [])

        # Compute mean confidence
        confidences = [fm.confidence for fm in mappings]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Identify unmapped source fields
        # Handle LLM returning "table.field" or just "field" in source_field
        mapped_source: set[str] = set()
        for fm in mappings:
            if fm.destination_field:
                # Strip table prefix if present
                sf_name = fm.source_field.split(".")[-1] if "." in fm.source_field else fm.source_field
                mapped_source.add(sf_name)
        all_source = source_fields_by_table.get(source_table, set())
        unmapped_source = sorted(all_source - mapped_source)

        # Identify unmapped destination fields
        mapped_dest = {fm.destination_field for fm in mappings if fm.destination_field}
        all_dest = dest_fields_by_collection.get(dest_collection, set())
        unmapped_dest = sorted(all_dest - mapped_dest)

        reasoning = routing_reasoning.get(source_table, "")

        tm = TableMapping(
            source_table=source_table,
            destination_collection=dest_collection,
            confidence=round(mean_confidence, 4),
            reasoning=reasoning,
            field_mappings=mappings,
            unmapped_source_fields=unmapped_source,
            unmapped_destination_fields=unmapped_dest,
        )
        table_mappings.append(tm)

    # --- 6. Build FinalOutput ---
    final_output = FinalOutput(
        generated_at=datetime.now(timezone.utc),
        tables=table_mappings,
    )

    # Validate the full output structure
    FinalOutput.model_validate(final_output.model_dump())

    # --- 7. Serialize to JSON ---
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "mappings" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "schema_mapping_output.json"

    output_json = final_output.model_dump(mode="json")
    output_path.write_text(json.dumps(output_json, indent=2) + "\n")

    logger.info(f"Final output written to {output_path}")

    # --- Write to state ---
    state["table_mappings"] = table_mappings
    state["final_output"] = final_output
    state["errors"] = errors

    logger.info(
        f"assemble_output complete: "
        f"{len(table_mappings)} table mappings, "
        f"{sum(len(tm.field_mappings) for tm in table_mappings)} total field mappings"
    )

    return state
