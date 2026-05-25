"""
Step 14 — Conditional Edge: flag_for_review

LLM calls: None

Trigger Condition:
    Any FieldMapping with confidence < 0.75 after relationship validation.

Responsibilities:
    - Collect all low-confidence mappings from state
    - Write a separate low_confidence_flags.json alongside primary output
    - Include candidate_alternatives (top-2 candidates not chosen)
    - Does NOT modify schema_mapping_output.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.state import PipelineState
from src.constants import LOW_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def should_flag_for_review(state: PipelineState) -> bool:
    """Conditional edge check: are there any low-confidence mappings?"""
    field_mappings = state.get("field_mappings", [])
    return any(fm.confidence < LOW_CONFIDENCE_THRESHOLD for fm in field_mappings)


def flag_for_review(state: PipelineState) -> PipelineState:
    """LangGraph node: Write low-confidence mappings to audit file.

    LLM calls: None

    Reads from state:
        - field_mappings: list[FieldMapping]
        - candidate_matches: dict[str, list[CandidateMatch]]
        - source_fields: list[SourceField]

    Writes:
        - mappings/output/low_confidence_flags.json (file only — does not modify state)
    """
    field_mappings = state["field_mappings"]
    candidate_matches = state.get("candidate_matches", {})
    source_fields = state["source_fields"]

    # Build field_name -> table lookup
    # Handle both "field_name" and "table.field_name" formats from LLM
    field_to_table: dict[str, str] = {}
    for sf in source_fields:
        field_to_table[sf.field_name] = sf.table_name
        field_to_table[f"{sf.table_name}.{sf.field_name}"] = sf.table_name

    # Collect low-confidence mappings
    flags: list[dict[str, Any]] = []

    for fm in field_mappings:
        if fm.confidence >= LOW_CONFIDENCE_THRESHOLD:
            continue

        # Find candidate alternatives (top-2 not chosen)
        table_name = field_to_table.get(fm.source_field, "unknown")
        key = f"{table_name}.{fm.source_field}"
        candidates = candidate_matches.get(key, [])

        # Alternatives are candidates that weren't selected
        alternatives = []
        for c in candidates:
            if c.destination_field != fm.destination_field:
                alternatives.append({
                    "destination_field": c.destination_field,
                    "hybrid_score": c.hybrid_score,
                    "retrieval_confidence_prior": c.retrieval_confidence_prior,
                })

        flags.append({
            "source_field": fm.source_field,
            "source_table": table_name,
            "destination_field": fm.destination_field,
            "confidence": fm.confidence,
            "reasoning": fm.reasoning,
            "candidate_alternatives": alternatives[:2],
        })

    # Write to file
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "mappings" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "low_confidence_flags.json"

    output_path.write_text(json.dumps(flags, indent=2) + "\n")

    logger.info(
        f"flag_for_review: {len(flags)} low-confidence mappings "
        f"(< {LOW_CONFIDENCE_THRESHOLD}) written to {output_path}"
    )

    return state
