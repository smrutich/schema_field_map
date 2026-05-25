"""
parse_schemas — Load raw pseudo-JSON schema files directly into Pydantic models.

Single-pass regex parse → SourceField / DestinationField. No intermediate
JSON files. Replaces the previous two-step "convert_raw_schemas + parse_schemas"
flow which wrote and re-read disk for no benefit.

Schema input format (mappings/mysql/schema.json, mappings/mongodb/schema.json):
    Pseudo-JSON with inline SQL/BSON type tokens, constraints, and `--` comments.

LLM calls: None.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.models import DestinationField, SourceField, build_source_text_repr
from src.state import PipelineState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# MySQL field line: `"name":  TYPE[(args)]  [constraints]  [-- comment]`
_MYSQL_FIELD = re.compile(
    r'"(?P<name>[^"]+)":\s+'
    r"(?P<type>\w+(?:\([^)]*\))?)"
    r"(?P<rest>.*)"
)

# MongoDB field line: `"name":  BSONType  [-- comment]`
_MONGO_FIELD = re.compile(
    r'"(?P<name>[^"]+)":\s+(?P<type>\w+)(?P<rest>.*)'
)

_FK = re.compile(r"FK\s*->\s*(?P<target>\S+)")
_COMMENT = re.compile(r"--\s*(?P<comment>.+)$")
_REF = re.compile(r"ref\s*->\s*(\S+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_top_level_blocks(text: str) -> list[tuple[str, str]]:
    """Yield (name, inner_body) for each `"name": { … }` block at depth 0."""
    blocks: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = re.search(r'"(\w+)":\s*\{', text[pos:])
        if not m:
            break
        name = m.group(1)
        start = pos + m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        blocks.append((name, text[start : i - 1]))
        pos = i
    return blocks


# ---------------------------------------------------------------------------
# MySQL parser
# ---------------------------------------------------------------------------


def _parse_mysql_field(line: str, table: str) -> SourceField | None:
    """Parse one MySQL field line into a SourceField, or None if not a field."""
    line = line.strip().rstrip(",")
    m = _MYSQL_FIELD.match(line)
    if not m:
        return None

    name = m.group("name")
    sql_type = m.group("type")
    rest = m.group("rest").strip()

    comment: str | None = None
    cm = _COMMENT.search(rest)
    if cm:
        comment = cm.group("comment").strip()
        rest = rest[: cm.start()].strip()

    constraints: list[str] = []
    nullable = True
    upper = rest.upper()

    if "PRIMARY KEY" in upper:
        constraints.append("PRIMARY KEY")
        nullable = False
    if "NOT NULL" in upper:
        nullable = False
    if "UNIQUE" in upper:
        constraints.append("UNIQUE")

    fk = _FK.search(rest)
    if fk:
        constraints.append(f"FK -> {fk.group('target')}")

    return SourceField(
        table_name=table,
        field_name=name,
        sql_type=sql_type,
        nullable=nullable,
        constraints=constraints,
        comment=comment,
    )


def _parse_mysql(raw: str) -> list[SourceField]:
    """Parse the full MySQL pseudo-JSON file into SourceFields."""
    tables_match = re.search(r'"tables":\s*\{(.+)\}\s*\}', raw, re.DOTALL)
    if not tables_match:
        return []
    fields: list[SourceField] = []
    for table_name, body in _split_top_level_blocks(tables_match.group(1)):
        for line in body.split("\n"):
            sf = _parse_mysql_field(line, table_name)
            if sf is not None:
                fields.append(sf)
    return fields


# ---------------------------------------------------------------------------
# MongoDB parser (recursive flatten)
# ---------------------------------------------------------------------------


def _parse_mongo_block(
    body: str, collection: str, prefix: str = ""
) -> list[DestinationField]:
    """Parse a MongoDB block, recursing into nested sub-documents."""
    fields: list[DestinationField] = []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip().rstrip(",")

        # Nested sub-doc: `"name": {`
        nest = re.match(r'"([^"]+)":\s*\{', line)
        if nest:
            sub = nest.group(1)
            depth = 1
            j = i + 1
            while j < len(lines) and depth > 0:
                depth += lines[j].count("{")
                depth -= lines[j].count("}")
                j += 1
            inner = "\n".join(lines[i + 1 : j - 1])
            fields.extend(_parse_mongo_block(inner, collection, f"{prefix}{sub}."))
            i = j
            continue

        # Regular field
        m = _MONGO_FIELD.match(line)
        if m:
            comment: str | None = None
            ref_target: str | None = None
            rest = m.group("rest").strip()
            cm = _COMMENT.search(rest)
            if cm:
                comment = cm.group("comment").strip()
                ref = _REF.search(comment)
                if ref:
                    ref_target = ref.group(1)
            fields.append(
                DestinationField(
                    collection_name=collection,
                    path=f"{prefix}{m.group('name')}",
                    bson_type=m.group("type"),
                    comment=comment,
                    ref_target=ref_target,
                )
            )
        i += 1
    return fields


def _parse_mongo(raw: str) -> list[DestinationField]:
    """Parse the full MongoDB pseudo-JSON file into DestinationFields."""
    coll_match = re.search(r'"collections":\s*\{(.+)\}\s*\}', raw, re.DOTALL)
    if not coll_match:
        return []
    fields: list[DestinationField] = []
    for coll_name, body in _split_top_level_blocks(coll_match.group(1)):
        fields.extend(_parse_mongo_block(body, coll_name))
    return fields


# ---------------------------------------------------------------------------
# Enrichment — append PK/FK context to comments
# ---------------------------------------------------------------------------


def _enrich_source_fields(fields: list[SourceField]) -> list[SourceField]:
    """Append PK/FK context notes to each field's comment so the embedding/LLM
    sees the relational structure even when the original comment is sparse."""
    enriched: list[SourceField] = []
    for sf in fields:
        notes: list[str] = []
        if "PRIMARY KEY" in sf.constraints:
            notes.append(
                "Primary key — may require ID generation strategy for document store"
            )
        for c in sf.constraints:
            if not c.startswith("FK ->"):
                continue
            target = c.replace("FK ->", "").strip()
            parts = target.split(".")
            if len(parts) != 2:
                continue
            target_table = parts[0]
            if target_table == sf.table_name:
                notes.append(
                    f"Self-referential FK — references same entity ({target})"
                )
            else:
                notes.append(
                    f"FK reference to {target_table} — may map to embedded sub-document or reference ID"
                )

        if not notes:
            enriched.append(sf)
            continue

        joined = " | ".join(notes)
        new_comment = f"{sf.comment} | {joined}" if sf.comment else joined
        # model_copy doesn't re-fire the model_validator, so rebuild text_repr explicitly
        new_text_repr = build_source_text_repr(
            table_name=sf.table_name,
            field_name=sf.field_name,
            sql_type=sf.sql_type,
            nullable=sf.nullable,
            constraints=sf.constraints,
            comment=new_comment,
        )
        enriched.append(
            sf.model_copy(update={"comment": new_comment, "text_repr": new_text_repr})
        )
    return enriched


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def parse_schemas(state: PipelineState) -> PipelineState:
    """LangGraph node: load raw schemas → SourceField / DestinationField lists.

    LLM calls: None.

    Reads from disk:
        mappings/mysql/schema.json     (raw pseudo-JSON)
        mappings/mongodb/schema.json   (raw pseudo-JSON)

    Writes to state:
        source_fields: list[SourceField]   (with PK/FK enrichment applied)
        destination_fields: list[DestinationField]   (flattened to dot-notation)
        errors: list[dict]                 (initialized if absent)
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    mysql_raw = (project_root / "mappings" / "mysql" / "schema.json").read_text()
    mongo_raw = (project_root / "mappings" / "mongodb" / "schema.json").read_text()

    source_fields = _enrich_source_fields(_parse_mysql(mysql_raw))
    destination_fields = _parse_mongo(mongo_raw)

    logger.info(
        f"parse_schemas: {len(source_fields)} source fields, "
        f"{len(destination_fields)} destination fields"
    )

    state["source_fields"] = source_fields
    state["destination_fields"] = destination_fields
    state.setdefault("errors", [])
    return state
