"""
Step 1.11 & Step 5: Schema Format Conversion + parse_schemas Node

Step 1.11 — convert_raw_schemas:
    Reads the raw pseudo-JSON schema files (MySQL and MongoDB) that use inline
    comments and non-standard formatting, parses them with regex, and outputs
    clean valid JSON files compatible with the Pydantic models.

Step 5 — parse_schemas (LangGraph node):
    Loads schemas (raw or clean JSON), hydrates SourceField and DestinationField
    Pydantic models, handles special cases, and writes into PipelineState.

LLM calls: None (fully deterministic).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.models import DestinationField, SourceField
from src.state import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MySQL Parser
# ---------------------------------------------------------------------------

# Matches lines like:
#   "emp_id":        INT           PRIMARY KEY
#   "emp_cd":        VARCHAR(20)    UNIQUE NOT NULL    -- human-readable employee code
#   "dept_id":       INT            FK -> dept_info.dept_id
_MYSQL_FIELD_RE = re.compile(
    r'"(?P<field_name>[^"]+)":\s+'
    r"(?P<sql_type>\w+(?:\([^)]*\))?)"  # type with optional (precision)
    r"(?P<rest>.*)"  # everything after type
)

_FK_RE = re.compile(r"FK\s*->\s*(?P<target>\S+)")
_COMMENT_RE = re.compile(r"--\s*(?P<comment>.+)$")


def _parse_mysql_field(line: str) -> dict[str, Any] | None:
    """Parse a single MySQL field line into a structured dict."""
    line = line.strip().rstrip(",")
    match = _MYSQL_FIELD_RE.match(line)
    if not match:
        return None

    field_name = match.group("field_name")
    sql_type = match.group("sql_type")
    rest = match.group("rest").strip()

    # Extract comment (everything after --)
    comment = None
    comment_match = _COMMENT_RE.search(rest)
    if comment_match:
        comment = comment_match.group("comment").strip()
        rest = rest[: comment_match.start()].strip()

    # Extract constraints from remaining text
    constraints: list[str] = []
    nullable = True  # Default: nullable unless NOT NULL present

    if "PRIMARY KEY" in rest.upper():
        constraints.append("PRIMARY KEY")
        rest = re.sub(r"PRIMARY\s+KEY", "", rest, flags=re.IGNORECASE).strip()
        nullable = False  # PKs are implicitly NOT NULL

    if "NOT NULL" in rest.upper():
        nullable = False
        rest = re.sub(r"NOT\s+NULL", "", rest, flags=re.IGNORECASE).strip()

    if "UNIQUE" in rest.upper():
        constraints.append("UNIQUE")
        rest = re.sub(r"UNIQUE", "", rest, flags=re.IGNORECASE).strip()

    fk_match = _FK_RE.search(rest)
    if fk_match:
        constraints.append(f"FK -> {fk_match.group('target')}")
        rest = rest[: fk_match.start()] + rest[fk_match.end() :]
        rest = rest.strip()

    return {
        "field_name": field_name,
        "sql_type": sql_type,
        "nullable": nullable,
        "constraints": constraints,
        "comment": comment,
    }


def parse_mysql_schema(raw_text: str) -> dict[str, Any]:
    """Parse raw MySQL schema text into a clean structured dict.

    Returns:
        {
            "database": "legacy_hrm",
            "type": "MySQL (Relational)",
            "tables": {
                "table_name": [
                    {"field_name": ..., "sql_type": ..., "nullable": ...,
                     "constraints": [...], "comment": ...},
                    ...
                ]
            }
        }
    """
    # Extract database name and type from the raw text
    db_match = re.search(r'"database":\s*"([^"]+)"', raw_text)
    type_match = re.search(r'"type":\s*"([^"]+)"', raw_text)

    database = db_match.group(1) if db_match else "unknown"
    db_type = type_match.group(1) if type_match else "unknown"

    # Find all table blocks
    tables: dict[str, list[dict[str, Any]]] = {}

    # Match table name and its content block
    table_pattern = re.compile(
        r'"(\w+)":\s*\{([^}]*)\}', re.DOTALL
    )

    # Find the "tables" section first
    tables_section_match = re.search(
        r'"tables":\s*\{(.+)\}\s*\}', raw_text, re.DOTALL
    )
    if not tables_section_match:
        return {"database": database, "type": db_type, "tables": {}}

    tables_section = tables_section_match.group(1)

    # Find individual table blocks within the tables section
    # We need to handle nested braces carefully
    current_pos = 0
    while current_pos < len(tables_section):
        # Find next table name
        table_name_match = re.search(
            r'"(\w+)":\s*\{', tables_section[current_pos:]
        )
        if not table_name_match:
            break

        table_name = table_name_match.group(1)
        block_start = current_pos + table_name_match.end()

        # Find matching closing brace
        brace_count = 1
        pos = block_start
        while pos < len(tables_section) and brace_count > 0:
            if tables_section[pos] == "{":
                brace_count += 1
            elif tables_section[pos] == "}":
                brace_count -= 1
            pos += 1

        block_content = tables_section[block_start : pos - 1]
        current_pos = pos

        # Parse fields from this table block
        fields = []
        for line in block_content.split("\n"):
            field = _parse_mysql_field(line)
            if field:
                fields.append(field)

        if fields:
            tables[table_name] = fields

    return {"database": database, "type": db_type, "tables": tables}


# ---------------------------------------------------------------------------
# MongoDB Parser
# ---------------------------------------------------------------------------

_MONGO_FIELD_RE = re.compile(
    r'"(?P<field_name>[^"]+)":\s+'
    r"(?P<bson_type>\w+)"  # ObjectId, String, ISODate, Number, Boolean
    r"(?P<rest>.*)"
)


def _parse_mongo_collection(
    block: str, collection_name: str, path_prefix: str = ""
) -> list[dict[str, Any]]:
    """Recursively parse a MongoDB collection/sub-document block into flat fields.

    Handles nested sub-documents by flattening to dot-notation.
    """
    fields: list[dict[str, Any]] = []
    lines = block.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip().rstrip(",")

        # Check if this is a nested sub-document: "fieldName": {
        nested_match = re.match(r'"([^"]+)":\s*\{', line)
        if nested_match:
            sub_name = nested_match.group(1)
            sub_path = f"{path_prefix}{sub_name}" if path_prefix else sub_name

            # Collect the nested block content
            brace_count = 1
            block_start = i + 1
            j = block_start
            while j < len(lines) and brace_count > 0:
                if "{" in lines[j]:
                    brace_count += lines[j].count("{")
                if "}" in lines[j]:
                    brace_count -= lines[j].count("}")
                j += 1

            nested_block = "\n".join(lines[block_start : j - 1])
            # Recurse into the nested block
            nested_fields = _parse_mongo_collection(
                nested_block, collection_name, path_prefix=f"{sub_path}."
            )
            fields.extend(nested_fields)
            i = j
            continue

        # Regular field line
        field_match = _MONGO_FIELD_RE.match(line)
        if field_match:
            field_name = field_match.group("field_name")
            bson_type = field_match.group("bson_type")
            rest = field_match.group("rest").strip()

            # Extract comment
            comment = None
            comment_match = _COMMENT_RE.search(rest)
            if comment_match:
                comment = comment_match.group("comment").strip()

            # Extract ref_target from comment
            ref_target = None
            if comment:
                ref_match = re.search(r"ref\s*->\s*(\S+)", comment)
                if ref_match:
                    ref_target = ref_match.group(1)

            full_path = f"{path_prefix}{field_name}"

            fields.append({
                "collection_name": collection_name,
                "path": full_path,
                "bson_type": bson_type,
                "comment": comment,
                "ref_target": ref_target,
            })

        i += 1

    return fields


def parse_mongodb_schema(raw_text: str) -> dict[str, Any]:
    """Parse raw MongoDB schema text into a clean structured dict.

    Returns:
        {
            "database": "people_platform",
            "type": "MongoDB (Document)",
            "collections": {
                "collection_name": [
                    {"collection_name": ..., "path": ..., "bson_type": ...,
                     "comment": ..., "ref_target": ...},
                    ...
                ]
            }
        }
    """
    db_match = re.search(r'"database":\s*"([^"]+)"', raw_text)
    type_match = re.search(r'"type":\s*"([^"]+)"', raw_text)

    database = db_match.group(1) if db_match else "unknown"
    db_type = type_match.group(1) if type_match else "unknown"

    collections: dict[str, list[dict[str, Any]]] = {}

    # Find the "collections" section
    collections_match = re.search(
        r'"collections":\s*\{(.+)\}\s*\}', raw_text, re.DOTALL
    )
    if not collections_match:
        return {"database": database, "type": db_type, "collections": {}}

    collections_section = collections_match.group(1)

    # Find individual collection blocks (top-level only)
    current_pos = 0
    while current_pos < len(collections_section):
        coll_name_match = re.search(
            r'"(\w+)":\s*\{', collections_section[current_pos:]
        )
        if not coll_name_match:
            break

        coll_name = coll_name_match.group(1)
        block_start = current_pos + coll_name_match.end()

        # Find matching closing brace (accounting for nesting)
        brace_count = 1
        pos = block_start
        while pos < len(collections_section) and brace_count > 0:
            if collections_section[pos] == "{":
                brace_count += 1
            elif collections_section[pos] == "}":
                brace_count -= 1
            pos += 1

        block_content = collections_section[block_start : pos - 1]
        current_pos = pos

        # Parse fields from this collection
        fields = _parse_mongo_collection(block_content, coll_name)
        if fields:
            collections[coll_name] = fields

    return {"database": database, "type": db_type, "collections": collections}


# ---------------------------------------------------------------------------
# Main Converter
# ---------------------------------------------------------------------------


def convert_raw_schemas(
    mysql_path: str | Path,
    mongodb_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert raw schema files to clean JSON and write to output directory.

    Args:
        mysql_path: Path to raw MySQL schema file
        mongodb_path: Path to raw MongoDB schema file
        output_dir: Directory to write clean JSON files. If None, writes
                    to the same directory as each input file.

    Returns:
        Tuple of (mysql_clean_dict, mongodb_clean_dict)
    """
    mysql_path = Path(mysql_path)
    mongodb_path = Path(mongodb_path)

    # Parse raw files
    mysql_raw = mysql_path.read_text()
    mongodb_raw = mongodb_path.read_text()

    mysql_clean = parse_mysql_schema(mysql_raw)
    mongodb_clean = parse_mongodb_schema(mongodb_raw)

    # Determine output paths
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        mysql_out = out / "mysql_schema_clean.json"
        mongodb_out = out / "mongodb_schema_clean.json"
    else:
        mysql_out = mysql_path.parent / "schema_clean.json"
        mongodb_out = mongodb_path.parent / "schema_clean.json"

    # Write clean JSON
    mysql_out.write_text(json.dumps(mysql_clean, indent=2) + "\n")
    mongodb_out.write_text(json.dumps(mongodb_clean, indent=2) + "\n")

    print(f"MySQL schema: {len(mysql_clean.get('tables', {}))} tables parsed")
    for table, fields in mysql_clean.get("tables", {}).items():
        print(f"  {table}: {len(fields)} fields")

    print(f"MongoDB schema: {len(mongodb_clean.get('collections', {}))} collections parsed")
    for coll, fields in mongodb_clean.get("collections", {}).items():
        print(f"  {coll}: {len(fields)} fields")

    print(f"\nClean files written:")
    print(f"  {mysql_out}")
    print(f"  {mongodb_out}")

    return mysql_clean, mongodb_clean


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    mysql_path = project_root / "mappings" / "mysql" / "schema.json"
    mongodb_path = project_root / "mappings" / "mongodb" / "schema.json"

    convert_raw_schemas(mysql_path, mongodb_path)


# ---------------------------------------------------------------------------
# Step 5 — parse_schemas LangGraph Node
# ---------------------------------------------------------------------------


def _enrich_from_metadata(source_fields: list[SourceField]) -> list[SourceField]:
    """Dynamically enrich source fields based on parsed metadata.

    Derives additional context from constraints and field properties:
    - Primary keys: notes ID generation strategy requirement
    - Foreign keys: notes reference relationship and target
    - Self-referential FKs: notes same-collection reference

    No hardcoded field names — works generically on any schema.
    """
    # Build a lookup of table -> primary key field names
    pk_fields: dict[str, str] = {}
    for sf in source_fields:
        if "PRIMARY KEY" in sf.constraints:
            pk_fields[sf.table_name] = sf.field_name

    enriched = []
    for sf in source_fields:
        notes: list[str] = []

        # Primary key: flag for ID generation strategy
        if "PRIMARY KEY" in sf.constraints:
            notes.append("Primary key — may require ID generation strategy for document store")

        # Foreign key analysis
        for constraint in sf.constraints:
            if constraint.startswith("FK ->"):
                fk_target = constraint.replace("FK ->", "").strip()
                # Parse "table.field" from FK target
                parts = fk_target.split(".")
                if len(parts) == 2:
                    target_table, target_field = parts
                    # Self-referential FK (points to own table)
                    if target_table == sf.table_name:
                        notes.append(f"Self-referential FK — references same entity ({fk_target})")
                    else:
                        notes.append(f"FK reference to {target_table} — may map to embedded sub-document or reference ID")

        # Apply notes if any were generated
        if notes:
            note_str = " | ".join(notes)
            if sf.comment:
                new_comment = f"{sf.comment} | {note_str}"
            else:
                new_comment = note_str

            sf = SourceField(
                table_name=sf.table_name,
                field_name=sf.field_name,
                sql_type=sf.sql_type,
                nullable=sf.nullable,
                constraints=sf.constraints,
                comment=new_comment,
            )

        enriched.append(sf)
    return enriched


def parse_schemas(state: PipelineState) -> PipelineState:
    """LangGraph node: Parse raw schemas into SourceField and DestinationField lists.

    Reads from clean JSON files (produced by convert_raw_schemas) or
    parses raw schema files directly. Handles special cases for fields
    that require additional context for mapping.

    LLM calls: None

    Writes to state:
        - source_fields: list[SourceField]
        - destination_fields: list[DestinationField]
        - errors: list[dict] (initialized if not present)
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    mysql_clean_path = project_root / "mappings" / "mysql" / "schema_clean.json"
    mongodb_clean_path = project_root / "mappings" / "mongodb" / "schema_clean.json"

    # If clean files don't exist, generate them from raw schemas
    if not mysql_clean_path.exists() or not mongodb_clean_path.exists():
        logger.info("Clean schema files not found — running convert_raw_schemas")
        mysql_raw_path = project_root / "mappings" / "mysql" / "schema.json"
        mongodb_raw_path = project_root / "mappings" / "mongodb" / "schema.json"
        convert_raw_schemas(mysql_raw_path, mongodb_raw_path)

    # Load clean JSON
    with open(mysql_clean_path) as f:
        mysql_data = json.load(f)
    with open(mongodb_clean_path) as f:
        mongo_data = json.load(f)

    # --- Hydrate SourceField models ---
    source_fields: list[SourceField] = []
    for table_name, fields in mysql_data["tables"].items():
        for field_dict in fields:
            sf = SourceField(table_name=table_name, **field_dict)
            source_fields.append(sf)

    # Enrich fields dynamically based on parsed metadata (PKs, FKs, self-refs)
    source_fields = _enrich_from_metadata(source_fields)

    logger.info(
        f"Parsed {len(source_fields)} source fields from "
        f"{len(mysql_data['tables'])} tables"
    )

    # --- Hydrate DestinationField models ---
    destination_fields: list[DestinationField] = []
    for coll_name, fields in mongo_data["collections"].items():
        for field_dict in fields:
            df = DestinationField(**field_dict)
            destination_fields.append(df)

    logger.info(
        f"Parsed {len(destination_fields)} destination fields from "
        f"{len(mongo_data['collections'])} collections"
    )

    # --- Write to state ---
    state["source_fields"] = source_fields
    state["destination_fields"] = destination_fields

    # Initialize errors list if not present
    if "errors" not in state:
        state["errors"] = []

    return state
