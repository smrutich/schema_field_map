"""
Prompt templates for all LLM-calling pipeline nodes.

All prompts are defined here as module-level constants so they can be
reviewed, tested, and versioned independently of node logic.
"""

# ---------------------------------------------------------------------------
# Step 6 — build_semantic_profiles
# ---------------------------------------------------------------------------

SEMANTIC_PROFILE_SYSTEM_PROMPT = """You are a domain expert in HR data systems and database schema design.

Your task is to generate semantic profiles for database fields. For each field, provide:
1. The business entity it belongs to (e.g., employee, department, location)
2. The concept it represents (e.g., identifier, name, status, date, salary)
3. A clear one-sentence business meaning
4. A list of synonyms and related terms that would help match this field to equivalent fields in other systems

Focus on what the field MEANS in a business context, not its technical implementation.
Be thorough with keywords — include common abbreviations, full forms, and related terms."""

SEMANTIC_PROFILE_USER_TEMPLATE = """Generate semantic profiles for the following fields from the "{table_name}" table:

{fields_block}

Return a profile for EACH field listed above. The field_name in your response must exactly match the field names provided."""

# ---------------------------------------------------------------------------
# Step 8 — route_tables
# ---------------------------------------------------------------------------

ROUTE_TABLES_SYSTEM_PROMPT = """You are a schema migration expert. Your task is to determine which source tables map to which destination collections based on their names and descriptions.

Rules:
- Each source table should map to exactly one destination collection.
- Base your decision on the semantic meaning of the table/collection names and descriptions.
- If a mapping is ambiguous, choose the most likely match and reflect lower confidence.
- Provide a brief reasoning for each routing decision."""

ROUTE_TABLES_USER_TEMPLATE = """Map each source table to its corresponding destination collection.

Source tables:
{source_tables_block}

Destination collections:
{destination_collections_block}

Return one routing decision per source table."""

# ---------------------------------------------------------------------------
# Step 10 — map_fields
# (Includes value/type transformation logic in the `notes` field — there is
# no separate derive_transformations pass.)
# ---------------------------------------------------------------------------

MAP_FIELDS_SYSTEM_PROMPT = """You are a schema migration expert specializing in relational-to-document database migrations.

For each source field, select the best destination field match from the provided candidates, or indicate no match exists.

Rules:
- Select exactly ONE candidate as the destination, or set destination_field to null if none are appropriate.
- Prefer precision over guessing — if no candidate is a strong semantic match, return null.
- For type_transform, specify the source type and destination type separated by " -> " (e.g., "INT -> ObjectId", "VARCHAR(50) -> String").
- Provide a brief reasoning explaining WHY you chose that candidate (or why none matched).
- Consider the field's constraints, comments, and semantic meaning when making your decision.

Transformation logic in `notes`:
Whenever the mapping requires a value-level transformation, describe it concisely in `notes`.
Use one of these tags as a prefix when applicable, then a short rule and a before→after example:
- [VALUE_MAP] coded values translated to readable strings (e.g., "A→active, I→inactive, T→terminated")
- [TYPE_CAST] explicit type conversion (e.g., "TINYINT(1) 0/1 → Boolean false/true")
- [ID_STRATEGY] primary key requires new ID generation, original preserved as legacyId (e.g., "INT 12345 → ObjectId(...) with legacyId=12345")
- [FORMAT_CHANGE] same value, different format (e.g., "DATETIME 2024-01-15 09:00:00 → ISODate 2024-01-15T09:00:00Z UTC")
- [NONE] direct passthrough — leave `notes` null when no transformation is needed.

Keep `notes` to a single sentence. Do not invent transforms when the source and destination types are equivalent."""

MAP_FIELDS_USER_TEMPLATE = """Select the best destination field match for each source field below.

{source_fields_block}

For each source field, the top candidate matches from the destination schema are listed. Select the best one or return null if none are appropriate. Fill `notes` with transformation logic only when a value-level conversion is required."""
