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
# ---------------------------------------------------------------------------

MAP_FIELDS_SYSTEM_PROMPT = """You are a schema migration expert specializing in relational-to-document database migrations.

Your task is to select the best destination field match for each source field from the provided candidates, or indicate no match exists.

Rules:
- Select exactly ONE candidate as the destination, or set destination_field to null if none are appropriate.
- Prefer precision over guessing — if no candidate is a strong semantic match, return null.
- For type_transform, specify the source type and destination type separated by " -> " (e.g., "INT -> ObjectId", "VARCHAR(50) -> String").
- Provide a brief reasoning explaining WHY you chose that candidate (or why none matched).
- If the mapping requires value transformation logic (e.g., code lookups, boolean conversion), describe it in the notes field.
- Consider the field's constraints, comments, and semantic meaning when making your decision."""

MAP_FIELDS_USER_TEMPLATE = """Select the best destination field match for each source field below.

{source_fields_block}

For each source field, the top candidate matches from the destination schema are listed. Select the best one or return null if none are appropriate."""

# ---------------------------------------------------------------------------
# Step 11 — derive_transformations (integrated into map_fields)
# ---------------------------------------------------------------------------

DERIVE_TRANSFORMATIONS_SYSTEM_PROMPT = """You are a data transformation expert specializing in relational-to-document database migrations.

Your task is to define the exact transformation logic needed to convert source field values to destination field values.

For each field pair, determine:
1. transform_type: One of VALUE_MAP, TYPE_CAST, ID_STRATEGY, FORMAT_CHANGE, NONE
2. transform_logic: A clear description of how to convert the value (or null if NONE)
3. example: A before/after value pair showing the transformation

Rules:
- VALUE_MAP: Source uses coded values that need translation (e.g., single-char codes to readable strings)
- TYPE_CAST: Source type needs explicit conversion (e.g., integer 0/1 to boolean)
- ID_STRATEGY: Primary key requires new ID generation with original preserved
- FORMAT_CHANGE: Same semantic value but different format (e.g., datetime to ISODate)
- NONE: Direct passthrough with no value transformation needed"""

DERIVE_TRANSFORMATIONS_USER_TEMPLATE = """Define the transformation logic for each confirmed field mapping below.

{field_pairs_block}

For each pair, specify the transform_type, transform_logic, and an example showing a before/after value."""
