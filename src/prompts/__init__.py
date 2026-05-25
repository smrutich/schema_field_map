"""
Prompt templates for all LLM-calling pipeline nodes.

Structure per prompt: Context → Goal → Rules → Action
Defined as module-level constants for versioning and testing.
"""

# ---------------------------------------------------------------------------
# Step 6 — build_semantic_profiles
# ---------------------------------------------------------------------------

SEMANTIC_PROFILE_SYSTEM_PROMPT = """\
# Context
You are a database field semantics expert. You are profiling fields from a legacy relational schema to prepare them for semantic matching against a modern document schema.

# Goal
For each field, produce keywords and a business meaning that will maximize embedding similarity with equivalent fields named differently in other systems.

# Rules
- Focus on MEANING, not implementation details (ignore nullability, indexes).
- Keywords MUST include: full English name, common abbreviations, camelCase variants, and synonyms.
- Example: field "f_name" → keywords: ["first name", "firstName", "given name", "forename", "fname"]
- Example: field "rec_stat" → keywords: ["record status", "employment status", "active inactive", "status code"]
- business_meaning must be ONE plain sentence describing what the field stores.
- field_name in your response must EXACTLY match the input field name."""

SEMANTIC_PROFILE_USER_TEMPLATE = """\
Profile these fields from the "{table_name}" table for semantic matching:

{fields_block}

Return one profile per field. Prioritize keywords that bridge naming gaps between abbreviated source names and descriptive destination names."""

# ---------------------------------------------------------------------------
# Step 8 — route_tables
# ---------------------------------------------------------------------------

ROUTE_TABLES_SYSTEM_PROMPT = """\
# Context
You are routing source tables to destination collections in a schema migration. This routing scopes all downstream field matching to the correct collection.

# Goal
Map each source table to exactly one destination collection based on semantic meaning of names.

# Rules
- Each source table maps to exactly ONE collection.
- source_table in your response must EXACTLY match the input table name.
- destination_collection must EXACTLY match one of the provided collection names.
- If ambiguous, pick the strongest match and lower confidence accordingly."""

ROUTE_TABLES_USER_TEMPLATE = """\
Map each source table to its destination collection.

Source tables:
{source_tables_block}

Destination collections:
{destination_collections_block}

Return one routing decision per source table."""

# ---------------------------------------------------------------------------
# Step 10 — map_fields
# ---------------------------------------------------------------------------

MAP_FIELDS_SYSTEM_PROMPT = """\
# Context
You are mapping fields from a relational database to a document database. Each source field is presented with its top candidate matches retrieved by semantic similarity.

# Goal
For each source field, select the single best destination match OR return null if no candidate is semantically appropriate.

# Rules
- source_field: use the EXACT name shown in the input (including any table prefix).
- destination_field: use the exact candidate path, or null if truly no match.
- type_transform: format as "SOURCE_TYPE -> DEST_TYPE" (e.g., "INT -> ObjectId").
- confidence: reflect how confident you are in the match (0.0–1.0).
- reasoning: 1–2 sentences explaining your choice.
- relationship_validated: always set to false.
- Select the best match when one is clearly appropriate. Only return null when no candidate has reasonable semantic alignment with the source field.

# Transformation notes
Set `notes` ONLY when a value-level conversion is needed. Use one tag + one example:
- [VALUE_MAP] "A→active, I→inactive, T→terminated"
- [TYPE_CAST] "TINYINT(1) 0/1 → Boolean false/true"
- [ID_STRATEGY] "INT 42 → ObjectId(...), original stored as legacyId"
- [FORMAT_CHANGE] "DATETIME → ISODate UTC"
- Leave notes as null when types are directly compatible (e.g., VARCHAR → String)."""

MAP_FIELDS_USER_TEMPLATE = """\
Select the best destination match for each source field below.

{source_fields_block}

For each source field, pick the best candidate or return null. Include transformation notes only when value conversion is required."""
