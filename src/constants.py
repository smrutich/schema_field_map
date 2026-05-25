"""
Pipeline constants and tunable parameters.

Adjust these values to control retrieval quality, batching behavior,
and confidence thresholds without modifying node logic.
"""

# ---------------------------------------------------------------------------
# Retrieval (Step 9 — retrieve_candidates)
# ---------------------------------------------------------------------------

# Number of top candidates to return per source field
RETRIEVAL_TOP_K = 5

# Minimum hybrid score to include a candidate (below this = unmapped)
RETRIEVAL_THRESHOLD = 0.38

# Hybrid score weights
EMBEDDING_WEIGHT = 0.8
LEXICAL_WEIGHT = 0.2

# Confidence prior thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.90
CONFIDENCE_MEDIUM_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Field Mapping (Step 10 — map_fields)
# ---------------------------------------------------------------------------

# Maximum source fields per batched LLM call
MAP_FIELDS_BATCH_SIZE = 5

# Maximum field pairs per transformation LLM call
TRANSFORM_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Validation (Step 12 — validate_relationships)
# ---------------------------------------------------------------------------

# Confidence boost applied when FK alignment is confirmed
FK_CONFIDENCE_BOOST = 0.10

# Maximum confidence after boost
MAX_CONFIDENCE = 1.0

# ---------------------------------------------------------------------------
# Review Flagging (Step 14 — flag_for_review)
# ---------------------------------------------------------------------------

# Mappings below this confidence are flagged for human review
LOW_CONFIDENCE_THRESHOLD = 0.75
