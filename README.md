# Schema Field Mapper

AI-powered pipeline that automatically maps fields between heterogeneous database schemas using hybrid retrieval (embeddings + lexical similarity) and structured LLM reasoning.

---

## Problem

Migrating from a legacy relational database (MySQL) to a modern document store (MongoDB) requires mapping every source field to its destination equivalent. Legacy schemas use cryptic abbreviated names (`rec_stat`, `f_name`, `dept_head_id`) while modern schemas use descriptive nested paths (`employment.status`, `fullName.firstName`, `headEmployeeId`).

Manual mapping is:
- **Error-prone** — abbreviated names are ambiguous without domain context
- **Expensive** — scales linearly with schema size
- **Non-reproducible** — different engineers produce different mappings

---

## Solution Overview

A stateful LangGraph pipeline that breaks the mapping problem into discrete, verifiable steps:

1. **Parse** — Extract typed field metadata from raw schemas
2. **Enrich** — LLM generates business meaning + keywords for cryptic field names
3. **Embed** — Convert enriched descriptions to vectors for similarity search
4. **Route** — LLM maps source tables to destination collections (scopes retrieval)
5. **Retrieve** — Hybrid search (embedding + lexical) finds top-k candidates per field
6. **Map** — LLM selects best match from candidates with structured reasoning
7. **Validate** — Rule-based FK alignment check boosts confidence on correct mappings
8. **Assemble** — Produce validated JSON output with unmapped field tracking
9. **Flag** — Separate audit file for low-confidence mappings requiring human review

### Key Constraint

The pipeline never passes both full schemas to an LLM in a single prompt. Each LLM call sees only scoped, minimal context — this ensures scalability and avoids token limits.

---

## Technology Stack

| Component | Role |
|-----------|------|
| **LangGraph** | Stateful pipeline orchestration with conditional edges |
| **Instructor + Pydantic v2** | Structured LLM output with automatic retry on validation failure |
| **SentenceTransformers** | Local embedding model (`all-MiniLM-L6-v2`, 384-dim) |
| **RapidFuzz** | Lexical similarity (token sort ratio) for hybrid scoring |
| **NumPy** | Cosine similarity via normalized dot product |
| **OpenAI / Groq / Azure / Snowflake Cortex / Ollama** | LLM backend (auto-detected) |

---

## Architecture

```
START → parse_schemas → build_semantic_profiles → embed_destination_fields
      → route_tables → retrieve_candidates → map_fields
      → [conditional: mappings exist?]
          ├── YES → validate_relationships → assemble_output
          │         → [conditional: low confidence?]
          │             ├── YES → flag_for_review → END
          │             └── NO → END
          └── NO → error_handler → END
```

---

## Pipeline Nodes

| Node | LLM Calls | Purpose |
|------|-----------|---------|
| `parse_schemas` | 0 | Regex parse raw schemas → typed Pydantic models |
| `build_semantic_profiles` | 2–3 | Enrich abbreviated names with keywords for embedding |
| `embed_destination_fields` | 0 | Pre-compute destination embedding matrix (40 fields × 384 dim) |
| `route_tables` | 1 | Map source tables → destination collections |
| `retrieve_candidates` | 0 | Hybrid retrieval: `0.8 × cosine + 0.2 × lexical`, top-5 |
| `map_fields` | 5–10 | Select best match + inline transformation notes |
| `validate_relationships` | 0 | FK alignment check (+0.10 confidence boost) |
| `assemble_output` | 0 | Build final JSON, compute unmapped fields |
| `flag_for_review` | 0 | Write low-confidence (<0.75) audit file |

**Total LLM calls: ~8–14** (vs. 40–60+ for naive field-by-field)

---

## Version Improvements

### v1 — Initial Implementation

- Full pipeline implemented (Steps 1–14)
- `reasoning` max_length=200 caused all Groq/Llama calls to fail (model too verbose)
- `top_k=3`, `threshold=0.40` missed `_id` and `stateOrProvince` candidates
- `assemble_output` couldn't handle LLM returning table-prefixed source_field names

**Result:** 31/34 fields mapped | Confidence: 0.79 / 0.87 / 0.71

### v2 — Constants + Bug Fixes

Changes:
- `reasoning` max_length raised to 500
- Extracted all magic numbers to `src/constants.py`
- `RETRIEVAL_TOP_K` increased from 3 → 5
- `RETRIEVAL_THRESHOLD` lowered from 0.40 → 0.38
- Fixed `assemble_output`, `validate_relationships`, `flag_for_review` to handle both `"field_name"` and `"table.field_name"` formats
- Removed intermediate `schema_clean.json` files (parsed directly to Pydantic)
- Folded `derive_transformations` into `map_fields` prompt (saves 1–2 LLM calls)

**Result:** 33/34 fields mapped | Confidence: 0.89 / 0.86 / 0.79

### v3 — Prompt Engineering

Changes:
- Restructured all prompts with Context → Goal → Rules format
- Semantic profiling prompt now explicitly optimizes for embedding similarity
- Added keyword bridging examples (`f_name → firstName, first name, given name`)
- Added `[VALUE_MAP]`, `[TYPE_CAST]`, `[ID_STRATEGY]`, `[FORMAT_CHANGE]` tags for transformation notes
- Fixed over-conservative confidence regression from initial v3 prompt

**Result:** 33/34 fields mapped | Confidence: 0.89 / 0.86 / 0.79 | Clean transformation tags

### Mapping Accuracy (34 total source fields)

| Field | v1 | v2 | v3 | Expected |
|-------|----|----|----|----|
| `emp_id → _id` | ✗ | 0.75 | 0.90 | ✓ |
| `state_prov → stateOrProvince` | ✗ | 0.85 | 0.80 | ✓ |
| `loc_cd → code` | ✗ | ✗ | 0.70 | ✓ |
| `dob → null` | ✓ | ✓ | ✓ | ✓ (no dest field exists) |
| `rec_stat → employment.status` | 0.85 | 0.95 | 0.95 | ✓ |
| `dept_stat → isActive` | 0.85 | 0.85 | 0.90 | ✓ |

---

## Scalability Considerations

### Schema Size
- **Embedding matrix**: O(n) dot product per source field — handles thousands of destination fields
- **Batched LLM calls**: Up to 5 fields per call with disjoint candidates, reducing calls from O(n) to O(n/5)
- **Routing pre-filter**: Restricts retrieval to relevant collection only

### Provider Portability
- `LLMManager` auto-detects provider from credentials — no code changes to switch
- Instructor enforces structured output identically across all providers
- No provider-specific logic in pipeline nodes

### Error Resilience
- Individual field failures logged to `state["errors"]`, pipeline continues
- Instructor retries up to 3x on malformed LLM output
- Partial results produced even when some batches fail

### Adding New Schema Types
- `parse_schemas.py` uses modular regex parsers — add PostgreSQL, Oracle, etc.
- `SourceField` and `DestinationField` are database-agnostic
- Prompts in `src/prompts/` editable without touching pipeline code

---

## Future Extensions

- **Checkpointing** — LangGraph state persistence for resuming failed runs
- **Parallel profiling** — async LLM calls across tables
- **Custom embedding models** — swap `all-MiniLM-L6-v2` for domain-specific model
- **Multi-schema support** — multiple source/destination pairs in one run
- **Denormalization awareness** — teach the prompt that embedded sub-documents (e.g., `department.code`) are populated via FK lookup, not direct 1:1 mapping
- **Separate `derive_transformations` node** — when downstream consumers need machine-readable `TransformationRule` objects instead of free-text notes

---

## Outputs

| File | Purpose |
|------|---------|
| `mappings/output/schema_mapping_output.json` | Primary deliverable — validated field mappings with confidence scores |
| `mappings/output/low_confidence_flags.json` | Audit artifact — mappings below 0.75 with candidate alternatives |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Configure credentials
cp .env.example .env
# Edit .env with your API key

# Run the pipeline
python -c "
from dotenv import load_dotenv
load_dotenv(override=True)
import os; os.environ.pop('OPENAI_BASE_URL', None)
from src.pipeline import run_pipeline
state = run_pipeline()
print(state['final_output'].model_dump_json(indent=2))
"
```

## Configuration

Provider is **auto-detected** from whichever credential is set in `.env`:

| Provider | Required Variables | Detection Priority |
|----------|-------------------|--------------------|
| `openai` | `OPENAI_API_KEY` | 1st |
| `groq` | `GROQ_API_KEY` | 2nd |
| `azure` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | 3rd |
| `snowflake` | `SNOWFLAKE_ACCOUNT` + `SNOWFLAKE_PAT` | 4th |
| `ollama` | (fallback) `OLLAMA_BASE_URL` | 5th |

Override with `LLM_PROVIDER=openai` (or `groq`, `azure`, `snowflake`, `ollama`).

```env
# Model selection
ROUTING_MODEL=gpt-4o-mini       # Profiling & routing (lighter tasks)
MAPPING_MODEL=gpt-4o-mini       # Field mapping (heavier reasoning)
MAX_RETRIES=3                   # Instructor retry count
```

## Tunable Parameters

All retrieval and scoring thresholds live in `src/constants.py`:

```python
RETRIEVAL_TOP_K = 5             # Candidates returned per field
RETRIEVAL_THRESHOLD = 0.38      # Minimum hybrid score to include
EMBEDDING_WEIGHT = 0.8          # Cosine similarity weight
LEXICAL_WEIGHT = 0.2            # RapidFuzz token sort ratio weight
LOW_CONFIDENCE_THRESHOLD = 0.75 # Below this → flagged for review
FK_CONFIDENCE_BOOST = 0.10      # Added when FK alignment confirms mapping
```
