# Schema Field Mapper

An AI-powered pipeline that automatically maps fields between heterogeneous database schemas using hybrid retrieval (embeddings + lexical similarity) and structured LLM reasoning via Snowflake Cortex.

## Problem

Migrating from a legacy relational database (MySQL) to a modern document store (MongoDB) requires mapping every source field to its destination equivalent. Source schemas often use cryptic abbreviated names (`rec_stat`, `f_name`, `dept_head_id`) while destinations use descriptive paths (`employment.status`, `fullName.firstName`, `headEmployeeId`). Manual mapping is error-prone and doesn't scale.

## Solution

A stateful LangGraph pipeline that:
1. Parses and enriches schemas with semantic profiles (LLM-generated business meaning)
2. Uses hybrid retrieval (embedding cosine similarity + lexical matching) to find candidate matches
3. Applies structured LLM reasoning to select the best match per field
4. Validates results against FK relationships (rule-based)
5. Produces a validated JSON output with confidence scores and audit artifacts

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

## Project Structure

```
schema_field_map/
├── src/
│   ├── pipeline.py                  # LangGraph graph definition + run_pipeline()
│   ├── state.py                     # PipelineState TypedDict (shared state)
│   ├── clients/
│   │   └── __init__.py              # LLMManager (Cortex/OpenAI/Azure/Ollama)
│   ├── models/
│   │   ├── __init__.py              # Re-exports all Pydantic models
│   │   └── schemas.py              # All data contracts (11 Pydantic models)
│   ├── nodes/
│   │   ├── convert_raw_schema.py   # Schema parsing + parse_schemas node
│   │   ├── build_semantic_profiles.py  # LLM semantic enrichment
│   │   ├── embedding_manager.py    # Local embedding model + hybrid scoring
│   │   ├── embed_destination_fields.py # Embed destinations once
│   │   ├── route_tables.py         # LLM table-to-collection routing
│   │   ├── retrieve_candidates.py  # Hybrid retrieval (top-k candidates)
│   │   ├── map_fields.py           # LLM field mapping + transformation derivation
│   │   ├── validate_relationships.py  # Rule-based FK validation
│   │   ├── assemble_output.py      # Final JSON assembly
│   │   └── flag_for_review.py      # Low-confidence audit file
│   └── prompts/
│       └── __init__.py              # All prompt templates (versioned)
├── mappings/
│   ├── mysql/
│   │   ├── schema.json             # Raw MySQL schema (source)
│   │   └── schema_clean.json       # Parsed clean JSON
│   ├── mongodb/
│   │   ├── schema.json             # Raw MongoDB schema (destination)
│   │   └── schema_clean.json       # Parsed clean JSON
│   └── output/
│       ├── schema_mapping_output.json   # Primary deliverable
│       └── low_confidence_flags.json    # Audit artifact
├── .env.example                    # Environment variable template
├── pyproject.toml                  # Dependencies (uv-managed)
└── main.py                         # Entry point
```

## Pipeline Nodes

| Node | LLM Calls | Purpose |
|------|-----------|---------|
| `parse_schemas` | 0 | Parse raw schemas into typed Pydantic models |
| `build_semantic_profiles` | 2–3 | Enrich abbreviated field names with business meaning |
| `embed_destination_fields` | 0 | Pre-compute destination embedding matrix |
| `route_tables` | 1 | Map source tables → destination collections |
| `retrieve_candidates` | 0 | Hybrid retrieval (0.8 × cosine + 0.2 × lexical) |
| `map_fields` | 5–10 + 1–2 | Select best match + derive transformation rules |
| `validate_relationships` | 0 | FK alignment check (+0.10 confidence boost) |
| `assemble_output` | 0 | Build final JSON with unmapped field tracking |
| `flag_for_review` | 0 | Write low-confidence (<0.75) audit file |

**Total LLM calls: ~9–16** (vs. 40–60+ for a naive field-by-field approach)

## Quick Start

```bash
# Install dependencies
uv sync

# Configure credentials (copy and fill in)
cp .env.example .env

# Run the pipeline
python -c "
from dotenv import load_dotenv
load_dotenv()
from src.pipeline import run_pipeline
state = run_pipeline()
print(state['final_output'].model_dump_json(indent=2))
"
```

## Configuration

Set `LLM_PROVIDER` in `.env` to choose your backend:

| Provider | Required Variables |
|----------|-------------------|
| `snowflake_cortex` (default) | `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_PAT` |
| `azure_openai` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| `openai` | `OPENAI_API_KEY` |
| `ollama` | `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`) |

Model selection:
```env
ROUTING_MODEL=gpt-4o-mini       # For profiling & routing (lighter tasks)
MAPPING_MODEL=gpt-4o            # For field mapping & transformations (heavier reasoning)
```

## Scalability Considerations

### Schema Size

The pipeline is designed to scale beyond the current 3-table demo:

- **Embedding matrix**: Destination fields are embedded once and stored as a numpy matrix. Retrieval is O(n) dot product per source field — handles thousands of fields without degradation.
- **Batched LLM calls**: Source fields are grouped (up to 5 per call with disjoint candidate sets), reducing total LLM calls from O(n) to O(n/5).
- **Routing as pre-filter**: Table routing restricts candidate retrieval to the relevant collection, eliminating cross-collection noise as schemas grow.

### Adding New Source/Destination Types

- Schema parsers (`convert_raw_schema.py`) are modular — add a new parser function for PostgreSQL, Oracle, etc. without touching node logic.
- The `SourceField` and `DestinationField` models are generic — not MySQL/MongoDB-specific.
- Prompt templates are externalized in `src/prompts/` for easy modification without code changes.

### LLM Provider Portability

- `LLMManager` wraps any OpenAI-compatible endpoint via a single `LLM_PROVIDER` env var.
- No provider-specific logic in pipeline nodes — all calls go through `call_routing()` or `call_mapping()`.
- Instructor handles structured output enforcement identically across all providers.

### Error Resilience

- Individual field failures are captured in `state["errors"]` without halting the pipeline.
- Instructor retries (max 3) recover from malformed LLM outputs automatically.
- The pipeline produces partial results even when some fields fail.

### Future Extensions

- **Checkpointing**: LangGraph supports state persistence — failed nodes can be resumed without re-running upstream steps.
- **Parallel profiling**: `build_semantic_profiles` can be parallelized across tables with LangGraph's async support.
- **Custom embedding models**: `EmbeddingManager` accepts any `sentence-transformers` model name — swap to a domain-specific model for better retrieval on specialized schemas.
- **Multi-schema support**: The pipeline state is schema-agnostic — extend `parse_schemas` to handle multiple source/destination pairs in a single run.

## Outputs

| File | Purpose |
|------|---------|
| `schema_mapping_output.json` | Primary deliverable — validated field mappings |
| `low_confidence_flags.json` | Audit artifact — mappings below 0.75 confidence with alternatives |
| `pipeline_run_log.json` | Observability — call counts, tokens, duration (future) |

## Technology Stack

| Component | Role |
|-----------|------|
| LangGraph | Stateful pipeline orchestration with conditional edges |
| Instructor + Pydantic v2 | Structured LLM output with automatic retry on validation failure |
| SentenceTransformers | Local embedding model (all-MiniLM-L6-v2, 384-dim) |
| RapidFuzz | Lexical similarity for hybrid scoring |
| NumPy | Cosine similarity via normalized dot product |
| Snowflake Cortex / OpenAI / Azure | LLM backend (configurable) |
