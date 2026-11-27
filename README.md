# IFR-QA Agent — Indian Financial-Report QA Agent

## Overview

A Python RAG question-answering agent for Indian corporate financial-report PDFs. It combines
PDF ingestion, a local hybrid index, a LangGraph-style Planner → Executor → Critic → Answer
flow, and a FastAPI backend with a Streamlit UI. Chat and embeddings come from external
OpenAI-compatible endpoints (≤8B models); no GPU or fine-tuning is used. See `pyproject.toml`
and `AGENTS.md` for the declared stack and constraints.

## Architecture and data flow

1. **Corpus** — `src/ingestion/` fetches, checksums, deduplicates, extracts and records PDFs in
   `data/corpus/ledger.jsonl`; raw PDFs under `data/corpus/raw/` are git-ignored. Tools:
   `src/ingestion/pipeline.py`, `manual_ingest.py`, `gate_report.py`, `reconcile.py`.
2. **Index** — `src/rag_ingestion/` chunks documents and builds BM25/dense artifacts.
   `configs/retrieval.yaml` selects hybrid retrieval (RRF fusion) plus a reranker; the local
   index lives at `results/full_index_v2/bge_small/` (large artifacts are git-ignored).
3. **Agent** — `src/agent/graph.py` registers nine default tools (`search_index`, `get_chunk`,
   `calculator`, `list_companies`, `get_metric`, `rank_values`, `rank_companies`, `sum_values`,
   `compare_periods`) and caps the loop at 10 steps. `src/agent/extract.py` performs
   deterministic figure extraction; `src/agent/service.py` calls `src/llm/client.py`; intent
   refusal is in `src/safety/refusal.py`. Optional tracing is in `src/llmops/`.
4. **Delivery/evaluation** — `src/api/__init__.py` exposes `/health`, `/ready`, `/ask`,
   `/ask/stream` and a `/corpus` stub; `src/ui/app.py` is the Streamlit client. Evaluation lives
   in `src/eval/` and `eval/run_eval.py`.

Data flow: PDF → ledger/page + table extracts → chunks/index → tool-backed agent turn →
answer with citations → API/UI or scored evaluation row.

## Project structure

| Path | Contents |
| --- | --- |
| `src/ingestion/`, `src/tables/`, `src/normalisation/` | Corpus, statement/table and normalisation logic |
| `src/rag_ingestion/`, `src/tools/` | Chunking, index build and retrieval helpers |
| `src/agent/`, `src/agents/`, `src/llm/`, `src/safety/` | Agent graph, tools, LLM client, refusals; `src/agents/` also holds the Track B reflection skeleton |
| `src/api/`, `src/ui/` | FastAPI service and Streamlit app |
| `src/eval/`, `eval/`, `scripts/` | Evaluation harness, gold tooling, runnable scripts |
| `tests/`, `tests/fixtures/` | Unit/integration tests and fixtures |
| `docs/`, `data/`, `results/`, `config/`, `configs/`, `models/`, `notebooks/` | Reports, datasets/ledger, evidence, runtime and retrieval configuration, local models, experiments |

## Prerequisites

- Python ≥ 3.11 (`pyproject.toml`) and `uv` for the commands below; Docker is optional
  (`Dockerfile`, `docker-compose.yml`).
- OpenAI-compatible chat and embeddings endpoints, CA certificate files, and values for the
  required environment variables (see below). Do not commit keys or certificates.
- A locally built BGE-small index directory; index artifacts are git-ignored and no ZIP is
  present in the repository.
- Native-digital PDFs only; OCR is not implemented.

## Setup

```bash
uv sync
cp .env.example .env
# Fill in .env (at minimum LLM_CHAT_URL, LLM_MODEL, LLM_API_KEY and AGENT_INDEX_PATH).
# Place the index at results/full_index_v2/bge_small/ before starting the API.
```

`src/api/__init__.py` refuses to start without `AGENT_INDEX_PATH`; `/ready` turns true only
after index, meta-row, reranker and query-embedding warmup succeed.

## Environment variables (names only)

Loaded from `.env` by `src/config.py`, `src/api/__init__.py`, `src/ui/app.py`, the LLM client
and the evaluation scripts. Values are intentionally omitted here.

| Group | Names |
| --- | --- |
| Chat/embeddings | `LLM_BASE_URL`, `LLM_CHAT_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_CA_CERT`, `LLM_EMBEDDINGS_URL`, `LLM_EMBEDDINGS_API_KEY`, `EMBEDDINGS_URL`, `EMBEDDINGS_MODEL`, `EMBEDDINGS_API_KEY`, `EMBEDDINGS_CA_CERT`, `MAX_TOKENS_CHAT`, `MAX_TOKENS_EMBEDDING` |
| Agent/index | `AGENT_INDEX_PATH`, `AGENT_STATE_DB`, `AGENT_RUN_TIMEOUT`, `AGENT_QUESTION_TIMEOUT_SECONDS`, `AGENT_TOKEN_BUDGET`, `AGENT_MAX_STEPS`, `AGENT_MAX_REVISIONS`, `AGENT_META_CACHE_DIR`, `AGENT_SESSION_LOG_DIR` |
| Retrieval/runtime | `RETRIEVAL_HYBRID`, `RETRIEVAL_HYBRID_CANDIDATE_K`, `RETRIEVAL_RERANK`, `RETRIEVAL_RERANK_POOL`, `RERANK_MODEL`, `RERANK_CACHE_DIR`, `RERANK_THREADS`, `QUERY_EMBED_WARM_TIMEOUT_SECONDS`, `LLM_CALL_TIMEOUT_SECONDS`, `EMBED_CALL_TIMEOUT_SECONDS`, `EXTERNAL_CALL_MAX_RETRIES`, `EXTERNAL_CALL_BACKOFF_SECONDS`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `OMP_NUM_THREADS`, `ORT_INTRA_OP_NUM_THREADS`, `INDEX_FREEZE_VERSION`, `PROMPT_SET_HASH`, `MODEL_PARAMS_HASH` |
| API/UI | `API_HOST`, `API_PORT`, `API_RATE_LIMIT_PER_MINUTE`, `API_RATE_LIMIT_WINDOW_SECONDS`, `FASTAPI_URL`, `BACKEND_URL`, `BACKEND_TIMEOUT_SECONDS`, `STREAMLIT_SERVER_PORT`, `STREAMLIT_SERVER_ADDRESS`, `LOG_LEVEL`, `SESSION_TTL_SECONDS`, `CACHE_TTL_SECONDS` |
| Evaluation | `GOLD_PATH`, `EVAL_GOLD_PATH`, `EVAL_RESULTS_DIR`, `EVAL_REPORT_PATH`, `EVAL_MODEL_ID`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Tracing/reflection | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `REFLECTION_MAX_ITERATIONS`, `REFLECTION_MAX_TOTAL_TOKENS` |

The full template is `.env.example`. `src/agent/service.py` requires
`LLM_CHAT_URL`, `LLM_MODEL`, `LLM_API_KEY` and `AGENT_RUN_TIMEOUT`.

## How to run

```bash
# Local backend (terminal 1)
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# Local UI (terminal 2)
uv run streamlit run src/ui/app.py --server.port 8501

# Containers (alternative)
docker compose build && docker compose up -d

# Readiness
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"<question>"}'
```

The UI reads `BACKEND_URL` and polls `/ready`. Corpus rebuild/repair commands are
`uv run -m src.ingestion.pipeline run --candidates <path> --storage-dir data/corpus/raw`,
`uv run -m src.ingestion.reconcile --storage-dir data/corpus --ledger data/corpus/ledger.jsonl`
and `uv run -m src.ingestion.gate_report --storage-dir data/corpus/raw`; see each module's CLI
help for flags.

## Tests and evaluation

```bash
uv run pytest tests/ -q
uv run ruff check src tests
uv run mypy --explicit-package-bases --ignore-missing-imports src tests
make eval                                   # src.eval.capstone_hard_gates

# Agent harness (mock by default; --live calls the configured endpoint)
uv run python -m src.eval.agent_eval --run-id <unique-id> --out-dir <new-dir>
uv run python -m src.eval.agent_eval --live --limit 10 --run-id <unique-id> --out-dir <new-dir>

# Legacy CSV harness over eval/gold.jsonl
AGENT_INDEX_PATH=results/full_index_v2/bge_small uv run python eval/run_eval.py --workers 4
```

Use a new run ID/output directory for every evaluation run; frozen gold and existing evidence
are not overwritten. Per-query JSONL is append-only and rerunning an ID skips completed rows
(`docs/WEEK34_AGENT_EVALUATION.md`).

## Current status

- `docs/STATUS.md` (2026-11-24) labels figure-QA fixes as **mitigated, not fully DONE**:
  fixture fail-before/pass-after passes exist, but post-change live confirmation is pending.
- The same report still calls the v2 embedding build pending, while
  `results/full_index_v2/bge_small/embedding_progress.json` records `status: complete`
  (466,799 chunks, dim 384); reconcile that discrepancy — TBD.
- `results/evaluation/day5_full_dev_eval_final/manifest.json` records a 20-row live dev run
  stopped by `error_rate_exceeded` (2 errors); `docs/WEEK34_AGENT_EVALUATION.md` describes its
  evidence as deterministic mock output — reconcile — TBD.
- Gold sign-off, the v1-vs-v2 retrieval comparison, clean-environment dry run and a full live
  dev run after the latest agent changes: TBD. Evidence directories include
  `results/evaluation/capstone_hard_gates_v1/` and `results/evidence/`.

## Limitations

- Native-digital PDFs only; no OCR, chart-number or XBRL extraction.
- External ≤8B endpoints are required; chat calls are serialised/rate-limited by project rules.
- The runtime reranker is a config default and is not benchmarked against the frozen metrics
  (`configs/retrieval.yaml`); `/corpus` is a stub (`src/api/__init__.py`).
- Deterministic figure extraction covers six fixture label patterns; other labels use the
  legacy path, and label equivalence is limited (`docs/STATUS.md`).
- No holdout split, automatic LLM judge or published endpoint price; cost is a token proxy
  (`docs/WEEK34_AGENT_EVALUATION.md`).
- Fresh clones need a locally supplied index; no ZIP is committed.
