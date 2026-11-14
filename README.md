# IFR-QA Agent — Indian Financial-Report QA Agent

A multi-agent RAG system (Planner → Executor → Critic) over Indian corporate financial reports (PDFs).

## Architecture

Four stages (per `architecture-flow-for-mentor-review.txt`):

- **Stage A — Corpus acquisition/freeze**: PDF ingestion, deduplication, page-level extraction, corpus freeze
- **Stage B — Chunking/hybrid index/reranker**: BM25 + dense vectors, RRF fusion, reranking
- **Stage C — Agent online**: Planner → Executor → Critic loop with 3-tool cap and 10-step budget
- **Stage D — Ship/report**: Deployed API + UI, trace dashboard, evaluation harness

## Quick Start

```bash
# Clone the docs repo (tasks, LLDs, handoff)
git clone <dataoss-capstone-project-files-only>

# Clone this repo (code)
git clone <ifr-qa-agent>

# Install dependencies
cd IFR-QA-Agent
uv sync

# Copy environment config
cp .env.example .env
# Edit .env with your LLM endpoint credentials

# Run the API
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run the UI (in another terminal)
uv run streamlit run src/ui/app.py
```

## Commands

```bash
# Download corpus PDFs through the ingestion pipeline (fetch -> checksum ->
# dedupe -> extract; QA + freeze are follow-on steps)
# URL-driven mode (LLD-34): pass source URLs directly, repeatable --url
uv run -m src.ingestion.pipeline run \
    --url https://www.bseindia.com/xml-data/corpfiling/AttachHis/<real-filename>.pdf \
    --storage-dir data/corpus/raw

# ...or via a candidates file (each entry needs a real source_url)
uv run -m src.ingestion.pipeline run \
    --candidates data/candidates.example.json \
    --storage-dir data/corpus/raw

# Run the test suite
uv run pytest tests/ -q

# Lint
uv run ruff check src tests

# Typecheck
uv run mypy --ignore-missing-imports src tests

# Gold v2 (after gold_v2.jsonl is built & signed; --out avoids touching v1.4)
# gold_check needs --input wired (defaults to gold_data.jsonl); freeze/export are path-configurable.
# uv run python -m src.eval.gold_check --input data/eval/datasets/gold_v2.jsonl
uv run python -m src.eval.gold_freeze --input data/eval/datasets/gold_v2.jsonl --out data/eval/datasets/gold_v2.split_manifest.json --seed 42 --splitter-version v2
PYTHONPATH=. uv run python scripts/make_gold_public.py --input data/eval/datasets/gold_v2.jsonl --out data/eval/datasets/gold_v2_public.jsonl
```


## Constraints

- Python 3.11+
- LLM <= 8B parameters, OpenAI-compatible endpoints
- No GPU, no fine-tuning
- Native-digital PDFs only (no OCR)
- Exactly 3 tools: retriever, calculator, comparator
- Hard 10-step cap on agent loop
## Reflection loop (Track B skeleton)

Minimal generate -> critic -> revise loop for an OpenAI-compatible <=8B endpoint.
No RAG, tools, UI, or Docker involved. Config comes from `.env`
(`REFLECTION_MAX_ITERATIONS`, `REFLECTION_MAX_TOTAL_TOKENS`, plus `LLM_BASE_URL`,
`LLM_API_KEY`, `LLM_MODEL`); copy `.env.example` to `.env` first.

Run one query with a trace:

```
uv run python scripts/run_reflection_demo.py --query "What was FY26 revenue?"
```

The critic must return strict JSON (`{"score": 1-5, "issues": [...], "verdict": "pass"|"revise"}`);
malformed JSON is retried once, then the loop falls back to verdict `pass` and logs the failure.

Test offline with a fake client (no network, no key needed):

```
uv run pytest tests/unit/test_reflection_loop.py -q
uv run ruff check src/agents tests/unit/test_reflection_loop.py scripts/run_reflection_demo.py
```

Modules: `src/agents/reflection_loop.py` (orchestrator + trace), `src/agents/generator.py`,
`src/agents/critic.py` (pydantic validation), `src/agents/reviser.py`,
`src/agents/llm_client.py` (thin adapter over `src/llm/client.py`), `src/agents/prompts.py`.
