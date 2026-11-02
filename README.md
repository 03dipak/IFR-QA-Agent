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


## Constraints

- Python 3.11+
- LLM <= 8B parameters, OpenAI-compatible endpoints
- No GPU, no fine-tuning
- Native-digital PDFs only (no OCR)
- Exactly 3 tools: retriever, calculator, comparator
- Hard 10-step cap on agent loop