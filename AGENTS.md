# AGENTS.md — IFR-QA-Agent Code Repository

This repository contains the **code implementation** for the Indian Financial-Report QA Agent.

## Cross-Repo Rule

| Concern | Location | Rule |
|---------|----------|------|
| Docs, LLDs, tasks, handoff | `../DataOSS_Capstone_Project_Files_Only/` | Never create code here |
| Code implementation | This repo (`IFR-QA-Agent/`) | All Python code lives here |

Every code module references its governing LLD by filename in its docstring.
Every commit cites its task ID from `TASKBOARD.jsonl` (e.g., `LLD-01: ...`).

## LLD → Code Module Mapping

| LLD Task | LLD File | Code Module |
|----------|----------|-------------|
| LLD-01..03 | `lld_ingestion_pdf.txt` | `src/ingestion/` |
| LLD-04..06 | `lld_tables.txt` | `src/tables/` |
| LLD-07..09 | `lld_normalisation.txt` | `src/normalisation/` |
| LLD-10..12 | `lld_comparator.txt` | `src/comparator/` |
| LLD-13..14 | `lld_entity_registry.txt` | `src/entity_registry/` |
| LLD-15..17 | `lld_citation_labels.txt` | `src/citations/` |
| LLD-18..20 | `lld_llmops.txt` | `src/llmops/` |
| LLD-21..23 | `lld_agents.txt` | `src/agents/` |
| LLD-24..25 | `lld_api_ui.txt` | `src/api/`, `src/ui/` |
| LLD-26..28 | `lld_safety.txt` | `src/safety/` |
| LLD-29 | `lld_eval.txt` | `src/eval/` |
| LLD-30 | `lld_db.txt` | `src/db/` |
| LLD-31 | `lld_tools.txt` | `src/tools/` |

## Tech Stack (Python only)

- **API**: FastAPI + uvicorn
- **UI**: Streamlit
- **LLM**: OpenAI-compatible chat + embeddings endpoints (<=8B parameters)
- **PDF**: pypdf (native-digital only, no OCR)
- **Config**: environment variables via `.env`

## Constraints

- No cloud SDK packages before they produce numbers (O3 rule)
- No vector backend dependency before benchmark results (O4 rule)
- No PHP, Node.js, or JavaScript front-end
- Streamlit is the only UI framework

## Task Board

Task status lives in `../DataOSS_Capstone_Project_Files_Only/docs/reviews/TASKBOARD.jsonl`.
Use `python3 taskboard.py --check` to validate the board from that repo.