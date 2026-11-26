# IFR-QA Agent — Indian Financial-Report QA Agent

A multi-agent RAG system (Planner → Executor → Critic) over Indian corporate financial reports (PDFs).

## Architecture

Four stages (per `architecture-flow-for-mentor-review.txt`):

- **Stage A — Corpus acquisition/freeze**: PDF ingestion, deduplication, page-level extraction, corpus freeze
- **Stage B — Chunking/hybrid index/reranker**: BM25 + dense vectors, RRF fusion, reranking
- **Stage C — Agent online**: Planner → Executor → Critic loop with 3-tool cap and 10-step budget
- **Stage D — Ship/report**: Deployed API + UI, trace dashboard, evaluation harness

## Quickstart

The retrieval index is gitignored. Mount an existing v2 BGE index at
`./results/full_index_v2/bge_small`, or unzip `results/full_index_v2/bge_small.zip`.

```bash
git clone <ifr-qa-agent>
cd IFR-QA-Agent
uv sync

cp .env.example .env
# Fill LLM_CHAT_URL, LLM_MODEL, LLM_API_KEY, EMBEDDINGS_MODEL,
# EMBEDDINGS_API_KEY, and any CA settings in .env.

# If you received bge_small.zip, extract it before starting the backend.
unzip results/full_index_v2/bge_small.zip -d results/full_index_v2

# Backend (.env supplies AGENT_INDEX_PATH)
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# In another terminal (.env supplies BACKEND_URL)
uv run streamlit run src/ui/app.py --server.port 8501
```

For containers instead, run:

```bash
docker compose build && docker compose up -d
```

Then verify either setup:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is Coforge'\''s revenue from operations for FY2025 consolidated?"}'
```

Open the UI at `http://localhost:8501` and ask the same question. If you already
have an index, skip the unzip step; compose mounts
`./results/full_index_v2/bge_small` into the API container.

## Run without Docker

Both backend and UI load `.env` from the repo root, so no inline
`AGENT_INDEX_PATH=` or `BACKEND_URL=` prefix is needed.

Windows PowerShell (VS Code terminal):

```powershell
uv sync
copy .env.example .env
# Edit .env: AGENT_INDEX_PATH=results/full_index_v2/bge_small
#            BACKEND_URL=http://127.0.0.1:8000
if (Test-Path results\full_index_v2\bge_small.zip) {
  Expand-Archive results\full_index_v2\bge_small.zip results\full_index_v2 -Force
}

# Terminal 1
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2
uv run streamlit run src/ui/app.py --server.port 8501
```

WSL/Linux bash:

```bash
uv sync
cp .env.example .env
# Edit .env: AGENT_INDEX_PATH=results/full_index_v2/bge_small
#            BACKEND_URL=http://127.0.0.1:8000
if [ -f results/full_index_v2/bge_small.zip ]; then
  unzip -o results/full_index_v2/bge_small.zip -d results/full_index_v2
fi

# Terminal 1
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2
uv run streamlit run src/ui/app.py --server.port 8501
```

Check readiness with `curl http://127.0.0.1:8000/health` and
`curl http://127.0.0.1:8000/ready`. Chat is enabled only when `/ready`
returns `ready: true`.

## Agent demo (Track B)

`src/agent/demo.py` runs one real-LLM graph turn and prints the plan, tool calls,
final answer, critique, steps used, token count, and errors.

Required environment variables:

- `LLM_CHAT_URL` — OpenAI-compatible chat endpoint (base URL or `/chat/completions`)
- `LLM_MODEL` — chat model name
- `LLM_API_KEY` — bearer token for the endpoint

Optional environment variables:

- `LLM_CA_CERT` — path to a CA bundle for the endpoint
- `AGENT_STATE_DB` — SQLite checkpoint path (default: `results/agent_state.sqlite`)
- `AGENT_TOKEN_BUDGET` — agent token budget (default: `12000`)
- `AGENT_INDEX_PATH` — retrieval index path (default: `results/full_index_v2/bge_small`)
- `AGENT_RUN_TIMEOUT` — question watchdog (default: `120`; use `600` for eval); the legacy
  `AGENT_QUESTION_TIMEOUT_SECONDS` name is still accepted

The first run builds a machine-local `data/cache/meta_rows_*.pkl` (gitignored, ~338 s for
466,799 chunks). Warm starts took ~18 s and subsequent questions ~6 s; rebuild it per machine.

```bash
# Run from the repo root; do not paste secrets into this file.
.venv/bin/python -m src.agent.demo "What was total revenue?" --thread-id demo-1
```

## Observability (Langfuse)

Langfuse tracing is optional and enabled only when `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` are set (placeholders are in `.env.example`).
Each `run_question` call emits one trace with the session id, Planner/Executor/Critic/Answer
spans, search tool calls, final metadata, and a flush at completion. Missing configuration
logs one warning and does not affect answers; keys are never logged.

## Commands

All pipeline commands use one ledger. The canonical ledger is
`data/corpus/ledger.jsonl` (the pipeline default, no flag needed); pass
`--ledger <path>` explicitly to pin a non-default path — and use one path
for every run, never two, or dedupe and resume stop working.

```bash
# Fetch + extract one or more source URLs (run mode; repeatable --url).
# Accepted PDFs are written to <storage-dir>/<sha256>.pdf:
uv run -m src.ingestion.pipeline run \
    --url https://www.bseindia.com/xml-data/corpfiling/AttachHis/<real-filename>.pdf \
    --storage-dir data/corpus/raw

# ...or via a candidates JSON file (each entry needs a real source_url):
uv run -m src.ingestion.pipeline run \
    --candidates data/candidates.example.json \
    --storage-dir data/corpus/raw

# Bulk mode: process a discovery CSV (--candidates needs a required 'url'
# column). Rerun the same command to resume after Ctrl-C (exit 130):
uv run -m src.ingestion.pipeline bulk \
    --candidates data/candidates.csv \
    --storage-dir data/corpus/raw

# Ingestion exit codes: 0 = completed and reconciled, 2 = invalid input or
# accounting gap, 130 = interrupted (Ctrl-C resumes on rerun).

# Operator-dropped PDFs: data/manual_inbox/<company_code>/*.pdf are matched to
# the same ledger. Dry run by default (writes reports + <ledger dir>/hash_cache.json
# only); add --apply to copy PDFs into raw and append one ledger row per document
# (fsynced, so it is resumable). Runs are scoped per company or per batch:
uv run -m src.ingestion.manual_ingest \
    --inbox data/manual_inbox \
    --ledger data/corpus/ledger.jsonl \
    --raw data/corpus/raw \
    --coverage-matrix data/seeds/coverage_matrix.csv \
    --company-registry data/acquisition/company_registry.csv \
    --batch 1 --batch-size 5 --apply     # or --company l_and_t --apply

# Rerun the same command to resume: finished documents are skipped, and the
# hash cache makes a no-change run take about 1 s.

# Read-only view of one persisted page + its tables (writes nothing):
uv run -m src.ingestion.view \
    --storage-dir data/corpus/extract \
    --doc a19bddbe --page 172

# Reconcile ignored raw PDFs against tracked ledger/extracts (dry run is
# read-only; repairs need --re-extract / --register-orphans; see Recovery):
uv run -m src.ingestion.reconcile \
    --storage-dir data/corpus --ledger data/corpus/ledger.jsonl

# Read-only corpus gate report: per doc keep/exclude, fired rules, anomalies,
# reconciliation status, USD/IFRS flags:
uv run -m src.ingestion.gate_report \
    --storage-dir data/corpus/raw --ledger data/corpus/ledger.jsonl \
    --out data/corpus/gate_report.json

# Coverage matrix probe (network; honest client name, <=512 B ranged GET,
# 1 req/s). Probes rows that carry a url not yet PROBED_OK, writes
# PROBED_OK or a failure reason in place, prints per-company and total
# FOUND vs MISSING and projected_document_count:
uv run -m src.ingestion.coverage --matrix data/seeds/coverage_matrix.csv

# Read-only index/filter diagnostic (v1 meta coverage, company periods,
# statement_kind counts, and balance_sheet/FY2025/empty-period checks):
uv run scripts/coverage_report.py --filters

# Run the test suite
uv run pytest tests/ -q

# Lint
uv run ruff check src tests

# Typecheck
uv run mypy --explicit-package-bases --ignore-missing-imports src tests

# Gold v2 (after gold_v2.jsonl is built & signed; --out avoids touching v1.4)
# gold_check needs --input wired (defaults to gold_data.jsonl); freeze/export are path-configurable.
# uv run python -m src.eval.gold_check --input data/eval/datasets/gold_v2.jsonl
uv run python -m src.eval.gold_freeze --input data/eval/datasets/gold_v2.jsonl --out data/eval/datasets/gold_v2.split_manifest.json --seed 42 --splitter-version v2
PYTHONPATH=. uv run python scripts/make_gold_public.py --input data/eval/datasets/gold_v2.jsonl --out data/eval/datasets/gold_v2_public.jsonl

# Agent eval harness: import team gold into eval/gold.jsonl (keeps existing ids)
uv run python scripts/import_gold.py

# Full v1 run; CSV is appended row-by-row and uses 4 workers by default
AGENT_INDEX_PATH=results/full_index_v2/bge_small uv run eval/run_eval.py --workers 4

# Resume into a fixed CSV (skips ids already present) and fixed-seed sample
AGENT_INDEX_PATH=results/full_index_v2/bge_small uv run eval/run_eval.py --resume --out eval/results_page_hash384.csv --workers 4
AGENT_INDEX_PATH=results/full_index_v2/bge_small uv run eval/run_eval.py --sample 3 --workers 4

# Eval-only checks
uv run pytest tests/unit/test_eval_harness.py -q
uv run ruff check eval
```

## Gold review (sign-off)

Current status (2026-11-10): signing in progress (100 of 199 live rows
signed in 3 interactive batches; fastest dwell 36 s). Freeze requires
>= 60 signed rows, all 12 buckets covered, multi_turn pairs both signed,
double-reviewed ids agreeing, and `gold_check` green. The 20 s dwell is
the default guard; reviewers who have already read a batch may re-run a
slice with `--fast '<reason>'` — the reason is logged on each row and
fast-signed rows are listed in the freeze report.

All gold rows are currently UNREVIEWED after the `--reset-review` (100 rows
batched in 2.7 min, not reviewed). Each reviewer signs a disjoint slice with
`--verify`; double-review is required for the listed ids (so a second
reviewer signs them later, clearing `no_separation`):

```bash
# reviewer-1 (36 rows)
uv run -m src.eval.gold_author --verify --reviewer reviewer-1 --ids G-001,G-003,G-004,G-022,G-023,G-024,G-036,G-037,G-038,G-071,G-072,G-073,G-101,G-102,G-103,G-111,G-112,G-113,G-121,G-123,G-127,G-140,G-141,G-142,G-155,G-156,G-157,G-170,G-171,G-172,G-180,G-181,G-182,G-190,G-191,G-192

# reviewer-2 (36 rows)
uv run -m src.eval.gold_author --verify --reviewer reviewer-2 --ids G-005,G-006,G-007,G-021,G-034,G-035,G-039,G-040,G-041,G-074,G-075,G-076,G-104,G-105,G-106,G-114,G-115,G-116,G-125,G-126,G-128,G-143,G-144,G-145,G-158,G-159,G-161,G-173,G-174,G-175,G-183,G-184,G-185,G-193,G-194,G-195

# reviewer-3 (36 rows)
uv run -m src.eval.gold_author --verify --reviewer reviewer-3 --ids G-008,G-010,G-011,G-025,G-026,G-027,G-042,G-043,G-044,G-077,G-078,G-080,G-107,G-108,G-109,G-117,G-118,G-119,G-122,G-124,G-129,G-146,G-147,G-148,G-162,G-163,G-164,G-176,G-177,G-178,G-186,G-187,G-188,G-196,G-197,G-198
```
# batch A - remaining 66 rows (reviewer-1)
uv run -m src.eval.gold_author --verify --reviewer reviewer-1 --fast "pre-read in earlier interactive batches; reviewer time-constrained" --ids G-002,G-009,G-012,G-013,G-014,G-015,G-016,G-017,G-018,G-019,G-020,G-028,G-029,G-030,G-031,G-032,G-033,G-045,G-046,G-047,G-048,G-049,G-050,G-051,G-052,G-053,G-054,G-055,G-056,G-057,G-058,G-059,G-060,G-061,G-062,G-063,G-064,G-065,G-066,G-067,G-068,G-069,G-070,G-079,G-081,G-082,G-083,G-084,G-085,G-086,G-087,G-088,G-089,G-090,G-091,G-092,G-093,G-094,G-095,G-096,G-097,G-098,G-099,G-100,G-110,G-120

# batch B - remaining 33 rows (reviewer-2)
uv run -m src.eval.gold_author --verify --reviewer reviewer-2 --fast "pre-read in earlier interactive batches; reviewer time-constrained" --ids G-130,G-131,G-132,G-133,G-134,G-135,G-136,G-137,G-138,G-139,G-149,G-150,G-151,G-152,G-153,G-154,G-160,G-165,G-166,G-167,G-168,G-169,G-179,G-189,G-199,G-200,G-201,G-202,G-203,G-204,G-205,G-206,G-207

# double-review - 21 ids (reviewer-1)
uv run -m src.eval.gold_author --verify --reviewer reviewer-1 --fast "second-review pass; reviewer time-constrained" --ids G-010,G-023,G-025,G-027,G-039,G-040,G-073,G-075,G-076,G-080,G-101,G-102,G-103,G-105,G-109,G-155,G-173,G-175,G-180,G-185,G-194

Each command is the exact `--verify --ids` command from the reviewer plan.
Rows signed by one reviewer that are also in the double-review list must be
signed by a second reviewer:

```bash
# double-review coverage (21 ids — ask a different reviewer to sign these too):
# G-010,G-023,G-025,G-027,G-039,G-040,G-073,G-075,G-076,G-080,G-101,G-102,G-103,G-105,G-109,G-155,G-173,G-175,G-180,G-185,G-194
```

Workflow per row: page text + PNG path are printed (requires `pdftoppm`);
`[enter] = sign (20+ s dwell enforced by default; `--fast '<reason>'`
skips the dwell and logs the reason on the row); `x` = reject (reason
asked); `s` = skip.

After each signing session, re-check the gates:

```bash
uv run -m src.eval.gold_author --status
uv run -m src.eval.gold_author --check
```

Freeze (run only when ALL gates pass: >= 60 signed rows, all 12 buckets
have signed rows, every multi_turn pair signed, 20 s dwell respected, no
double-review disagreements, `gold_check` green):

```bash
python -m src.eval.gold_merge
python -m src.eval.gold_check
python -m src.eval.gold_freeze
```

Never sign rows programmatically, never edit reviewer fields by hand.


## Constraints

- Python 3.11+
- LLM <= 8B parameters, OpenAI-compatible endpoints
- No GPU, no fine-tuning
- Native-digital PDFs only (no OCR)
- Exactly 3 tools: retriever, calculator, comparator
- Hard 10-step cap on agent loop
## Recovery (raw PDFs are git-ignored)

`/data/corpus/raw/*.pdf` is the only git-ignored corpus artefact. The ledger
(`data/corpus/ledger.jsonl`), page extracts and table records under
`data/corpus/extract/` are tracked. Copies of PDFs, a fresh clone, deleted
extracts or a lost extract file can therefore drift out of step.

Detect drift (read-only dry run; exit 2 if anything is not OK):

```
uv run -m src.ingestion.reconcile --storage-dir data/corpus \
    --ledger data/corpus/ledger.jsonl
```

States printed per document: `OK`, `MISSING_EXTRACT`, `MISSING_RAW`,
`ORPHAN_RAW`, `CORRUPT_EXTRACT`, `CHECKSUM_MISMATCH`.

Repair, explicitly:

```
uv run -m src.ingestion.reconcile --storage-dir data/corpus \
    --ledger data/corpus/ledger.jsonl --re-extract           # rebuild extract JSONLs from intact raw PDFs (no downloads)
uv run -m src.ingestion.reconcile --storage-dir data/corpus \
    --ledger data/corpus/ledger.jsonl --register-orphans     # add ledger rows for unreferenced local PDFs (source=local;
                                                             # refuses bytes of an excluded candidate; override with
                                                             # --allow-excluded-orphans)
```

`MISSING_RAW` is never repaired silently: re-download with `bulk` (a run
whose extracts are intact is skipped from re-download). Writes are atomic
and idempotent; a second identical run changes nothing.

# 1) register manual PDFs (per company or batch; dry-run first without --apply)
uv run -m src.ingestion.manual_ingest \
    --inbox data/manual_inbox \
    --ledger data/corpus/ledger.jsonl \
    --raw data/corpus/raw \
    --coverage-matrix data/seeds/coverage_matrix.csv \
    --company-registry data/acquisition/company_registry.csv \
    --company l_and_t --apply

# 2) write page/table JSONL for every applied doc missing extracts
uv run -m src.ingestion.reconcile \
    --storage-dir data/corpus \
    --ledger data/corpus/ledger.jsonl \
    --re-extract
	
for b in 1 2 3; do
  uv run -m src.ingestion.manual_ingest \
    --inbox data/manual_inbox --ledger data/corpus/ledger.jsonl \
    --raw data/corpus/raw --coverage-matrix data/seeds/coverage_matrix.csv \
    --company-registry data/acquisition/company_registry.csv \
    --batch "$b" --batch-size 20 --apply
done
uv run -m src.ingestion.reconcile --storage-dir data/corpus \
  --ledger data/corpus/ledger.jsonl --re-extract
Output: data/corpus/extract/page_extracts/<document_id>.pages.jsonl and table_records/<document_id>.tables.jsonl. Use --batch 1 --batch-size 5 instead of --company to cover groups.

**Back up the ledger.** It is append-only and the single source of truth
for "done": which candidates completed, which documents exist, which were
excluded. Without it, raw PDFs become orphans and bulk cannot resume
without re-fetching everything.

## NSE quarterly ixbrl -> print-PDF flow (generic; no NSE API)

Standing rules: never call nseindia.com/api; honest UA only (no cookies, no UA
spoofing); robots.txt checked before any fetch; 1 request/second; stop on 403
and list the URL for manual saving; print-to-pdf only for nsearchives
...iXBRL_WEB.html files.

```
# 0. You seed data/manual_inbox/_nse_urls_v1.csv with one row per company:
#      company_code,url   (url = the NSE company integrated-filing page)
#    The page itself is JS-rendered and cannot be scraped honestly, so save it
#    yourself (Ctrl+S) to:
#      data/manual_inbox/_nse_pages/<company_code>.html

# 1. Extract filing rows (quarter_end, variant, audit, consol/standalone,
#    Details ixbrl link) from the saved page; appends deduped rows in-cap
#    (>= 30-Jun-2024) to _nse_urls_v1.csv:
uv run python scripts/nse_extract_links.py --company <company_code>

# 2. Fetch + print + verify + name (re-runnable; fetched HTMLs and printed
#    PDFs are never repeated):
uv run python scripts/nse_ixbrl_pipeline.py --company <company_code>
#    fetch : honest UA, sleep 1, HTML only to data/manual_inbox/_scratch/
#    print : headless Chrome --print-to-pdf; files > 1 MB are sliced at row
#            boundaries and pdfunite'd back (Chrome stalls on multi-MB
#            single-shot prints); landscape re-print on failed checks
#    verify: pdftotext (non-empty text, contains "Total income", contains the
#            quarter-end date)
#    name  : <TICKER>_quarterly_Q#FY####_<standalone|consolidated>_<A|U>_printed.pdf
#            Revision rows: <TICKER>_quarterly_..._<A|U>_printed_revision.pdf

# 3. Revision rule (encoded in the script): a Revision is ingested only if its
#    strip-tag content hash differs from the Original of the same
#    (quarter_end, consolidated/standalone). Content-identical Revisions stay
#    on disk but are marked withdrawn in <company>/manifest.json.

# 4. Ingest -- dry run only; --apply is run manually by the operator when
#    the review passes:
uv run -m src.ingestion.manual_ingest \
    --inbox data/manual_inbox --ledger data/corpus/ledger.jsonl \
    --raw data/corpus/raw --coverage-matrix data/seeds/coverage_matrix.csv \
    --company-registry data/acquisition/company_registry.csv \
    --company <company_code>
```

Round 1 (adani_green) reference numbers: 20 links -> 20 fetched -> 19 printed
PDFs (7 revision rows identical to their Original per the revision rule ->
6 withdrawn files in manifest.json; 2 kept revisions; 13 ledger-new quarterly
docs in the dry run).

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

## Gold set update flow (gold_data.jsonl -> gold_public.jsonl)

`gold_data.jsonl` is the internal frozen master; `gold_public.jsonl` is the
mentor-shareable export (reviewer identity stripped). Never edit
`gold_public.jsonl` by hand — regenerate it.

1. Author / review rows in the master (drafts, verify, sign):

```
uv run python -m src.eval.gold_author --status
uv run python -m src.eval.gold_author --verify --reviewer <name>
```

2. Validate the master before freezing:

```
uv run python -m src.eval.gold_check
```

3. Re-freeze the split (writes `gold_data.jsonl` split fields + `gold_data.split_manifest.json`;
   refuses below 60 reviewed rows; re-freeze requires `--force` AND a new `--splitter-version`):

```
uv run python -m src.eval.gold_freeze --seed 42 --splitter-version <vN> --force
```

4. Export the public file (merges the flow-bucket overlay, strips reviewer identity):

```
PYTHONPATH=. uv run python scripts/make_gold_public.py
```

Rules: freezing only touches `gold_data.jsonl`/manifest; `gold_public.jsonl` is
always re-derived from step 4. Any change to a signed row must go through
`gold_author --supersede` + a new signed revision, never an in-place edit.
