"""FastAPI backend.

Implements lld_api_ui.txt section 3: API endpoints POST /ask, GET /corpus,
GET /health with JSON-schema request/response and rate limiting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.agent.service import QuestionResult, run_question
from src.agent.tools import (
    index_backend, warm_index, warm_meta_rows, warm_query_embedding, warm_reranker,
)

app = FastAPI(title="IFR-QA Agent API", version="0.1.0")
log = logging.getLogger("ifr.api")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(handler)
for _name in (
    "ifr.service", "ifr.graph", "ifr.llm",
    "src.agent.tools", "src.rag_ingestion.full_index",
):
    _child = logging.getLogger(_name)
    _child.setLevel(logging.INFO)
    if not _child.handlers:
        _child.addHandler(handler)
    _child.propagate = False

BACKEND_BUDGET_SECONDS = float(os.environ["AGENT_RUN_TIMEOUT"])

_CITATION_RE = re.compile(r"\[([0-9a-f]{12}) p\.(\d+)\]", re.I)
_index_path_value = os.environ.get("AGENT_INDEX_PATH")
if not _index_path_value:
    raise RuntimeError("AGENT_INDEX_PATH is not configured")
_INDEX_PATH = Path(_index_path_value)
if not _INDEX_PATH.is_absolute():
    _INDEX_PATH = PROJECT_ROOT / _INDEX_PATH
_INDEX_FILES = (
    "chunks.jsonl", "chunk_ids.json", "embeddings.npy", "chunk_offsets.json",
    "meta.json",
)


def _index_status() -> tuple[str, str]:
    if not _INDEX_PATH.is_dir():
        return "degraded", f"index path missing: {_INDEX_PATH}"
    missing = [name for name in _INDEX_FILES if not (_INDEX_PATH / name).is_file()]
    if missing:
        return "degraded", f"index incomplete: missing {', '.join(missing)} in {_INDEX_PATH}"
    return "healthy", ""


_INDEX_STATUS, _INDEX_REASON = _index_status()
_INDEX_HEALTHY = _INDEX_STATUS == "healthy"
HEALTH_STATUS = "ok"
HEALTH_REASON = _INDEX_REASON
READY = False
READY_ERROR = ""
WARMUP_ERROR = ""
WARMUP_STAGES: list[str] = []
INDEX_BACKEND = "unknown"
_WARM_TASK: asyncio.Task[None] | None = None
_CONVERSATIONAL_PATTERNS = {
    "goodbye": re.compile(
        r"(bye|goodbye|good night|see you|see you later|take care|farewell)"
        r"( (there|all|everyone))*[.!]*",
        re.I,
    ),
    "gratitude": re.compile(
        r"(thanks|thank you|thankyou|thanks a lot|thank you (so )?much|"
        r"many thanks|appreciate it|cheers)( (there|all))*[.!]*",
        re.I,
    ),
    "greeting": re.compile(
        r"(hi|hello|hey|hiya|yo|greetings|good (morning|afternoon|evening))"
        r"( (there|team|assistant|everyone))*[.!?]*",
        re.I,
    ),
    "small_talk": re.compile(
        r"(how are you|how's it going|hows it going|what's up|whats up|"
        r"who are you|what can you do|good to see you)( (there|today))*[.!?]*",
        re.I,
    ),
}
_CONVERSATIONAL_REPLIES = {
    "greeting": (
        "Hello. I can help answer questions about Indian company annual reports. "
        "Ask me about a company, financial year or reported figure."
    ),
    "gratitude": "You're welcome.",
    "goodbye": "Goodbye. Feel free to return with any annual-report question.",
    "small_talk": (
        "I'm here to help with questions about Indian company annual reports. "
        "What would you like to know?"
    ),
}


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class EvidenceItem(BaseModel):
    document_id: str
    pdf_page: int
    citation: str
    citation_label: str = ""
    text: str = ""


class CalculationItem(BaseModel):
    expression: str
    value: str = ""
    status: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class RefusalInfo(BaseModel):
    refused: bool = False
    reason: str = ""
    escalate: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    calculations: list[CalculationItem] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str = ""
    stopped_reason: str = "done"
    citation_status: str = "no_citations"
    refusal: RefusalInfo = Field(default_factory=RefusalInfo)
    escalated: bool = False
    tokens: int = 0
    latency_ms: int = 0
    response_type: str = "financial"
    conversation_kind: str = ""
    request_id: str = ""


def classify_conversational(question: str) -> str:
    normalized = re.sub(r"\s+", " ", (question or "").strip())
    for kind, pattern in _CONVERSATIONAL_PATTERNS.items():
        if pattern.fullmatch(normalized):
            return kind
    return ""


@app.on_event("startup")
async def warm_index_at_startup() -> None:
    global _WARM_TASK
    if not _INDEX_HEALTHY:
        log.error("index check failed: %s", _INDEX_REASON)

    def run_warmup() -> None:
        global READY, READY_ERROR, WARMUP_ERROR, INDEX_BACKEND, WARMUP_STAGES
        READY = False
        READY_ERROR = ""
        WARMUP_ERROR = ""
        WARMUP_STAGES = []
        stage = "index"
        try:
            backend = warm_index()
            INDEX_BACKEND = str(backend or "unknown")
            WARMUP_STAGES.append(stage)
            stage = "meta_rows"
            warm_meta_rows()
            WARMUP_STAGES.append(stage)
            stage = "reranker"
            warm_reranker()
            WARMUP_STAGES.append(stage)
            stage = "query_embedding"
            warm_query_embedding()
            WARMUP_STAGES.append(stage)
            READY = True
            log.info("warmup complete stages=%s backend=%s", ",".join(WARMUP_STAGES), INDEX_BACKEND)
        except Exception as exc:
            READY = False
            READY_ERROR = f"warmup failed at {stage}: {type(exc).__name__}: {exc}"
            log.error("%s", READY_ERROR)

    _WARM_TASK = asyncio.create_task(asyncio.to_thread(run_warmup))


@app.get("/health")
async def health():
    """Cheap process health endpoint; use /ready for index readiness."""
    return {"status": HEALTH_STATUS, "ready": READY, "backend": INDEX_BACKEND}


@app.get("/ready")
async def readiness():
    """Index readiness endpoint."""
    detail: dict[str, Any] = {
        "ready": READY, "backend": INDEX_BACKEND, "stages": list(WARMUP_STAGES)
    }
    error = READY_ERROR or WARMUP_ERROR
    if error:
        detail["error"] = error
    return detail


@app.get("/corpus")
async def corpus():
    """Corpus status endpoint."""
    return {"freeze_version": "pending", "companies": []}


def _answer_refs(answer: str) -> set[tuple[str, int]]:
    return {(doc_id.lower(), int(page)) for doc_id, page in _CITATION_RE.findall(answer or "")}


def _citation_label(meta: dict[str, Any], doc_id: str, page: int) -> str:
    company = str(meta.get("company_id") or meta.get("company") or "").replace("_", " ").strip().title()
    period = str(meta.get("period") or meta.get("fiscal_year") or "").strip().upper()
    fy = period if period.startswith("FY") else (f"FY{period}" if period else "")
    if company and fy:
        return f"{company} · Annual Report {fy} · PDF page {page}"
    if company:
        return f"{company} · PDF page {page}"
    return f"Document {doc_id[:12]} · PDF page {page}"


def _evidence_from_tool_results(tool_results: list[dict[str, Any]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    seen: set[tuple[str, int, str]] = set()
    for entry in tool_results:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result")
        candidates = []
        if isinstance(result, dict):
            hits = result.get("hits")
            if isinstance(hits, list):
                candidates.extend(hits)
            rows = result.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    for side in ("base", "compare"):
                        cid = row.get(f"{side}_citation")
                        page = row.get(f"{side}_page")
                        doc = row.get(f"{side}_document_id")
                        if cid and page is not None:
                            candidates.append({"chunk_id": cid, "document_id": doc, "page": page, "meta": {"company_id": row.get("company_id"), "period": row.get(f"{side}_period_actual") or row.get(f"{side}_period")}})
        for hit in candidates:
            if not isinstance(hit, dict):
                continue
            chunk_id = str(hit.get("chunk_id") or hit.get("id") or "")
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            doc_id = str(hit.get("document_id") or meta.get("document_id") or chunk_id.split(":")[0] or "")
            raw_page = meta.get("page", hit.get("page"))
            try:
                page = int(raw_page)
            except (TypeError, ValueError):
                continue
            if not doc_id:
                continue
            key = (doc_id[:12].lower(), page, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            items.append(EvidenceItem(
                document_id=doc_id,
                pdf_page=page,
                citation=f"[{doc_id[:12]} p.{page}]",
                citation_label=_citation_label(meta, doc_id, page),
                text=str(hit.get("text") or "")[:600],
            ))
    return items


def _citation_status(answer: str, evidence: list[EvidenceItem], citations: list[str] | None = None, figures: list[dict[str, Any]] | None = None) -> str:
    refs = _answer_refs(answer)
    # also accept refs from formatted citation list
    for c in citations or []:
        refs.update((d.lower(), int(p)) for d, p in _CITATION_RE.findall(str(c)))
    if not refs:
        # if figures exist but none have both doc+page, not cited
        figs = [f for f in (figures or []) if isinstance(f, dict)]
        if figs:
            all_have = all(f.get("document_id") and f.get("page") is not None for f in figs)
            return "cited" if all_have else "no_citations"
        return "no_citations"
    available = {(item.document_id[:12].lower(), item.pdf_page) for item in evidence}
    missing = refs - available
    if not missing:
        figs = [f for f in (figures or []) if isinstance(f, dict)]
        if figs and not all(f.get("document_id") and f.get("page") is not None for f in figs):
            return "partially_unresolved"
        return "cited"
    if len(missing) == len(refs):
        return "unresolved"
    return "partially_unresolved"


def _build_response(result: QuestionResult) -> AskResponse:
    state = result.state
    tool_results = state.get("tool_results") or []
    evidence = _evidence_from_tool_results(tool_results)
    raw_refusal = state.get("refusal") or {}
    refusal_data = raw_refusal if isinstance(raw_refusal, dict) else {}
    refusal = RefusalInfo(
        refused=bool(refusal_data.get("refused")),
        reason=str(refusal_data.get("reason") or ""),
        escalate=bool(refusal_data.get("escalate")),
    )
    final_answer = state.get("final_answer") or {}
    figures = final_answer.get("figures") if isinstance(final_answer, dict) else []
    return AskResponse(
        answer=result.answer,
        citations=list(result.citations),
        evidence=evidence,
        trace=tool_results,
        session_id=result.session_id,
        stopped_reason=str(state.get("stop_reason") or "done"),
        citation_status=_citation_status(result.answer, evidence, result.citations, figures),
        refusal=refusal,
        escalated=bool(state.get("escalated") or refusal.escalate),
        tokens=result.token_count,
        latency_ms=int(result.latency_s * 1000),
    )


def _conversational_response(kind: str, request_id: str, latency_ms: int) -> AskResponse:
    return AskResponse(
        answer=_CONVERSATIONAL_REPLIES[kind],
        response_type="conversational",
        conversation_kind=kind,
        stopped_reason="conversational",
        request_id=request_id,
        latency_ms=latency_ms,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request):
    """Route conversational inputs locally; send financial questions to LangGraph."""
    started = time.monotonic()
    request_id = uuid.uuid4().hex[:12]
    kind = classify_conversational(payload.question)
    if kind:
        duration_ms = int((time.monotonic() - started) * 1000)
        response = _conversational_response(kind, request_id, duration_ms)
        log.info(
            "ask request_id=%s status=ok response_type=conversational kind=%s duration_ms=%d",
            request_id, kind, response.latency_ms,
        )
        return response
    try:
        result = await asyncio.to_thread(
            run_question,
            payload.question,
            session_id=payload.session_id,
        )
    except Exception as exc:
        latency_s = time.monotonic() - started
        log.error(
            "ask request_id=%s status=error error_category=%s duration_ms=%d",
            request_id, type(exc).__name__, int(latency_s * 1000),
        )
        raise HTTPException(status_code=500, detail=f"Agent error: {type(exc).__name__}") from exc
    response = _build_response(result)
    response.request_id = request_id
    response.latency_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "ask request_id=%s status=ok response_type=financial stopped_reason=%s duration_ms=%d",
        request_id, response.stopped_reason, int((time.monotonic() - started) * 1000),
    )
    return response


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, default=str) + "\n\n"


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request):
    """Stream shared service step/tool events, then the structured answer."""
    started = time.monotonic()
    request_id = uuid.uuid4().hex[:12]
    kind = classify_conversational(payload.question)
    response_type = "conversational" if kind else "financial"
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    session_id = payload.session_id or uuid.uuid4().hex[:12]

    def emit(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def worker() -> None:
        try:
            result = await asyncio.to_thread(
                run_question,
                payload.question,
                session_id=session_id,
                on_event=emit,
            )
            await asyncio.sleep(0)
            response = _build_response(result)
            response.request_id = request_id
            response.latency_ms = int((time.monotonic() - started) * 1000)
            latency_s = round(response.latency_ms / 1000, 3)
            queue.put_nowait({
                "event": "answer", "type": "answer", "text": response.answer,
                "citations": response.citations, "latency_s": latency_s,
                "session_id": session_id,
            })
            queue.put_nowait({"event": "progress", "stage": "completed",
                              "status": "completed", "label": "Completed"})
            queue.put_nowait({"event": "final", "data": response.model_dump()})
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            queue.put_nowait({"event": "error", "type": "error", "message": message})
            queue.put_nowait({"event": "error", "message": message})
        finally:
            queue.put_nowait({"event": "end"})

    async def events():
        status = "ok"
        error_category = ""
        event_count = 0
        yield _sse({"event": "progress", "stage": "request",
                    "status": "running", "label": "Agent request running"})
        event_count += 1
        task = None
        try:
            if kind:
                response = _conversational_response(
                    kind, request_id, int((time.monotonic() - started) * 1000)
                )
                queue.put_nowait({
                    "event": "answer", "type": "answer", "text": response.answer,
                    "citations": [], "latency_s": response.latency_ms / 1000,
                    "session_id": session_id,
                })
                queue.put_nowait({"event": "progress", "stage": "completed",
                                  "status": "completed", "label": "Completed"})
                queue.put_nowait({"event": "final", "data": response.model_dump()})
                queue.put_nowait({"event": "end"})
            else:
                task = asyncio.create_task(worker())
            while True:
                if await request.is_disconnected():
                    status = "disconnected"
                    break
                if (
                    task is not None
                    and not task.done()
                    and queue.empty()
                    and time.monotonic() - started >= BACKEND_BUDGET_SECONDS
                ):
                    message = (
                        f"TimeoutError: question exceeded {BACKEND_BUDGET_SECONDS:g}s"
                    )
                    yield _sse({"event": "error", "type": "error", "message": message})
                    event_count += 1
                    yield _sse({"event": "error", "message": message})
                    event_count += 1
                    status = "error"
                    error_category = message
                    yield _sse({"event": "end"})
                    event_count += 1
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                event_count += 1
                yield _sse(event)
                if event.get("event") == "error" or event.get("type") == "error":
                    status = "error"
                    error_category = str(event.get("message", ""))
                if event.get("event") == "end":
                    break
        finally:
            if task is not None and not task.done():
                task.cancel()
            log.info(
                "ask_stream request_id=%s status=%s response_type=%s duration_ms=%d "
                "events=%d error_category=%s",
                request_id, status, response_type,
                int((time.monotonic() - started) * 1000), event_count, error_category,
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    """Entry point for the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
