"""Week 3-4 end-to-end agent evaluation harness (LLD-29, lld_eval.txt sections 6-9).

Runs the API agent path (src.agent.service.run_question, src.agent.graph)
against the frozen Week 1 gold, stores per-query evidence and aggregate metrics.
No model calls unless --live is given; mock runs are labelled not_live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.answer_scoring import citation_match
from src.eval.prompting.runner import redact_secrets
from src.llmops.tracing import Tracer

GOLD = Path(os.environ.get("EVAL_GOLD_PATH", "data/eval/datasets/gold_v2_eval_v8_public.jsonl"))
RESULTS = Path(os.environ.get("EVAL_RESULTS_DIR", "results/evaluation"))
REPORT = Path(os.environ.get("EVAL_REPORT_PATH", "docs/WEEK34_AGENT_EVALUATION.md"))
MODEL_ID = os.environ.get("EVAL_MODEL_ID", "Qwen/Qwen3-8B")
COST_PER_M_PROMPT = 0.0
COST_PER_M_COMPLETION = 0.0
PRICE_SOURCE = "college endpoint: no published price; cost proxy is tokens only"
_CITATION_RE = re.compile(
    r"\[(?:doc_id\s*:?\s*)?['\"]?([0-9a-f]{8,64})['\"]?\s+((?:p\.\s*\d+\s*,?\s*)+)\]",
    re.I,
)
_CITATION_PAGE_RE = re.compile(r"p\.\s*(\d+)", re.I)

FAILURE_CLASSES = (
    "incorrect_answer", "citation_issue", "tool_failure", "token_exhaustion",
    "loop", "timeout", "incomplete_task", "false_refusal", "missed_refusal",
    "agent_error", "quota_exhausted",
)
FAILURE_RULES = {
    "tool_failure": "a tool raised or returned an error observation",
    "token_exhaustion": "stopped_reason indicates token_budget",
    "loop": "stopped_reason indicates loop detection or no progress",
    "timeout": "stopped_reason indicates timeout",
    "incomplete_task": "run did not complete a plan or returned no answer",
    "false_refusal": "refused a gold answer row",
    "missed_refusal": "answered a gold abstain/refusal row",
    "citation_issue": "answer citation missing or not matching gold doc/page",
    "incorrect_answer": "answer figure/text does not match gold",
    "agent_error": "live agent raised an exception",
    "quota_exhausted": "endpoint quota or rate-limit error",
}


@dataclass
class AgentOutput:
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    latency_ms: int = 0
    stopped_reason: str = "done"
    refusal: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Completion is deliberately strict: "accepted" counts only when the run
    # had no recorded errors and a genuine critic accept, not a fallback.
    errors: list[str] = field(default_factory=list)
    critic_passed: bool = False
    critic_fallback: bool = False
    critic_verdict: str = ""
    completion_kind: str = ""
    trace_id: str = ""
    retrieved_doc_pages: list[str] = field(default_factory=list)
    recall_k: int = 5


def load_gold(path: Path = GOLD) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gold_hash(path: Path = GOLD) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mock_agent(question: str, *, row: dict[str, Any] | None = None, **_: Any) -> AgentOutput:
    """Deterministic offline stand-in. Produces evidence-shaped output, never scored as live."""
    row = row or {}
    answer = str(row.get("expected_answer") or "")
    if row.get("expected_behaviour") == "abstain":
        return AgentOutput(answer="ABSTAIN", stopped_reason="refused", refusal="out_of_corpus",
                           tokens=12, latency_ms=5, steps=[])
    pages = [int(e["pdf_page"]) for e in row.get("evidence", [])
             if isinstance(e.get("pdf_page"), int)]
    doc = str(row.get("document_id") or "0" * 12)
    cites = [f"[{doc[:12]} p.{p}]" for p in pages]
    return AgentOutput(
        answer=f"answer: {answer} " + " ".join(cites), citations=cites,
        steps=[{"tool": "retrieve", "args": {"query": question[:40]},
                "latency_ms": 4, "error": False}],
        tokens=40, latency_ms=10, stopped_reason="done",
        prompt_tokens=30, completion_tokens=10,
        retrieved_doc_pages=cites, recall_k=5,
    )





def _search_hits(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict) and isinstance(result.get("hits"), list):
        return [hit for hit in result["hits"] if isinstance(hit, dict)]
    if isinstance(result, list):
        return [hit for hit in result if isinstance(hit, dict)]
    return []


def _retrieved_doc_pages(state: Any) -> list[str]:
    """Return ranked, deduplicated retrieved document/page references."""
    if not isinstance(state, dict):
        return []
    refs: list[str] = []
    seen: set[tuple[str, int]] = set()
    for entry in state.get("tool_results", []) or []:
        for hit in _search_hits(entry.get("result")):
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            doc = str(meta.get("document_id") or hit.get("document_id") or "")
            page = meta.get("page", hit.get("page"))
            if not doc or not isinstance(page, int):
                continue
            key = (doc[:12], page)
            if key in seen:
                continue
            seen.add(key)
            refs.append(f"[{doc[:12]} p.{page}]")
    return refs


def _normalize_doc_page(value: Any) -> tuple[str, int] | None:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return str(value[0])[:12], int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = re.match(r"\[?\s*([0-9a-f]{8,64})\s*p\.?\s*(\d+)\s*\]?", value, re.I)
        if match:
            return match.group(1)[:12], int(match.group(2))
    return None


def _recall_at_k(output: AgentOutput, row: dict[str, Any]) -> bool | None:
    gold_pairs = {
        (str(e.get("document_id") or "")[:12], int(e["pdf_page"]))
        for e in row.get("evidence", [])
        if isinstance(e.get("document_id"), str) and isinstance(e.get("pdf_page"), int)
    }
    if not gold_pairs:
        return None
    retrieved = [
        pair for pair in (_normalize_doc_page(item) for item in output.retrieved_doc_pages)
        if pair is not None
    ]
    k = max(1, int(output.recall_k or 1))
    return any(pair in gold_pairs for pair in retrieved[:k])

def _service_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for event in events or []:
        step = dict(event)
        if "tool" not in step and "name" in step:
            step["tool"] = step["name"]
        steps.append(step)
    return steps


def _critic_fallback_from_state(state: dict[str, Any], errors: list[str]) -> bool:
    return bool(state.get("fallback_used")) or any(
        "critic: fallback used" in str(error) for error in errors
    )


def _completion_kind(state: dict[str, Any], *, critic_passed: bool,
                     critic_fallback: bool) -> str:
    feedback = str((state.get("critique") or {}).get("feedback") or "").lower()
    if (
        feedback == "safe fallback"
        or feedback == "all detected companies have hits"
        or feedback.endswith("(revision_cap reached)")
    ):
        return "deterministic_accept"
    if critic_passed and not critic_fallback:
        return "llm_critic_pass"
    return ""


def live_agent(question: str, *, row: dict[str, Any] | None = None, **kwargs: Any) -> AgentOutput:
    """Run through the same service path as the API (src.agent.service)."""
    from src.agent.service import run_question

    result = run_question(question, session_id=kwargs.get("session_id"))
    refusal = ""
    if result.stop_reason == "refused":
        refusal = result.answer
    elif isinstance(result.state, dict):
        refusal = str((result.state.get("refusal") or {}).get("reason") or "")
        if result.stop_reason != "refused":
            refusal = ""
    state = result.state if isinstance(result.state, dict) else {}
    errors = list(state.get("errors") or [])
    critique = state.get("critique") or {}
    critic_passed = str(critique.get("verdict") or "") == "accept"
    critic_fallback = _critic_fallback_from_state(state, errors)
    return AgentOutput(
        answer=result.answer, citations=list(result.citations),
        steps=_service_steps(result.steps), tokens=result.token_count,
        latency_ms=int(round(float(result.latency_s) * 1000)),
        stopped_reason=result.stop_reason, refusal=refusal,
        prompt_tokens=result.token_count, completion_tokens=0,
        errors=errors, critic_passed=critic_passed,
        critic_fallback=critic_fallback,
        critic_verdict=str(critique.get("verdict") or ""),
        completion_kind=_completion_kind(
            state, critic_passed=critic_passed, critic_fallback=critic_fallback
        ),
        trace_id=getattr(result, "trace_id", "") or result.session_id,
        retrieved_doc_pages=_retrieved_doc_pages(state),
        recall_k=5,
    )


def smoke_replay_agent(path: Path) -> Callable[..., AgentOutput]:
    """Replay captured live smoke outputs without any new model calls."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {str(r["row_id"]): r for r in payload.get("results", [])}

    def replay(question: str, *, row: dict[str, Any] | None = None, **_: Any) -> AgentOutput:
        record = by_id.get(str((row or {}).get("row_id")))
        if record is None:
            return AgentOutput(answer="", stopped_reason="no_evidence")
        return AgentOutput(
            answer=record.get("answer") or "",
            citations=list(record.get("citations") or []),
            steps=[dict(s) for s in record.get("steps") or []],
            tokens=int(record.get("tokens") or 0),
            latency_ms=int(record.get("latency_ms") or 0),
            stopped_reason=record.get("stopped_reason") or "done",
            refusal=record.get("refusal") or "",
            prompt_tokens=int(record.get("tokens") or 0), completion_tokens=0,
            errors=list(record.get("errors") or []),
            critic_passed=bool(record.get("critic_passed", False)),
            critic_fallback=bool(record.get("critic_fallback", False)),
            critic_verdict=str(record.get("critic_verdict") or ""),
            completion_kind=str(record.get("completion_kind") or ""),
            trace_id=str(record.get("trace_id") or ""),
            retrieved_doc_pages=list(record.get("retrieved_doc_pages") or []),
            recall_k=int(record.get("recall_k") or 5),
        )

    return replay


_ABSTAIN_MARKERS = (
    "abstain", "cannot answer", "label not found", "conflict:",
    "insufficient evidence", "no evidence", "not printed",
    "not disclosed", "unavailable",
)
_YEAR_RE = re.compile(r"\b(?:FY\s*)?(?:19|20)\d{2}\b", re.I)
_QUARTER_RE = re.compile(r"\bQ\s*[1-4](?:FY\s*(?:19|20)\d{2})?\b", re.I)
_VALUE_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")
_PERIOD_RE = re.compile(r"\b(?:Q[1-4]\s*)?FY\s*(?:19|20)?\d{2}\b", re.I)
_PERIOD_VALUE_RE = re.compile(
    r"(?P<period>\b(?:Q[1-4]\s*)?FY\s*(?:19|20)?\d{2}\b)\s*[=:]\s*"
    r"(?P<value>\(?-?\d[\d,]*(?:\.\d+)?\)?)",
    re.I,
)


def _norm_period(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _answer_values(text: str) -> list[float]:
    cleaned = _YEAR_RE.sub(" ", str(text or ""))
    cleaned = _QUARTER_RE.sub(" ", cleaned)
    values: list[float] = []
    for token in _VALUE_RE.findall(cleaned):
        negative = token.startswith("(") and token.endswith(")")
        token = token.strip("()").replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        values.append(-value if negative else value)
    return values


def _conflict_values_tied(expected: str, answer: str) -> bool:
    pairs = [
        (_norm_period(m.group("period")), m.group("value"))
        for m in _PERIOD_VALUE_RE.finditer(expected)
    ]
    if not pairs:
        return False
    answer_periods = list(_PERIOD_RE.finditer(answer))
    for period, value in pairs:
        want = _answer_values(value)
        tied = False
        for index, match in enumerate(answer_periods):
            if _norm_period(match.group(0)) != period:
                continue
            end = (
                answer_periods[index + 1].start()
                if index + 1 < len(answer_periods)
                else len(answer)
            )
            segment = answer[match.end():end]
            got = _answer_values(segment)
            if want and all(
                any(abs(actual - target) <= max(1e-6, abs(target) * 1e-6) for actual in got)
                for target in want
            ):
                tied = True
                break
        if not tied:
            return False
    return True


def answer_correct(row: dict[str, Any], answer: str) -> bool:
    expected = str(row.get("expected_answer") or "")
    lowered = str(answer or "").lower()
    if lowered.startswith("conflict:"):
        return _conflict_values_tied(expected, str(answer or ""))
    if row.get("expected_behaviour") == "abstain":
        return any(marker in lowered for marker in _ABSTAIN_MARKERS)
    if not expected or expected.upper() == "ABSTAIN":
        return False
    if expected.lower() in lowered:
        return True
    expected_values = _answer_values(expected)
    answer_values = _answer_values(answer)
    if expected_values and all(
        any(abs(actual - want) <= max(1e-6, abs(want) * 1e-6) for actual in answer_values)
        for want in expected_values
    ):
        return True
    from src.eval.answer_scoring import numeric_match
    return numeric_match(expected, answer)


def _citation_refs(answer: str) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for match in _CITATION_RE.finditer(answer or ""):
        doc_id = match.group(1)[:12]
        for page in _CITATION_PAGE_RE.findall(match.group(2)):
            refs.append((doc_id, int(page)))
    return list(dict.fromkeys(refs))


def citation_correct(row: dict[str, Any], answer: str, citations: list[str]) -> bool:
    if row.get("expected_behaviour") == "abstain":
        return True
    gold_pages = {int(e["pdf_page"]) for e in row.get("evidence", [])
                  if isinstance(e.get("pdf_page"), int)}
    if not gold_pages:
        return True
    doc_ids = {str(e.get("document_id") or "")[:12] for e in row.get("evidence", [])}
    # API results carry citations separately from answer text; score only those
    # returned citations, never retrieved-but-uncited chunks.
    if citations:
        refs = _citation_refs(" ".join(str(citation) for citation in citations))
        if not refs:
            return False
        return any(doc_id in doc_ids and page in gold_pages
                   for doc_id, page in refs)
    refs = _citation_refs(answer)
    if refs:
        return any(doc_id in doc_ids and page in gold_pages for doc_id, page in refs)
    return citation_match(gold_pages, answer)


def refusal_correct(row: dict[str, Any], output: AgentOutput) -> bool:
    abstain = row.get("expected_behaviour") == "abstain"
    refused = bool(output.refusal) or "cannot answer" in output.answer.lower() \
        or "abstain" in output.answer.lower()
    return abstain == refused


def classify_failure(row: dict[str, Any], output: AgentOutput, *,
                     correct: bool, cite_ok: bool, refusal_ok: bool) -> list[str]:
    reasons: list[str] = []
    if output.stopped_reason in ("token_budget",):
        reasons.append("token_exhaustion")
    if output.stopped_reason in ("loop_detected", "no_progress", "retrieval_loop"):
        reasons.append("loop")
    if output.stopped_reason in ("timeout", "wall_clock"):
        reasons.append("timeout")
    if output.stopped_reason == "quota_exhausted":
        reasons.append("quota_exhausted")
    if output.errors:
        reasons.append("agent_error")
    if any(step.get("error") for step in output.steps):
        reasons.append("tool_failure")
    if output.stopped_reason in ("agent_error", "no_evidence", "") or not output.answer:
        reasons.append("incomplete_task")
    if not refusal_ok:
        reasons.append("missed_refusal" if row.get("expected_behaviour") == "abstain"
                       else "false_refusal")
    if row.get("expected_behaviour") == "answer":
        if not cite_ok:
            reasons.append("citation_issue")
        if not correct:
            reasons.append("incorrect_answer")
    return [r for r in FAILURE_CLASSES if r in reasons]


def score_row(row: dict[str, Any], output: AgentOutput) -> dict[str, Any]:
    correct = answer_correct(row, output.answer)
    cite_ok = citation_correct(row, output.answer, output.citations)
    refusal_ok = refusal_correct(row, output)
    failures = classify_failure(row, output, correct=correct, cite_ok=cite_ok,
                                refusal_ok=refusal_ok)
    # Completed means a clean accepted run or one of the legacy terminal states.
    # "accepted" is not enough by itself: errors or a critic fallback disqualify it.
    no_errors = not output.errors and not any(
        step.get("error") for step in output.steps
    )
    real_critic_pass = output.critic_passed and not output.critic_fallback
    accepted_complete = (
        output.stopped_reason == "accepted"
        and no_errors
        and real_critic_pass
        and bool(output.answer)
    )
    legacy_complete = (
        output.stopped_reason in ("done", "critic_pass", "refused", "no_evidence")
        and bool(output.answer)
    )
    completed = accepted_complete or legacy_complete
    completion_kind = output.completion_kind if completed else "none"
    if completed and completion_kind not in {"llm_critic_pass", "deterministic_accept"}:
        completion_kind = "legacy"
    errors = [redact_secrets(error) for error in output.errors]
    prompt = output.prompt_tokens
    completion = output.completion_tokens or max(0, output.tokens - prompt)
    cost = prompt / 1_000_000 * COST_PER_M_PROMPT + completion / 1_000_000 * COST_PER_M_COMPLETION
    return {
        "row_id": row.get("row_id"), "question_id": row.get("question_id"),
        "split": row.get("split"), "expected_behaviour": row.get("expected_behaviour"),
        "answer": output.answer, "citations": output.citations,
        "answer_correct": correct, "citation_correct": cite_ok,
        "refusal_correct": refusal_ok, "task_completed": completed,
        "tool_calls": sum(1 for step in output.steps if step.get("tool")),
        "agent_steps": len(output.steps), "tokens": output.tokens,
        "prompt_tokens": prompt, "completion_tokens": completion,
        "latency_ms": output.latency_ms, "cost_proxy_usd": round(cost, 8),
        "errors": errors,
        "critic_verdict": output.critic_verdict,
        "critic_fallback": output.critic_fallback,
        "completion_kind": completion_kind,
        "trace_id": output.trace_id,
        "retrieved_doc_pages": list(output.retrieved_doc_pages),
        "recall_k": output.recall_k,
        "recall_at_k": _recall_at_k(output, row),
        "stopped_reason": output.stopped_reason,
        "stop_reason": output.stopped_reason,
        "failures": failures,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0, "note": "no rows evaluated"}
    answer_rows = [r for r in rows if r["expected_behaviour"] == "answer"]
    abstain_rows = [r for r in rows if r["expected_behaviour"] == "abstain"]
    failures: dict[str, int] = {name: 0 for name in FAILURE_CLASSES}
    for row in rows:
        for name in row["failures"]:
            failures[name] += 1
    def mean(key: str) -> float:
        return round(sum(float(r[key]) for r in rows) / n, 4)

    recall_values = [r.get("recall_at_k") for r in rows
                     if r.get("recall_at_k") is not None]
    recall_k = next((r.get("recall_k") for r in rows if r.get("recall_k") is not None), None)
    return {
        "n": n,
        "n_answer": len(answer_rows), "n_abstain": len(abstain_rows),
        "answer_accuracy": round(
            sum(r["answer_correct"] for r in answer_rows) / len(answer_rows), 4,
        ) if answer_rows else None,
        "citation_correctness": round(sum(r["citation_correct"] for r in answer_rows)
                                      / len(answer_rows), 4) if answer_rows else None,
        "refusal_correctness": round(sum(r["refusal_correct"] for r in abstain_rows)
                                      / len(abstain_rows), 4) if abstain_rows else None,
        "task_completion_rate": round(sum(r["task_completed"] for r in rows) / n, 4),
        "task_completion_split": {
            key: sum(1 for r in rows if r.get("completion_kind") == key)
            for key in ("llm_critic_pass", "deterministic_accept", "legacy", "none")
        },
        "recall_at_k": round(sum(bool(value) for value in recall_values) / len(recall_values), 4)
        if recall_values else None,
        "recall_k": recall_k,
        "n_recall_scored": len(recall_values),
        "failure_counts": failures,
        "mean_tool_calls": mean("tool_calls"), "mean_agent_steps": mean("agent_steps"),
        "mean_tokens": mean("tokens"), "mean_latency_ms": mean("latency_ms"),
        "mean_cost_proxy_usd": mean("cost_proxy_usd"),
        "total_tokens": sum(int(r["tokens"]) for r in rows),
    }


def validate_run_outputs(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    errors: list[str] = []
    for name in ("manifest.json", "per_query.jsonl", "metrics.json", "failures.json"):
        if not (directory / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"ok": False, "errors": errors, "n_rows": 0}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "per_query.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"row_id", "answer_correct", "citation_correct", "refusal_correct",
                "task_completed", "tool_calls", "agent_steps", "tokens", "latency_ms",
                "cost_proxy_usd", "stopped_reason", "stop_reason", "failures",
                "errors", "critic_verdict", "critic_fallback", "completion_kind",
                "trace_id",
                "retrieved_doc_pages", "recall_k", "recall_at_k"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            errors.append(f"row {index} missing {sorted(missing)}")
        invalid = set(row.get("failures", [])) - set(FAILURE_CLASSES)
        if invalid:
            errors.append(f"row {index} unknown failures {sorted(invalid)}")
    ids = [r.get("row_id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate row_id in per_query.jsonl")
    if len(rows) != int(manifest.get("n_rows", -1)):
        errors.append(f"row count {len(rows)} != manifest n_rows {manifest.get('n_rows')}")
    if metrics.get("n") != len(rows):
        errors.append(f"metrics n {metrics.get('n')} != rows {len(rows)}")
    return {"ok": not errors, "errors": errors, "n_rows": len(rows)}


def write_report(directory: Path, *, report_path: Path = REPORT) -> Path:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    failures = json.loads((directory / "failures.json").read_text(encoding="utf-8"))
    counts = metrics.get("failure_counts", {})
    source = str(manifest.get("source", ""))
    evidence_note = (
        "Captured live smoke outputs replayed from the cited file; no new model calls."
        if source.startswith("smoke_replay") else
        "Deterministic mock outputs; this validates the harness, not live agent accuracy."
    )
    lines = [
        "# Week 3–4 Agent Evaluation & Observability",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Mode: `{manifest.get('source', 'live' if manifest['live'] else 'not_live')}`",
        f"- Evidence: {evidence_note}",
        f"- Split: `{manifest['split']}`, rows: `{manifest['n_rows']}`",
        f"- Requested row IDs not found in frozen gold: `{manifest.get('missing_row_ids', [])}`",
        f"- Gold hash: `{manifest['gold_hash']}`",
        f"- Model: `{manifest['model']}`; orchestrator: `{manifest['orchestrator']}`",
        f"- Cost proxy: `{manifest['cost_price_source']}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in ("answer_accuracy", "citation_correctness", "refusal_correctness",
                "task_completion_rate", "mean_tool_calls", "mean_agent_steps",
                "mean_tokens", "mean_latency_ms", "mean_cost_proxy_usd", "total_tokens"):
        lines.append(f"| {key} | {metrics.get(key)} |")
    lines += ["", "## Failure analysis", "", "| Class | Count | Rule |", "| --- | --- | --- |"]
    for name, count in counts.items():
        lines.append(f"| `{name}` | {count} | {failures['failure_rules'][name]} |")
    lines += [
        "",
        f"Failed rows: {len(failures['rows'])} of {metrics.get('n', 0)}.",
        "",
        "## LangGraph usage finding",
        "",
        "`langgraph` is declared in `pyproject.toml` but is not importable in the active "
        "environment, and `src/agents/graph.py` is a hand-rolled Python flow, not a "
        "`StateGraph`. Nodes are `intent`, `resolve`, `plan`, `execute`, `generate`, "
        "`citation_check`, `critique`, optional `revise`, and `persist`; edges are fixed "
        "Python branches with no compiled graph or checkpointer. This harness reuses that "
        "flow and its `Tracer`; it does not introduce another orchestration framework.",
        "",
        "## Limitations",
        "",
        "- The full 73-row dev run is offline mock evidence, not live agent accuracy.",
        "- Smoke replay uses captured live outputs only for rows present in frozen gold "
        "(G-132 is absent and excluded).",
        "- Cost is a token proxy because the college endpoint publishes no price.",
        "- Holdout split is not enabled here; only frozen Week 1 dev rows are used.",
        "- No automatic LLM judge or model downloads are used.",
        "",
        "## Recommended next task",
        "",
        "Run a bounded live full-dev evaluation with the existing endpoint (serialised, "
        "cache-bypassed, rate-limited), then triage the token-exhaustion and citation failures "
        "before any LLD-conformance work.",
        "",
        "## Reproduce / resume",
        "",
        "```bash",
        "python -m src.eval.agent_eval --run-id agent_eval_v1",
        "python -m src.eval.agent_eval --live --run-id agent_eval_live",
        "```",
        "",
        "Per-query JSONL is append-only; rerunning with the same run ID skips completed "
        "query IDs and rewrites the aggregates from persisted rows.",
        "",
    ]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return Path(report_path)


def _trace_eval(run_id: str, query_id: str, output: AgentOutput, scored: dict[str, Any]) -> None:
    tracer = Tracer(f"eval_{run_id}")
    tracer.record(
        "eval_query", tool="agent", args={"query_id": query_id},
        latency_ms=output.latency_ms, prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        run_id=run_id, query_id=query_id, model=MODEL_ID, provider="college-openai-compatible",
        tokens=output.tokens, tool_calls=scored["tool_calls"],
        completion_status=output.stopped_reason,
        cost_proxy_usd=scored["cost_proxy_usd"], failures=scored["failures"],
    )


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in (
        "429", "quota", "rate limit", "too many requests", "capacity", "throttle",
    ))


def _row_has_error(row: dict[str, Any]) -> bool:
    return bool(row.get("errors")) or row.get("stopped_reason") in {
        "agent_error", "quota_exhausted",
    }


def _error_output(question: str, session_id: str, exc: Exception,
                  started: float) -> AgentOutput:
    message = redact_secrets(f"{type(exc).__name__}: {exc}")
    return AgentOutput(
        answer="",
        stopped_reason="quota_exhausted" if _is_quota_error(message) else "agent_error",
        errors=[message],
        latency_ms=int(round((time.monotonic() - started) * 1000)),
        trace_id=hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32],
    )


def warmup_models(directory: Path) -> dict[str, Any]:
    from src.agent import tools

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    for name, fn in (
        ("index", tools.warm_index),
        ("meta_rows", tools.warm_meta_rows),
        ("reranker", tools.warm_reranker),
    ):
        started = time.monotonic()
        entry: dict[str, Any] = {"name": name, "ok": True}
        try:
            value = fn()
            entry["result"] = str(value)
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = redact_secrets(f"{type(exc).__name__}: {exc}")
        entry["latency_ms"] = int(round((time.monotonic() - started) * 1000))
        steps.append(entry)
    started = time.monotonic()
    embedding: dict[str, Any] = {"name": "query_embedding", "ok": True}
    try:
        timeout_s = float(os.environ.get("QUERY_EMBED_WARM_TIMEOUT_SECONDS", "45"))
        entry_value = tools.warm_query_embedding(timeout_s=timeout_s)
        embedding["latency_s"] = round(float(entry_value), 3)
    except Exception as exc:
        embedding["ok"] = False
        embedding["error"] = redact_secrets(f"{type(exc).__name__}: {exc}")
    embedding["latency_ms"] = int(round((time.monotonic() - started) * 1000))
    steps.append(embedding)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_enabled": os.environ.get("AGENT_CACHE_ENABLED", "1"),
        "retrieval_hybrid": os.environ.get("RETRIEVAL_HYBRID"),
        "retrieval_rerank": os.environ.get("RETRIEVAL_RERANK"),
        "ok": all(entry["ok"] for entry in steps),
        "steps": steps,
    }
    (directory / "warmup.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_eval(
    *,
    live: bool = False,
    split: str = "dev",
    limit: int | None = None,
    run_id: str = "agent_eval_v1",
    out_dir: Path | None = None,
    agent: Callable[..., AgentOutput] | None = None,
    gold_path: Path = GOLD,
    row_ids: list[str] | None = None,
    source: str | None = None,
    warmup: bool = False,
    max_error_rate: float = 0.05,
) -> dict[str, Any]:
    rows = [r for r in load_gold(gold_path) if r.get("split") == split]
    missing_row_ids: list[str] = []
    if row_ids is not None:
        wanted = set(row_ids)
        rows = [r for r in rows if str(r.get("row_id")) in wanted]
        missing_row_ids = sorted(wanted - {str(r.get("row_id")) for r in rows})
    if limit is not None:
        rows = rows[:limit]
    directory = Path(out_dir) if out_dir else RESULTS / run_id
    directory.mkdir(parents=True, exist_ok=True)
    per_query = directory / "per_query.jsonl"
    done: set[str] = set()
    prior_rows: list[dict[str, Any]] = []
    if per_query.exists():
        prior_rows = [json.loads(line) for line in per_query.read_text(encoding="utf-8")
                      .splitlines() if line.strip()]
        done = {row["row_id"] for row in prior_rows}
    warmup_report = warmup_models(directory) if warmup and live else None
    manifest = {
        "run_id": run_id, "live": bool(live), "split": split, "limit": limit,
        "row_ids": row_ids, "missing_row_ids": missing_row_ids,
        "source": source or ("live" if live else "not_live (deterministic mock)"),
        "n_rows": len(rows), "gold_hash": gold_hash(gold_path),
        "model": MODEL_ID, "cost_price_source": PRICE_SOURCE,
        "index_version": "C2-cached-50221", "prompt_version": "trackb-mvp-v1",
        "orchestrator": "src.agent.service.run_question (src.agent.graph)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_enabled": os.environ.get("AGENT_CACHE_ENABLED", "1"),
        "warmup": warmup_report,
        "stopped_reason": "none", "max_error_rate": max_error_rate,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runner = agent or (live_agent if live else mock_agent)
    remaining = [row for row in rows if row.get("row_id") not in done]
    error_count = sum(_row_has_error(row) for row in prior_rows)
    stop_reason = "complete"
    with per_query.open("a", encoding="utf-8") as fh:
        for index, row in enumerate(remaining, start=1):
            question = row.get("question_text") or row.get("question") or ""
            session_id = f"eval_{run_id}_{row.get('row_id')}"
            kwargs = {"row": row, "session_id": session_id} if live else {"row": row}
            started = time.monotonic()
            try:
                output = runner(question, **kwargs)
            except Exception as exc:
                output = _error_output(question, session_id, exc, started)
            scored = score_row(row, output)
            fh.write(json.dumps(scored, sort_keys=True) + "\n")
            fh.flush()
            _trace_eval(run_id, str(row.get("row_id")), output, scored)
            error_count += int(_row_has_error(scored))
            completed = len(prior_rows) + index
            if output.stopped_reason == "quota_exhausted":
                stop_reason = "quota_exhausted"
                break
            if completed >= 20 and error_count / completed > max_error_rate:
                stop_reason = "error_rate_exceeded"
                break
    scored_rows = [json.loads(line) for line in per_query.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    manifest["n_rows"] = len(scored_rows)
    manifest["stopped_reason"] = stop_reason
    manifest["error_count"] = error_count
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics = aggregate(scored_rows)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    failures = [{"row_id": r["row_id"], "failures": r["failures"],
                 "stopped_reason": r["stopped_reason"]}
                for r in scored_rows if r["failures"]]
    (directory / "failures.json").write_text(json.dumps(
        {"failure_rules": FAILURE_RULES, "rows": failures}, indent=2), encoding="utf-8")
    validation = validate_run_outputs(directory)
    (directory / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics, "directory": str(directory),
            "n_failed_rows": len(failures), "validation": validation,
            "stopped_reason": stop_reason}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.eval.agent_eval")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default="agent_eval_v1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=REPORT)
    parser.add_argument("--from-smoke", type=Path, default=None,
                        help="replay captured live outputs from results/agent_smoke.json")
    parser.add_argument("--row-ids", default=None, help="comma-separated gold row IDs")
    parser.add_argument("--warmup", action="store_true", help="record model warmup separately")
    parser.add_argument("--max-error-rate", type=float, default=0.05,
                        help="stop when error rate exceeds this fraction")
    args = parser.parse_args(argv)
    if args.split == "holdout":
        raise SystemExit("holdout is not enabled by this harness; dev only")
    agent = smoke_replay_agent(args.from_smoke) if args.from_smoke else None
    source = (f"smoke_replay ({args.from_smoke})" if args.from_smoke else None)
    row_ids = [item.strip() for item in args.row_ids.split(",")] if args.row_ids else None
    result = run_eval(live=args.live, split=args.split, limit=args.limit,
                      run_id=args.run_id, out_dir=args.out_dir, agent=agent,
                      row_ids=row_ids, source=source, warmup=args.warmup,
                      max_error_rate=args.max_error_rate)
    report = None
    if result["validation"]["ok"]:
        report = str(write_report(result["directory"], report_path=args.report_path))
    print(json.dumps({"run_id": args.run_id, "metrics": result["metrics"],
                      "validation": result["validation"],
                      "stopped_reason": result["stopped_reason"], "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
