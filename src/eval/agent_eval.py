"""Week 3-4 end-to-end agent evaluation harness (LLD-29, lld_eval.txt sections 6-9).

Runs the existing hand-rolled planner->executor->critic graph (src.agents.graph)
against the frozen Week 1 gold, stores per-query evidence and aggregate metrics.
No model calls unless --live is given; mock runs are labelled not_live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.answer_scoring import citation_match
from src.llmops.tracing import Tracer

GOLD = Path("data/eval/datasets/gold_v2_eval_v8_public.jsonl")
RESULTS = Path("results/evaluation")
REPORT = Path("docs/WEEK34_AGENT_EVALUATION.md")
MODEL_ID = "Qwen/Qwen3-8B"
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
    )


def live_agent(question: str, *, row: dict[str, Any] | None = None, **kwargs: Any) -> AgentOutput:
    from src.agents.graph import run_agent

    result = run_agent(question, **kwargs)
    return AgentOutput(
        answer=result.answer, citations=list(result.citations),
        steps=[dict(s) for s in result.steps], tokens=result.tokens,
        latency_ms=result.latency_ms, stopped_reason=result.stopped_reason,
        refusal=result.refusal.reason if result.refusal else "",
        prompt_tokens=result.tokens, completion_tokens=0,
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


def answer_correct(row: dict[str, Any], answer: str) -> bool:
    expected = str(row.get("expected_answer") or "")
    lowered = str(answer or "").lower()
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
    completed = output.stopped_reason in ("done", "critic_pass", "refused", "no_evidence") \
        and bool(output.answer)
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
        "stopped_reason": output.stopped_reason, "failures": failures,
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
                "cost_proxy_usd", "stopped_reason", "failures"}
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
    if per_query.exists():
        done = {json.loads(line)["row_id"] for line in per_query.read_text(encoding="utf-8")
                .splitlines() if line.strip()}
    manifest = {
        "run_id": run_id, "live": bool(live), "split": split, "limit": limit,
        "row_ids": row_ids, "missing_row_ids": missing_row_ids,
        "source": source or ("live" if live else "not_live (deterministic mock)"),
        "n_rows": len(rows), "gold_hash": gold_hash(gold_path),
        "model": MODEL_ID, "cost_price_source": PRICE_SOURCE,
        "index_version": "C2-cached-50221", "prompt_version": "trackb-mvp-v1",
        "orchestrator": "hand_rolled (src.agents.graph); no LangGraph runtime",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runner = agent or (live_agent if live else mock_agent)
    with per_query.open("a", encoding="utf-8") as fh:
        for row in rows:
            if row.get("row_id") in done:
                continue
            question = row.get("question_text") or row.get("question") or ""
            kwargs = {"row": row, "session_id": f"eval_{run_id}_{row.get('row_id')}"} if live else {
                "row": row}
            output = runner(question, **kwargs)
            scored = score_row(row, output)
            fh.write(json.dumps(scored, sort_keys=True) + "\n")
            fh.flush()
            _trace_eval(run_id, str(row.get("row_id")), output, scored)
    scored_rows = [json.loads(line) for line in per_query.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
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
            "n_failed_rows": len(failures), "validation": validation}


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
    args = parser.parse_args(argv)
    if args.split == "holdout":
        raise SystemExit("holdout is not enabled by this harness; dev only")
    agent = smoke_replay_agent(args.from_smoke) if args.from_smoke else None
    source = (f"smoke_replay ({args.from_smoke})" if args.from_smoke else None)
    row_ids = [item.strip() for item in args.row_ids.split(",")] if args.row_ids else None
    result = run_eval(live=args.live, split=args.split, limit=args.limit,
                      run_id=args.run_id, out_dir=args.out_dir, agent=agent,
                      row_ids=row_ids, source=source)
    report = None
    if result["validation"]["ok"]:
        report = str(write_report(result["directory"], report_path=args.report_path))
    print(json.dumps({"run_id": args.run_id, "metrics": result["metrics"],
                      "validation": result["validation"], "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
