"""Unit tests for src.eval.agent_eval (LLD-29)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import agent_eval as ae
from src.llmops.tracing import Tracer


def _row(**overrides: object) -> dict:
    row = {
        "row_id": "G-001", "question_id": "Q-001", "split": "dev",
        "expected_behaviour": "answer", "expected_answer": "1,234.56 crore",
        "document_id": "a" * 12,
        "evidence": [{"document_id": "a" * 12, "pdf_page": 7}],
    }
    row.update(overrides)
    return row


def _out(**overrides: object) -> ae.AgentOutput:
    out = ae.AgentOutput(answer="answer: 1,234.56 crore [aaaaaaaaaaaa p.7]",
                         citations=["[aaaaaaaaaaaa p.7]"], prompt_tokens=10,
                         completion_tokens=5, tokens=15, latency_ms=20,
                         steps=[{"tool": "retrieve", "error": False}], stopped_reason="done")
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def test_answer_correct_exact_and_numeric() -> None:
    row = _row()
    assert ae.answer_correct(row, "The answer is 1,234.56 crore")
    assert ae.answer_correct(row, "1,234.6 crore")
    assert not ae.answer_correct(row, "9,999 crore")
    abstain = _row(expected_behaviour="abstain", expected_answer="ABSTAIN")
    assert ae.answer_correct(abstain, "ABSTAIN")
    assert not ae.answer_correct(abstain, "1,234.56 crore")


def test_citation_correct_doc_page_and_missing() -> None:
    row = _row()
    assert ae.citation_correct(row, "value [aaaaaaaaaaaa p.7]", [])
    assert ae.citation_correct(row, "value [doc_id:aaaaaaaaaaaa p.7]", [])
    assert not ae.citation_correct(row, "value [bbbbbbbbbbbb p.7]", [])
    assert not ae.citation_correct(row, "value [aaaaaaaaaaaa p.8]", [])
    abstain = _row(expected_behaviour="abstain")
    assert ae.citation_correct(abstain, "ABSTAIN", [])


def test_refusal_correct() -> None:
    assert ae.refusal_correct(_row(expected_behaviour="abstain"), _out(answer="ABSTAIN"))
    assert not ae.refusal_correct(_row(expected_behaviour="abstain"), _out())
    assert ae.refusal_correct(_row(), _out())
    assert not ae.refusal_correct(_row(), _out(refusal="out_of_scope"))


def test_failure_classification_covers_all_taxonomy() -> None:
    cases = [
        (_row(), _out(stopped_reason="token_budget"), "token_exhaustion"),
        (_row(), _out(stopped_reason="loop_detected"), "loop"),
        (_row(), _out(stopped_reason="timeout"), "timeout"),
        (_row(), _out(steps=[{"tool": "retrieve", "error": "boom"}]), "tool_failure"),
        (_row(), _out(answer="", stopped_reason="agent_error"), "incomplete_task"),
        (_row(), _out(refusal="policy"), "false_refusal"),
        (_row(expected_behaviour="abstain"), _out(), "missed_refusal"),
            (_row(), _out(answer="wrong [bbbbbbbbbbbb p.7]", citations=[]), "citation_issue"),
        (_row(), _out(answer="9,999 [aaaaaaaaaaaa p.7]"), "incorrect_answer"),
    ]
    for row, output, expected in cases:
        scored = ae.score_row(row, output)
        assert expected in scored["failures"], (expected, scored["failures"])


def test_score_row_has_all_measures() -> None:
    scored = ae.score_row(_row(), _out())
    for key in ("answer_correct", "citation_correct", "refusal_correct", "task_completed",
                "tool_calls", "agent_steps", "tokens", "latency_ms", "cost_proxy_usd",
                "stopped_reason", "stop_reason", "failures"):
        assert key in scored
    assert scored["tool_calls"] == 1
    assert scored["agent_steps"] == 1
    assert scored["tokens"] == 15
    assert scored["mean_cost_proxy_usd"] if "mean_cost_proxy_usd" in scored else True


def test_aggregate_metrics() -> None:
    rows = [
        ae.score_row(_row(row_id="A"), _out()),
            ae.score_row(_row(row_id="B"), _out(answer="wrong [bbbbbbbbbbbb p.7]",
                                                citations=[],
                                                stopped_reason="token_budget")),
        ae.score_row(_row(row_id="C", expected_behaviour="abstain"),
                     _out(answer="ABSTAIN", refusal="policy", steps=[])),
    ]
    metrics = ae.aggregate(rows)
    assert metrics["n"] == 3
    assert metrics["n_answer"] == 2
    assert metrics["n_abstain"] == 1
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["citation_correctness"] == 0.5
    assert metrics["refusal_correctness"] == 1.0
    assert metrics["task_completion_rate"] == pytest.approx(0.6667, abs=0.0001)
    assert metrics["failure_counts"]["incorrect_answer"] == 1
    assert metrics["failure_counts"]["token_exhaustion"] == 1
    assert metrics["mean_tool_calls"] == pytest.approx(0.6667, abs=0.0001)
    assert metrics["total_tokens"] == 45


def test_trace_extension_records_required_fields(monkeypatch, tmp_path: Path) -> None:
    seen: list[dict] = []
    real_tracer = Tracer

    class CapturingTracer(real_tracer):
        def record(self, *args: object, **kwargs: object) -> dict:
            seen.append(super().record(*args, **kwargs))
            return seen[-1]

    monkeypatch.setattr(ae, "Tracer", lambda sid: CapturingTracer(sid, root=tmp_path / "traces"))
    scored = ae.score_row(_row(), _out())
    ae._trace_eval("run1", "G-001", _out(), scored)
    trace = seen[0]
    for key in ("run_id", "query_id", "model", "provider", "tokens", "latency_ms",
                "tool_calls", "completion_status", "cost_proxy_usd"):
        assert key in trace
    assert trace["query_id"] == "G-001"
    assert trace["completion_status"] == "done"
    assert (tmp_path / "traces" / "eval_run1.jsonl").exists()


def test_run_eval_resumable_and_output_validation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ae, "Tracer", lambda sid: Tracer(sid, root=tmp_path / "traces"))
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(json.dumps(r) for r in [
            _row(row_id="A"), _row(row_id="B", expected_answer="2,000",
                                   evidence=[{"document_id": "a" * 12, "pdf_page": 8}]),
            _row(row_id="C", expected_behaviour="abstain", expected_answer="ABSTAIN"),
        ]) + "\n", encoding="utf-8",
    )
    calls: list[str] = []

    def agent(question: str, *, row: dict | None = None, **_: object) -> ae.AgentOutput:
        calls.append(str(row["row_id"]))
        return ae.mock_agent(question, row=row)

    out = tmp_path / "eval"
    first = ae.run_eval(limit=1, run_id="r", out_dir=out, agent=agent, gold_path=gold)
    second = ae.run_eval(run_id="r", out_dir=out, agent=agent, gold_path=gold)
    assert calls == ["A", "B", "C"]
    assert first["validation"]["ok"]
    assert second["validation"]["ok"]
    assert second["metrics"]["n"] == 3
    assert (out / "per_query.jsonl").read_text().count('"row_id"') == 3
    assert not second["manifest"]["live"]
    assert (out / "validation.json").exists()


def test_error_rate_stop_waits_for_20_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ae, "Tracer", lambda sid: Tracer(sid, root=tmp_path / "traces"))
    gold = tmp_path / "gold.jsonl"
    rows = [_row(row_id=f"A{i}") for i in range(25)]
    gold.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def agent(question: str, *, row: dict | None = None, **_: object) -> ae.AgentOutput:
        return ae.AgentOutput(
            answer="answer: 1,234.56 crore [aaaaaaaaaaaa p.7]",
            citations=["[aaaaaaaaaaaa p.7]"], stopped_reason="accepted",
            errors=["boom"], critic_passed=True,
        )

    result = ae.run_eval(run_id="stop20", out_dir=tmp_path / "eval",
                         agent=agent, gold_path=gold, max_error_rate=0.05)
    assert result["manifest"]["n_rows"] == 20
    assert result["stopped_reason"] == "error_rate_exceeded"


def test_smoke_replay_uses_captured_output_without_new_calls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ae, "Tracer", lambda sid: Tracer(sid, root=tmp_path / "traces"))
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"results": [{
        "row_id": "A", "answer": "1,234.56 crore [aaaaaaaaaaaa p.7]",
        "citations": [], "steps": [], "tokens": 9, "latency_ms": 3,
        "stopped_reason": "done", "refusal": None,
    }]}), encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(_row(row_id="A")) + "\n", encoding="utf-8")
    out = tmp_path / "replay"
    result = ae.run_eval(run_id="replay", out_dir=out, gold_path=gold,
                         agent=ae.smoke_replay_agent(smoke),
                         source="smoke_replay")
    assert result["metrics"]["answer_accuracy"] == 1.0
    assert "smoke_replay" in result["manifest"]["source"]


def test_run_eval_redacts_agent_error_and_stops_on_error_rate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ae, "Tracer", lambda sid: Tracer(sid, root=tmp_path / "traces"))
    monkeypatch.setenv("LLM_API_KEY", "sk-supersecret")
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(r) for r in [
        _row(row_id=f"A{i}") for i in range(20)
    ]) + "\n", encoding="utf-8")

    def failing(question: str, *, row: dict | None = None, **_: object) -> ae.AgentOutput:
        raise RuntimeError("upstream failed sk-supersecret")

    out = tmp_path / "errors"
    result = ae.run_eval(run_id="errors", out_dir=out, agent=failing, gold_path=gold)
    rows = [json.loads(line) for line in (out / "per_query.jsonl").read_text().splitlines()]
    assert result["stopped_reason"] == "error_rate_exceeded"
    assert len(rows) == 20
    assert rows[0]["stopped_reason"] == "agent_error"
    assert "agent_error" in rows[0]["failures"]
    assert rows[0]["errors"] and "sk-supersecret" not in rows[0]["errors"][0]
    assert result["validation"]["ok"]


def test_run_eval_stops_immediately_on_quota(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ae, "Tracer", lambda sid: Tracer(sid, root=tmp_path / "traces"))
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(r) for r in [
        _row(row_id="A"), _row(row_id="B"),
    ]) + "\n", encoding="utf-8")

    def failing(question: str, *, row: dict | None = None, **_: object) -> ae.AgentOutput:
        raise RuntimeError("endpoint returned HTTP 429 quota exceeded")

    out = tmp_path / "quota"
    result = ae.run_eval(run_id="quota", out_dir=out, agent=failing, gold_path=gold)
    rows = [json.loads(line) for line in (out / "per_query.jsonl").read_text().splitlines()]
    assert result["stopped_reason"] == "quota_exhausted"
    assert len(rows) == 1
    assert rows[0]["stopped_reason"] == "quota_exhausted"
    assert "quota_exhausted" in rows[0]["failures"]
    assert result["validation"]["ok"]


def test_warmup_models_writes_separate_report(monkeypatch, tmp_path: Path) -> None:
    from src.agent import tools

    monkeypatch.setattr(tools, "warm_index", lambda: "faiss")
    monkeypatch.setattr(tools, "warm_meta_rows", lambda: 12)
    monkeypatch.setattr(tools, "warm_reranker", lambda: True)
    monkeypatch.setattr(tools, "warm_query_embedding", lambda timeout_s=None: 0.25)

    report = ae.warmup_models(tmp_path)
    data = json.loads((tmp_path / "warmup.json").read_text(encoding="utf-8"))
    assert report["ok"]
    assert [step["name"] for step in data["steps"]] == [
        "index", "meta_rows", "reranker", "query_embedding",
    ]
    assert data["steps"][-1]["latency_s"] == 0.25


def test_validate_run_outputs_detects_missing_file(tmp_path: Path) -> None:
    result = ae.validate_run_outputs(tmp_path)
    assert not result["ok"]
    assert any("missing manifest.json" in error for error in result["errors"])


def test_write_report_contains_evidence(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    out = tmp_path / "eval"
    ae.run_eval(run_id="rep", out_dir=out, gold_path=gold,
                agent=lambda q, *, row=None, **_: ae.mock_agent(q, row=row))
    report_path = ae.write_report(out, report_path=tmp_path / "report.md")
    text = report_path.read_text(encoding="utf-8")
    assert "Week 3–4 Agent Evaluation" in text
    assert "not_live" in text
    assert "LangGraph usage finding" in text
