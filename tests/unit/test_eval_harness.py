import csv
from pathlib import Path

import eval.run_eval as run_eval


def _patch_stub(monkeypatch, calls):
    class StubGraph:
        def __init__(self, *args, **kwargs):
            self.budget = type("Budget", (), {"used": 7})()

        def run(self, query: str, thread_id: str):
            calls.append(thread_id)
            return {
                "final_answer": {
                    "figures": [{
                        "company_id": "acme",
                        "label": "Total assets",
                        "value": "100",
                    }],
                    "insufficient_evidence": False,
                },
                "requested": [{"company_id": "acme", "label": "Total assets"}],
                "token_count": 7,
                "stop_reason": "accepted",
                "errors": [],
            }

    monkeypatch.setattr(run_eval, "EnvLLM", lambda: object())
    monkeypatch.setattr(run_eval, "AgentGraph", StubGraph)


def test_run_eval_with_stub_agent(monkeypatch):
    calls = []
    _patch_stub(monkeypatch, calls)
    monkeypatch.setenv("AGENT_INDEX_PATH", "test_index.v1")
    assert run_eval.run_eval(["--sample", "3", "--workers", "2"]) == 0
    assert len(calls) == 3
    assert list(Path("eval").glob("results_test_index_v1_*.csv"))


def test_run_eval_resume_skips_existing_ids(tmp_path, monkeypatch):
    calls = []
    _patch_stub(monkeypatch, calls)
    out_path = tmp_path / "resume.csv"
    assert run_eval.run_eval([
        "--sample", "3", "--workers", "2", "--out", str(out_path)
    ]) == 0
    first_calls = len(calls)
    assert first_calls == 3

    assert run_eval.run_eval([
        "--resume", "--sample", "3", "--workers", "2", "--out", str(out_path)
    ]) == 0
    assert len(calls) == first_calls
    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all("question" in row for row in rows)
