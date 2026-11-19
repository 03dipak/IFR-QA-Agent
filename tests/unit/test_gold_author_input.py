import json
import sys
from pathlib import Path

import src.eval.gold_author as ga

GOLD = Path("data/eval/datasets/gold_data.jsonl")


def test_verify_input_tmp_leaves_gold_data_intact(tmp_path, monkeypatch, capsys):
    before = GOLD.read_bytes()
    gold = tmp_path / "gold.jsonl"
    row = {"id": "G-999", "status": "active", "review_status": "unreviewed",
           "split": "dev", "category": "unanswerable_abstain", "company_id": "x",
           "period": "FY2025", "question": "q?", "expected_answer": "ABSTAIN",
           "evidence": [], "notes": "", "reviewers": [], "annotated_at": None}
    gold.write_text(json.dumps(row) + "\n")
    ids = tmp_path / "ids.jsonl"
    ids.write_text(json.dumps({"id": "G-999", "source_file": "x"}) + "\n")
    calls = iter(["s"])  # skip action
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(calls))
    monkeypatch.setattr(sys, "argv", ["gold_author", "--verify", "--input", str(gold),
                                      "--ids-file", str(ids), "--reviewer", "tester"])
    rc = ga.main()
    assert rc == 0
    assert GOLD.read_bytes() == before
    saved = json.loads(gold.read_text().strip())
    assert saved["review_status"] == "unreviewed"
