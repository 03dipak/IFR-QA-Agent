"""Path-configurable gold_freeze / make_gold_public."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import src.eval.gold_freeze as gfrz
import src.eval.gold_rows as grow


V14_MANIFEST = Path("data/eval/datasets/gold_data.split_manifest.json")
V14_PUBLIC = Path("data/eval/datasets/gold_public.jsonl")


def _row(rid: str) -> dict:
    return {
        "id": rid, "status": "active", "review_status": "self_reviewed",
        "split": "dev", "category": "single_fact_lookup", "question": f"q {rid}",
        "evidence": [], "notes": "",
    }


def _tiny_gold(tmp_path: Path) -> Path:
    p = tmp_path / "gold.jsonl"
    p.write_text(json.dumps(_row("G-001")) + "\n" + json.dumps(_row("G-002")) + "\n")
    return p


def test_freeze_defaults_unchanged():
    ap = gfrz.argparse.ArgumentParser(prog="src.eval.gold_freeze")
    ap.add_argument("--input", default=gfrz.GOLD)
    ap.add_argument("--out", default=gfrz.MANIFEST)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--splitter-version", default=gfrz.SPLITTER_VERSION)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args([])
    assert args.input == "data/eval/datasets/gold_data.jsonl"
    assert args.out == "data/eval/datasets/gold_data.split_manifest.json"


def test_freeze_out_tmp_leaves_v14_manifest_intact(tmp_path, monkeypatch, capsys):
    before = V14_MANIFEST.read_bytes()
    gold = _tiny_gold(tmp_path)
    out = tmp_path / "v2.split_manifest.json"
    monkeypatch.setattr(gfrz, "MIN_REVIEWED", 1)
    monkeypatch.setattr(gfrz, "run_all", lambda rows, alloc: ([], [], {}))
    monkeypatch.setattr(gfrz, "check_drift", lambda rows, apply=True: [])
    monkeypatch.setattr(gfrz, "corpus_freeze_sha256", lambda: "x" * 64)
    monkeypatch.setattr(sys, "argv", ["gold_freeze", "--input", str(gold),
                                      "--out", str(out), "--seed", "1"])
    rc = gfrz.main()
    assert rc == 0
    assert out.is_file()
    assert V14_MANIFEST.read_bytes() == before
    m = json.loads(out.read_text())
    assert m["seed"] == 1


def test_freeze_guard_fires_on_existing_out(tmp_path, monkeypatch, capsys):
    gold = _tiny_gold(tmp_path)
    out = tmp_path / "v2.split_manifest.json"
    out.write_text("{}")
    monkeypatch.setattr(gfrz, "MIN_REVIEWED", 1)
    monkeypatch.setattr(gfrz, "run_all", lambda rows, alloc: ([], [], {}))
    monkeypatch.setattr(gfrz, "check_drift", lambda rows, apply=True: [])
    monkeypatch.setattr(gfrz, "corpus_freeze_sha256", lambda: "x" * 64)
    monkeypatch.setattr(sys, "argv", ["gold_freeze", "--input", str(gold),
                                      "--out", str(out), "--seed", "1"])
    rc = gfrz.main()
    assert rc == 2
    assert "refuses to re-split" in capsys.readouterr().out
    assert out.read_text() == "{}"


def test_make_gold_public_out_tmp_leaves_v14_public_intact(tmp_path, monkeypatch):
    before = V14_PUBLIC.read_bytes()
    # minimal active row and overlay
    gold = tmp_path / "gold.jsonl"
    row = {"id": "G-999", "status": "active", "review_status": "self_reviewed",
           "split": "dev", "category": "single_fact_lookup", "question": "q",
           "evidence": [], "notes": "", "reviewers": ["a"], "reviewer": "a"}
    gold.write_text(json.dumps(row) + "\n")
    overlay = tmp_path / "overlay.jsonl"
    overlay.write_text(json.dumps({"id": "G-999", "flow_bucket": "x"}) + "\n")
    out = tmp_path / "public.jsonl"
    import scripts.make_gold_public as mgp
    monkeypatch.setattr(mgp, "OVERLAY", str(overlay))
    monkeypatch.setattr(sys, "argv", ["make_gold_public", "--input", str(gold),
                                      "--out", str(out)])
    rc = mgp.main()
    assert rc == 0
    assert out.is_file()
    assert V14_PUBLIC.read_bytes() == before
    pub = json.loads(out.read_text().strip())
    assert "reviewers" not in pub and pub["flow_bucket"] == "x"
