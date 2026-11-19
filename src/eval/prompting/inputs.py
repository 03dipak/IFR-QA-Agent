"""Gold-dev inputs for Week-1 prompting (active rows only, no holdout)."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from src.config import GOLD_PATH
from src.eval.gold_rows import assert_no_holdout, load_rows

GOLD = Path(GOLD_PATH)
CORRECTIONS = Path("data/eval/datasets/gold_data.corrections.jsonl")


def _iter_rows(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_dev_rows(limit: int | None = None) -> list[dict]:
    """Active dev rows only. Holdout rows must never be read here."""
    rows = load_rows(str(GOLD), active_only=True)
    if not rows:
        rows = [r for r in _iter_rows(GOLD) if r.get("status", "active") == "active"]
    dev = [r for r in rows if r.get("split") == "dev"]
    assert_no_holdout(dev, context="week1 prompt inputs")
    if limit is not None:
        dev = dev[:limit]
    assert_no_holdout(dev, context="week1 prompt inputs (limited)")
    return dev


def load_holdout_ids() -> set[str]:
    ids = set()
    for r in _iter_rows(GOLD):
        if r.get("split") == "holdout":
            rid = r.get("id") or r.get("row_id")
            if rid:
                ids.add(str(rid))
    return ids


def _load_applied_corrections() -> dict[str, dict]:
    if not CORRECTIONS.is_file():
        return {}
    out: dict[str, dict] = {}
    for line in CORRECTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("confidence") == "HIGH" and r.get("issue") in ("note_number_column_bug", "note_number_column_bug_relaxed"):
            out[r["row_id"]] = r
    return out


def _apply_correction(row: dict, corr: dict) -> dict:
    r = copy.deepcopy(row)
    prop = corr.get("proposed") or {}
    ea_orig = str(row.get("expected_answer") or "")
    oldv = row.get("expected_value")
    if "expected_value" in prop and prop["expected_value"] is not None:
        r["expected_value"] = prop["expected_value"]
    if "expected_answer" in prop and prop["expected_answer"] is not None:
        r["expected_answer"] = str(prop["expected_answer"])
    if ("=" in ea_orig or ";" in ea_orig) and oldv is not None:
        try:
            oldf = float(oldv)
            newv = prop.get("expected_value", prop.get("expected_answer"))

            def repl(m):
                tok = m.group(0)
                t = tok.replace(",", "").strip("()")
                try:
                    if abs(float(t) - oldf) < 1e-6:
                        if isinstance(newv, float) and not newv.is_integer():
                            return str(newv)
                        if isinstance(newv, float) and newv.is_integer():
                            return str(int(newv)) if abs(newv) < 1e15 else str(newv)
                        return str(newv)
                except Exception:
                    pass
                return tok

            new_ea = re.sub(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", repl, ea_orig)
            if new_ea != ea_orig:
                r["expected_answer"] = new_ea
                r["expected_value"] = prop.get("expected_value", r.get("expected_value"))
        except Exception:
            pass
    r["_correction_applied"] = corr.get("issue")
    return r


def apply_corrections_overlay(rows: list[dict]) -> list[dict]:
    corrs = _load_applied_corrections()
    if not corrs:
        return rows
    return [(_apply_correction(r, corrs[r["id"]]) if r.get("id") in corrs else r) for r in rows]
