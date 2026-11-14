"""Public gold export: active rows only, reviewer identity stripped.

Reads the frozen data/eval/datasets/gold_data.jsonl, keeps only
status == "active" rows, merges the flow-bucket overlay
(data/eval/datasets/gold_flow_buckets.jsonl), and writes
data/eval/datasets/gold_public.jsonl with the fields `reviewers`,
`reviewer`, and any "[review history: ...]" clause in `notes` removed.
Adds `reviewer_count` (int) and `peer_reviewed` (bool). The internal
frozen file is never modified.

Run:  PYTHONPATH=. python scripts/make_gold_public.py
"""

import json
import re
from pathlib import Path

from src.eval.gold_rows import load_rows, persist

GOLD = "data/eval/datasets/gold_data.jsonl"
OVERLAY = "data/eval/datasets/gold_flow_buckets.jsonl"
PUBLIC = "data/eval/datasets/gold_public.jsonl"
HISTORY_RE = re.compile(r"\s*\[review history:[^\]]*\]")


def public_row(r: dict, flow_bucket: str) -> dict:
    out = dict(r)
    out.pop("reviewers", None)
    out.pop("reviewer", None)
    if out.get("notes"):
        out["notes"] = HISTORY_RE.sub("", out["notes"]).strip()
    out["reviewer_count"] = len(r.get("reviewers") or [])
    out["peer_reviewed"] = out["reviewer_count"] >= 2
    out["flow_bucket"] = flow_bucket
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=GOLD)
    ap.add_argument("--out", default=PUBLIC)
    args = ap.parse_args()
    overlay = {r["id"]: r["flow_bucket"]
               for r in (json.loads(l) for l in open(OVERLAY, encoding="utf-8")
                         if l.strip())}
    rows = load_rows(args.input, active_only=True)
    missing = [r["id"] for r in rows if r["id"] not in overlay]
    if missing:
        print(f"BLOCKED: {len(missing)} active rows without flow_bucket: "
              + ", ".join(missing), file=__import__("sys").stderr)
        return 1
    persist([public_row(r, overlay[r["id"]]) for r in rows], args.out)
    print(f"wrote {args.out}: {len(rows)} active rows (superseded excluded, "
          f"reviewer identity stripped, flow_bucket merged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
