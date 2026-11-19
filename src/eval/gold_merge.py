"""Merge N copies of the gold set into the canonical file.

Input: --inputs a.jsonl b.jsonl [c.jsonl ...] (argument order = precedence
tie-break), --output (default data/eval/datasets/gold_data.jsonl).

Merge rule per row id:
  1. group rows by signed_sha256 (content identity);
  2. if two DIFFERENT content groups both contain a signed row
     (review_status != unreviewed), the merge is REFUSED and the conflicts
     are reported (id, content hashes, signing reviewer(s)) — nothing is
     written;
  3. winner = the content group that contains a signed row (highest
     row_revision, ties broken by argument order); if none is signed, the
     group with the highest row_revision wins;
  4. for the winner, reviewer ids from ALL copies carry the same content
     are unioned; review_status recomputed from the union
     (>=2 distinct reviewers -> peer_reviewed);
  5. row_revision = max revision seen; annotated_at = earliest non-null.
Then persist (atomic) and print the new canonical sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from .gold_rows import load_rows, persist
from .gold_schema import canonical, signed_content_hash

GOLD = "data/eval/datasets/gold_data.jsonl"


def merge_rows(rows_per_file: list[list[dict]]):
    merged: dict[str, list[tuple[int, dict]]] = {}
    for idx, rows in enumerate(rows_per_file):
        for row in rows:
            merged.setdefault(row.get("id"), []).append((idx, row))
    out, conflicts = [], []
    for rid in sorted(merged):
        items = merged[rid]
        by_hash: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for idx, row in items:
            by_hash[signed_content_hash(row)].append((idx, row))
        signed_groups = [h for h, lst in by_hash.items()
                         if any(r.get("review_status") != "unreviewed"
                                for _, r in lst)]
        if len(signed_groups) >= 2:
            for h in signed_groups:
                for idx, row in by_hash[h]:
                    conflicts.append({
                        "id": rid,
                        "content": h[:12],
                        "reviewer": (row.get("reviewer")
                                     or ",".join(row.get("reviewers") or [])),
                        "src": idx,
                    })
            continue
        def group_rank(h):
            lst = by_hash[h]
            signed = any(r.get("review_status") != "unreviewed"
                        for _, r in lst)
            return (1 if signed else 0,
                    max((r.get("row_revision") or 0) for _, r in lst),
                    max(i for i, _ in lst))
        win_hash = max(by_hash, key=group_rank)
        winner_rows = [(i, r) for i, r in items
                       if signed_content_hash(r) == win_hash]
        _, base = max(winner_rows,
                      key=lambda t: ((t[1].get("row_revision") or 0), t[0]))
        base = dict(base)
        revs = set()
        ann = []
        for i, row in winner_rows:
            for rv in (row.get("reviewers") or []):
                if rv:
                    revs.add(rv)
            if row.get("reviewer"):
                revs.add(row["reviewer"])
            if row.get("annotated_at"):
                ann.append(row["annotated_at"])
        base["reviewers"] = sorted(revs)
        base["reviewer"] = sorted(revs)[0] if revs else None
        if len(revs) >= 2:
            base["review_status"] = "peer_reviewed"
        elif revs:
            base["review_status"] = "self_reviewed"
        else:
            base["review_status"] = "unreviewed"
        base["row_revision"] = max((r.get("row_revision") or 0)
                                   for _, r in winner_rows)
        if ann:
            base["annotated_at"] = min(ann)
        out.append(base)
    return out, conflicts


def main() -> int:
    ap = argparse.ArgumentParser(prog="src.eval.gold_merge")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", default=GOLD)
    args = ap.parse_args()
    files = [Path(p) for p in args.inputs]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        print("missing inputs: " + ", ".join(missing), file=sys.stderr)
        return 1
    rows_per_file = [load_rows(str(p)) for p in files]
    merged, conflicts = merge_rows(rows_per_file)
    if conflicts:
        print(f"MERGE REFUSED: {len(conflicts)} conflicting signed row(s)",
              file=sys.stderr)
        for c in conflicts:
            print(f"  {c['id']} content={c['content']} reviewer={c['reviewer']} "
                  f"src={c['src']}", file=sys.stderr)
        return 1
    out = Path(args.output)
    old_hash = None
    if out.is_file():
        old_hash = hashlib.sha256(out.read_bytes()).hexdigest()
    persist(merged, out)
    new_hash = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"merged {sum(len(r) for r in rows_per_file)} rows from {len(files)} "
          f"files -> {out}")
    if old_hash and old_hash != new_hash:
        print(f"  old sha256:{old_hash}")
    print(f"  sha256:{new_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
