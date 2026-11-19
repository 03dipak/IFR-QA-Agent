"""Precompute B1 oracle candidate pages for every gold-dev row."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.eval.prompting.context import (
    _compute_candidate_pages, _gold_hash, _corpus_freeze_sha)
from src.eval.prompting.inputs import load_dev_rows

OUT = Path("data/eval/prompts") / f"context_cache_{_corpus_freeze_sha()}.jsonl"


def main() -> int:
    rows = load_dev_rows()
    gold_hash = _gold_hash()
    freeze_sha = _corpus_freeze_sha()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"cache_id": "__meta__", "row_id": "__meta__",
                             "gold_hash": gold_hash,
                             "corpus_freeze_sha": freeze_sha,
                             "candidate_pages": []},
                            ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")) + "\n")
        for i, row in enumerate(rows, 1):
            t = time.time()
            pages = _compute_candidate_pages(row)
            rec = {
                "cache_id": row["id"],
                "row_id": row.get("row_id") or row["id"],
                "gold_hash": gold_hash,
                "corpus_freeze_sha": freeze_sha,
                "candidate_pages": pages,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{i}/{len(rows)} {rec['cache_id']} pages={len(pages)} "
                  f"row_time={time.time()-t:.1f}s total={time.time()-t0:.1f}s", flush=True)
    print(f"wrote {OUT} ({len(rows)} rows) in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
