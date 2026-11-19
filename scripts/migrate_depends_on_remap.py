"""One-time migration: remap stale depends_on draft ids to promoted gold ids.

Cause: gold_topup promotion rewrote row ids D-nnn -> G-nnn but left the
row-2s' depends_on pointing at the old draft ids, so pair checks see
dangling references. This script rewrites ONLY the depends_on field of the
five affected rows; reviewer fields are never touched. The change is
logged to data/eval/datasets/gold_data.corrections.jsonl.

Run once:  PYTHONPATH=. python scripts/migrate_depends_on_remap.py
"""

import json
import sys
from datetime import UTC, datetime

from src.eval.gold_rows import load_rows, persist, append_correction

GOLD = "data/eval/datasets/gold_data.jsonl"
CORRECTIONS = "data/eval/datasets/gold_data.corrections.jsonl"
REMAP = {
    "D-223": "G-123",
    "D-225": "G-125",
    "D-227": "G-127",
    "D-229": "G-129",
    "D-231": "G-200",
}


def main() -> int:
    rows = load_rows(GOLD)
    by_id = {r["id"]: r for r in rows}
    changed = []
    for r in rows:
        dep = r.get("depends_on")
        if dep in REMAP:
            r["depends_on"] = REMAP[dep]
            changed.append({"id": r["id"], "from": dep, "to": REMAP[dep]})
    if not changed:
        print("nothing to remap (already migrated)")
        return 0
    persist(rows, GOLD)
    append_correction(CORRECTIONS, {
        "action": "depends_on_remap",
        "reason": "gold_topup promotion rewrote D-nnn row ids to G-nnn but "
                  "did not remap row-2 depends_on pointers (dangling references)",
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actor": "scripts/migrate_depends_on_remap.py",
        "rows_affected": len(changed),
        "changes": changed,
    })
    print(f"remapped {len(changed)} depends_on pointers:")
    for c in changed:
        print(f"  {c['id']}: {c['from']} -> {c['to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
