"""Freeze the gold set: deterministic 25% holdout, persisted split manifest.

Refuses below MIN_REVIEWED reviewed rows ("blocked", exit 2 — not a failure).
Writes: data/eval/datasets/gold_data.jsonl (split fields updated, atomic) and
data/eval/datasets/gold_data.split_manifest.json (records corpus freeze_v1
manifest sha256, seed, splitter_version, per-row assignments).
Re-freeze requires --force AND a new --splitter-version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from .gold_schema import (canonical, HOLDOUT_FRAC, MIN_REVIEWED,
                          SPLITTER_VERSION, ALLOCATION)
from .gold_rows import load_rows, persist
from .gold_check import run_all, check_drift

GOLD = "data/eval/datasets/gold_data.jsonl"
MANIFEST = "data/eval/datasets/gold_data.split_manifest.json"
FREEZE = "data/corpus/freeze_v1.json"


def corpus_freeze_sha256() -> str:
    return json.load(open(FREEZE, encoding="utf-8"))["manifest_sha256"]


def gold_file_hash(path: str) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pair_units(rows: list[dict]) -> dict[str, list[dict]]:
    """Multi-turn pairs (depends_on -> live row) are atomic freeze units."""
    ids = {r["id"] for r in rows}
    unit_key: dict[str, str] = {}
    for r in rows:
        dep = r.get("depends_on")
        if dep in ids:
            unit_key[r["id"]] = dep
            unit_key[dep] = dep
    units: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        units[unit_key.get(r["id"], r["id"])].append(r)
    return units


def choose_holdout(rows: list[dict], seed: int) -> set[str]:
    by_cat: dict[str, list[list[dict]]] = defaultdict(list)
    for unit in _pair_units(rows).values():
        by_cat[unit[0]["category"]].append(unit)
    rng = random.Random(seed)
    holdout: set[str] = set()
    for cat, units in sorted(by_cat.items()):
        n_rows = sum(len(u) for u in units)
        n = len(units)
        target = math.floor(n * HOLDOUT_FRAC + 0.5)
        target = max(target, 1 if n_rows >= 4 else 0)
        target = min(target, n - 1) if n > 1 else (1 if n == 1 else 0)
        unit_ids = sorted(u[0]["id"] for u in units)
        rng.shuffle(unit_ids)
        keys = set(unit_ids[:target])
        for u in units:
            if u[0]["id"] in keys:
                holdout.update(r["id"] for r in u)
    return holdout


def freeze(rows: list[dict], seed: int, splitter_version: str) -> dict:
    reviewed = [r for r in rows
                if r["status"] == "active"
                and r["review_status"] in ("self_reviewed", "peer_reviewed")]
    holdout = choose_holdout(reviewed, seed)
    for r in rows:
        r["split"] = "holdout" if r["id"] in holdout else "dev"
    # team decision 2026-11-10: fast-signed rows are accepted; record them
    import re as _re
    fast_ids = sorted(r["id"] for r in reviewed if _re.search(r"fast \d", r.get("notes") or ""))
    pairs = {}
    for r in reviewed:
        dep = r.get("depends_on")
        if dep in {x["id"] for x in reviewed}:
            pairs[f"{dep}+{r['id']}"] = sorted([
                (dep, next(x for x in reviewed if x["id"] == dep)["split"]),
                (r["id"], r["split"])])
    pairs_same_side = all(len({side for _, side in v}) == 1 for v in pairs.values())
    import json as _json, hashlib as _hashlib
    from src.eval import scoring_bands as _sb
    dev_rows = _json.load(open("data/eval/datasets/dev_set.jsonl","rb")) if False else None
    try:
        dev = open("data/eval/datasets/dev_set.jsonl","rb").read()
        dev_sha = _hashlib.sha256(dev).hexdigest()
        dev_n = sum(1 for l in open("data/eval/datasets/dev_set.jsonl"))
    except FileNotFoundError:
        dev_sha, dev_n = None, None
    return {
        "splitter_version": splitter_version,
        "seed": seed,
        "n_rows": len(rows),
        "n_reviewed": len(reviewed),
        "n_holdout": len(holdout),
        "fast_signed_count": len(fast_ids),
        "fast_signed_ids": fast_ids,
        "fast_signed_policy": "accepted by team decision 2026-11-10; "
                              "reasons logged per row in review history",
        "multi_turn_pairs": pairs,
        "multi_turn_pairs_same_side": pairs_same_side,
        "assignments": {r["id"]: r["split"] for r in rows},
        "counts_by_category": {
            c: sum(1 for r in reviewed if r["category"] == c)
            for c in sorted({r["category"] for r in reviewed})
        },
        "dev_set_rows": dev_n,
        "dev_set_sha256": dev_sha,
        "scoring_bands_hash": _sb.bands_hash(),
        "normalisation_rules_version": _sb.normalisation_rules_version(),
        "registry_version": _sb.registry_version(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="src.eval.gold_freeze")
    ap.add_argument("--input", default=GOLD)
    ap.add_argument("--out", default=MANIFEST)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--splitter-version", default=SPLITTER_VERSION)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gold, manifest_out = args.input, args.out
    if not Path(gold).is_file():
        print(f"no gold file at {gold} — fold drafts in first: "
              f"python -m src.eval.gold_author --drafts ...", file=sys.stderr)
        return 1
    rows = load_rows(gold)
    reviewed = [r for r in rows
                if r["status"] == "active"
                and r["review_status"] in ("self_reviewed", "peer_reviewed")]
    if len(reviewed) < MIN_REVIEWED:
        print(f"BLOCKED: only {len(reviewed)} reviewed live rows "
              f"(need >= {MIN_REVIEWED}); freeze refused")
        return 2
    drift_problems = check_drift(rows, apply=True)
    if drift_problems:
        persist(rows, gold)
        print(f"BLOCKED: {len(drift_problems)} drift failure(s) flagged needs_review; freeze refused", flush=True)
        return 2
    fails, warns, _ = run_all(rows, ALLOCATION)
    if fails:
        print(f"BLOCKED: gold-validate reports {len(fails)} fail(s); freeze refused "
              f"(fix the checks first)")
        return 2
    manifest_path = Path(manifest_out)
    if manifest_path.is_file() and not args.force:
        print(f"refuses to re-split: {manifest_out} exists "
              f"(use --force AND a new --splitter-version)")
        return 2
    if manifest_path.is_file():
        old = json.load(open(manifest_path, encoding="utf-8"))
        if old.get("splitter_version") == args.splitter_version:
            print("refuses to re-split: unchanged --splitter-version; "
                  "pass a new one with --force")
            return 2
    seed = args.seed if args.seed is not None else int(gold_file_hash(gold)[:8], 16)
    manifest = freeze(rows, seed, args.splitter_version)
    persist(rows, gold)
    gold_hash = gold_file_hash(gold)
    manifest["gold_hash"] = gold_hash
    manifest["corpus_freeze_manifest_sha256"] = corpus_freeze_sha256()
    manifest["split_manifest_hash"] = hashlib.sha256(canonical(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                        indent=2) + "\n", encoding="utf-8")
    print(f"froze {manifest['n_reviewed']} reviewed rows; "
          f"{manifest['n_holdout']} in the holdout")
    print(f"  gold_hash          sha256:{gold_hash}")
    print(f"  split_manifest_hash sha256:{manifest['split_manifest_hash']}")
    print(f"  seed               {seed}")
    print(f"  splitter_version   {args.splitter_version}")
    print(f"  corpus_freeze_manifest_sha256 {manifest['corpus_freeze_manifest_sha256']}")
    print(f"  manifest           {manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
