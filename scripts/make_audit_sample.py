"""Audit sample: fixed-seed stratified sample of fast-signed rows.

Writes data/eval/packs/audit_sample.csv with id, bucket, PDF path, page so
a second reviewer can re-check the fast-signed rows against the PDF. The
sample is stratified across the flow-doc question-type buckets (the gold
category is carried as `bucket` because the gold schema uses its own
6-category vocabulary).

Run:  PYTHONPATH=. python scripts/make_audit_sample.py
"""

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

GOLD = "data/eval/datasets/gold_data.jsonl"
OUT = Path("data/eval/packs/audit_sample.csv")
SEED = 20261110
N = 25

# flow-doc bucket for each gold category / abstention subtype
def flow_bucket(r: dict) -> str:
    cat = r.get("category")
    sub = r.get("abstention_subtype")
    if cat == "single_fact_lookup":
        return "1_corpus_lookup_extraction"
    if cat == "numeric_from_table":
        return "1_corpus_lookup_extraction"
    if cat == "same_company_cross_period":
        return "2_multi_period_calculation"
    if cat == "cross_company_comparison":
        return "3_cross_company_comparison"
    if cat == "multi_hop":
        return "2_multi_period_calculation"
    if cat == "unanswerable_abstain":
        return "6_out_of_domain" if sub == "out_of_scope" else "7_8_not_in_corpus_figure_absent"
    return "unknown"


def main() -> int:
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    live = [r for r in rows if r.get("status") == "active"]
    fast = [r for r in live if re.search(r"fast \d", r.get("notes") or "")]
    buckets = defaultdict(list)
    for r in fast:
        buckets[flow_bucket(r)].append(r)
    rng = random.Random(SEED)
    # round-robin over buckets so every bucket with fast rows is represented
    order = sorted(buckets)
    picked = []
    idx = {b: 0 for b in order}
    while len(picked) < N and any(idx[b] < len(buckets[b]) for b in order):
        for b in order:
            if idx[b] < len(buckets[b]) and len(picked) < N:
                pool = buckets[b]
                cand = pool[rng.randrange(len(pool))]
                while cand in picked:
                    cand = pool[rng.randrange(len(pool))]
                picked.append(cand)
                idx[b] += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "bucket", "pdf_path", "pdf_page"])
        for r in sorted(picked, key=lambda r: r["id"]):
            ev = (r.get("evidence") or [{}])[0]
            did = ev.get("document_id") or ""
            page = ev.get("pdf_page")
            w.writerow([r["id"], flow_bucket(r),
                        f"data/corpus/raw/{did}.pdf" if did else "",
                        page if page is not None else ""])
    print(f"wrote {OUT}: {len(picked)} fast-signed rows, seed={SEED}, "
          f"buckets={sorted(buckets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
