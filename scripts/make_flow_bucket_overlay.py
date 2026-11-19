"""Post-freeze overlay: attach the flow-doc 12-bucket label to each active row.

The frozen gold file must stay byte-identical (gold_hash in
gold_data.split_manifest.json), so the label goes to a side-car overlay
(data/eval/datasets/gold_flow_buckets.jsonl) and the change is logged to
data/eval/datasets/gold_data.corrections.jsonl — the LLD corrections
mechanism (append-only, hashed).

Sources:
  * promoted rows: generator `bucket` field (from packs/drafts_topup.jsonl,
    carried into gold at promotion);
  * G-001..G-100 (no generator bucket): derived from the authored
    category + abstention_subtype (the vocabulary those rows were written
    under).

Run:  PYTHONPATH=. python scripts/make_flow_bucket_overlay.py
"""

import json
from collections import Counter
from datetime import UTC, datetime

from src.eval.gold_rows import load_rows, persist, append_correction

GOLD = "data/eval/datasets/gold_data.jsonl"
OVERLAY = "data/eval/datasets/gold_flow_buckets.jsonl"
CORRECTIONS = "data/eval/datasets/gold_data.corrections.jsonl"

# generator bucket name -> flow-doc bucket (architecture flow, 12 buckets)
GEN_TO_FLOW = {
    "matrix": "4_comparison_matrix_validity",
    "neutralitytrap": "5_neutrality_trap",
    "multi_turn": "12_multi_turn_follow_up",
    "figure_absent": "8_figure_absent_in_corpus",
    "out_of_domain": "6_out_of_domain_question",
    "not_in_corpus_company": "7_not_in_corpus_company",
    "ambiguous": "9_ambiguous_entity",
    "advice": "10_advice_refusal",
    "forecast": "11_forecast_refusal",
}


def derived_flow_bucket(r: dict) -> str:
    """Flow bucket for rows with no generator bucket (G-001..G-100)."""
    cat = r.get("category")
    sub = r.get("abstention_subtype")
    if cat == "single_fact_lookup":
        return "1_corpus_lookup_extraction"
    if cat == "numeric_from_table":
        return "1_corpus_lookup_extraction"
    if cat == "same_company_cross_period":
        return "2_multi_period_calculation"
    if cat == "multi_hop":
        return "2_multi_period_calculation"
    if cat == "cross_company_comparison":
        return "3_cross_company_comparison"
    if cat == "unanswerable_abstain":
        return "6_out_of_domain_question" if sub == "out_of_scope" else "7_not_in_corpus_company"
    return "unknown"


def main() -> int:
    rows = [r for r in load_rows(GOLD) if r.get("status") == "active"]
    out, table = [], Counter()
    for r in rows:
        gen = r.get("bucket")
        flow = GEN_TO_FLOW.get(gen) if gen else None
        if flow is None:
            flow = derived_flow_bucket(r)
        out.append({"id": r["id"], "flow_bucket": flow,
                    "source": "generator_bucket" if gen else "derived_category_subtype"})
        key = f"{gen or '(none: derived from category+subtype)'} -> {flow}"
        table[key] += 1
    print("MAPPING (generator name -> flow bucket -> row count):")
    for k, n in sorted(table.items()):
        print(f"  {n:3d}  {k}")
    persist(out, OVERLAY)
    append_correction(CORRECTIONS, {
        "action": "flow_bucket_overlay",
        "reason": "post-freeze metadata: attach flow-doc 12-bucket labels "
                  "without touching the frozen file (gold_hash must stay "
                  "sha256:7ee6f90cf6e865776d810060ad24de268d085e4aa7f2af5d19a8bf6a5edc9209)",
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actor": "scripts/make_flow_bucket_overlay.py",
        "rows_affected": len(out),
        "overlay": OVERLAY,
        "gold_file_unchanged": True,
    })
    print(f"wrote {OVERLAY}: {len(out)} active rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
