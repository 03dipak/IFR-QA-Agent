"""Phase 2: deterministic resolver accuracy on live gold-draft answerable rows."""
import json
from collections import Counter
from pathlib import Path
from src.resolver.query_resolver import resolve_query, load_company_registry

OUT = Path("results/phase2_resolver_eval.json")


def main():
    registry = load_company_registry()
    rows = [json.loads(l) for l in open("data/eval/datasets/gold_data.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("lifecycle_status") == "active"
            and r.get("expected_behaviour") == "answer"]
    n = len(rows)
    comp = per = basis_n = basis_hit = metric = 0
    missing_companies = Counter()
    for r in rows:
        res = resolve_query(r.get("question_text") or r.get("question") or "", registry)
        want_ids = r["company_id"].split("|")
        comp += int(all(c in res.company_ids for c in want_ids))
        fy = int(str(r.get("period", "")).replace("FY", ""))
        per += int(any(p.period_type in ("annual", "FY") and p.fiscal_year == fy
                       for p in res.periods))
        variant = r.get("variant")
        if variant in ("consolidated", "standalone"):
            basis_n += 1
            want = "consolidated" if variant == "consolidated" else "standalone"
            basis_hit += int(res.basis == want)
        metric += int(res.metric_id is not None)
        if not all(c in res.company_ids for c in want_ids):
            missing_companies[r["company_id"]] += 1
    out = {"n_answerable": n,
           "company_primary_accuracy": round(comp / n, 4),
           "period_accuracy": round(per / n, 4),
           "basis_accuracy": round(basis_hit / basis_n, 4) if basis_n else None,
           "n_basis_scored": basis_n,
           "metric_detected_rate": round(metric / n, 4),
           "unresolved_companies": dict(missing_companies)}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
