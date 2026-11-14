"""Generate bulk quarterly-variant corrections from the classifier output.

Writes:
  data/corpus/ledger_corrections_v3.jsonl   (variant fixes + supersession revocations)
  data/eval/datasets/quarterly_variant_revocations.json (every revoked pair)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

CLASSIFICATION = Path("data/eval/datasets/quarterly_variant_classification.json")
CORRECTIONS = Path("data/corpus/ledger_corrections_v3.jsonl")
REVOCATIONS = Path("data/eval/datasets/quarterly_variant_revocations.json")


def opposite(basis: str) -> str:
    return "consolidated" if basis == "standalone" else "standalone"


def main() -> int:
    data = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
    recs = {r["document_id"]: r for r in data["records"]}
    mismatches = [r for r in data["records"]
                  if r["in_freeze_v2"] and r["basis"] in ("consolidated", "standalone")
                  and r["freeze_variant"] != r["basis"]]

    lines: list[dict] = []
    revocations: list[dict] = []
    skipped = {"no_twin": 0, "twin_ambiguous": 0, "twin_same_basis": 0,
               "twin_other_company_or_period": 0, "twin_unknown": 0}
    for r in mismatches:
        lines.append({
            "kind": "variant_override",
            "document_id": r["document_id"],
            "from": r["freeze_variant"],
            "to": r["basis"],
            "company_id": r["company_id"],
            "fiscal_year": r["fiscal_year"],
            "evidence": {"marker_pages": r["marker_pages"], "markers": r["markers"]},
        })
        twin_ids = r["revised_by"] + ([r["supersedes"]] if r.get("supersedes") else [])
        twins = [t for t in twin_ids if t]
        twin = None
        for t in twins:
            tr = recs.get(t)
            if tr is None:
                skipped["twin_unknown"] += 1
                continue
            if tr["company_id"] != r["company_id"] or tr["fiscal_year"] != r["fiscal_year"]:
                skipped["twin_other_company_or_period"] += 1
                continue
            if tr["basis"] == opposite(r["basis"]):
                twin = (t, tr)
                break
            if tr["basis"].startswith("ambiguous"):
                skipped["twin_ambiguous"] += 1
            else:
                skipped["twin_same_basis"] += 1
        if not twins:
            skipped["no_twin"] += 1
            continue
        if twin is None:
            continue
        old_id, old = twin
        lines.append({
            "kind": "supersession_revocation",
            "from_document_id": old_id,
            "to_document_id": r["document_id"],
            "company_id": r["company_id"],
            "fiscal_year": r["fiscal_year"],
            "kept_basis": r["basis"],
            "twin_basis": old["basis"],
            "reason": ("twin is the opposite basis for the same company and period; "
                       "the standalone print does not supersede the consolidated print"),
        })
        revocations.append({
            "company_id": r["company_id"],
            "fiscal_year": r["fiscal_year"],
            "kept_document_id": r["document_id"],
            "kept_print_basis": r["basis"],
            "revoked_document_id": old_id,
            "revoked_print_basis": old["basis"],
        })

    header = {
        "kind": "corrections_header",
        "corrections_version": "v3.0",
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_classification": str(CLASSIFICATION),
        "variant_overrides": len(mismatches),
        "supersession_revocations": len(revocations),
        "skipped_pairs": skipped,
    }
    with CORRECTIONS.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    REVOCATIONS.write_text(json.dumps({
        "generated_at": header["created_at"],
        "revocation_count": len(revocations),
        "pairs": revocations,
    }, indent=1), encoding="utf-8")
    print("variant overrides:", len(mismatches), "| revocations:", len(revocations))
    print("skipped:", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
