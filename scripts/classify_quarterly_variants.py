"""Classify each quarterly document's true variant from printed page markers.

Reads the ledger (last-wins document rows) and the page extracts, decides the
basis from the "Nature of report" field, and writes:
  data/eval/datasets/quarterly_variant_classification.json
  data/eval/datasets/quarterly_variant_ambiguous.json
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

LEDGER = Path("data/corpus/ledger.jsonl")
PAGES = Path("data/corpus/extract/page_extracts")
FREEZE_V2 = Path("data/corpus/freeze_v2.json")
OUT_CLASS = Path("data/eval/datasets/quarterly_variant_classification.json")
OUT_AMBIG = Path("data/eval/datasets/quarterly_variant_ambiguous.json")

MARKER = re.compile(r"nature of report\s*(?:standalone or consolidated)?\s*"
                    r"(Standalone|Consolidated)", re.I)
AMBIG_NO_MARKER = "ambiguous_no_marker"
AMBIG_MIXED = "ambiguous_mixed"


def ledger_docs() -> dict[str, dict]:
    docs: dict[str, dict] = {}
    with LEDGER.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "document":
                continue
            data = rec.get("data") or {}
            if data.get("checksum"):
                docs[data["checksum"]] = data
    return docs


def revision_pairs() -> dict[str, list[str]]:
    """to_document_id -> [from_document_id] from revision and superseded rows."""
    pairs: dict[str, list[str]] = {}
    with LEDGER.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = rec.get("data") or {}
            old = data.get("from_document_id") or data.get("document_id")
            new = data.get("to_document_id") or data.get("superseded_by")
            if rec.get("kind") in ("revision", "superseded") and old and new:
                pairs.setdefault(new, []).append(old)
    return pairs


def page_markers(doc_id: str) -> list[dict]:
    path = PAGES / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            m = MARKER.search(rec.get("page_text") or "")
            if m:
                out.append({"pdf_page": rec["pdf_page_index"],
                            "marker": m.group(1).lower()})
    return out


def classify(markers: list[dict]) -> str:
    vals = {m["marker"] for m in markers}
    if not vals:
        return AMBIG_NO_MARKER
    if len(vals) > 1:
        return AMBIG_MIXED
    return vals.pop()


def main() -> int:
    docs = ledger_docs()
    pairs = revision_pairs()
    freeze = json.loads(FREEZE_V2.read_text(encoding="utf-8"))
    freeze_labels = {d["document_id"]: d.get("variant")
                     for d in freeze["documents"] if d["doc_type"] == "quarterly"}
    freeze_docs = set(freeze_labels)

    records = []
    for doc_id, doc in docs.items():
        if doc.get("doc_type") != "quarterly":
            continue
        markers = page_markers(doc_id)
        basis = classify(markers)
        records.append({
            "document_id": doc_id,
            "company_id": doc.get("company_id"),
            "fiscal_year": doc.get("fiscal_year"),
            "ledger_variant": doc.get("variant"),
            "freeze_variant": freeze_labels.get(doc_id),
            "in_freeze_v2": doc_id in freeze_docs,
            "basis": basis,
            "marker_pages": [m["pdf_page"] for m in markers],
            "markers": sorted({m["marker"] for m in markers}),
            "supersedes": doc.get("supersedes"),
            "revised_by": pairs.get(doc_id, []),
        })
    records.sort(key=lambda r: (r["company_id"] or "", r["fiscal_year"] or "", r["document_id"]))

    counts: dict[str, int] = {}
    for r in records:
        counts[r["basis"]] = counts.get(r["basis"], 0) + 1
    mismatches = [r for r in records
                  if r["in_freeze_v2"] and r["basis"] in ("consolidated", "standalone")
                  and r["freeze_variant"] != r["basis"]]
    classification = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "quarterly_docs_ledger": len(records),
        "basis_counts": counts,
        "freeze_mismatches": len(mismatches),
        "records": records,
    }
    OUT_CLASS.write_text(json.dumps(classification, indent=1), encoding="utf-8")

    ambiguous = [{"document_id": r["document_id"], "company_id": r["company_id"],
                  "fiscal_year": r["fiscal_year"], "freeze_variant": r["freeze_variant"],
                  "basis": r["basis"], "in_freeze_v2": r["in_freeze_v2"],
                  "marker_pages": r["marker_pages"], "markers": r["markers"],
                  "reason": ("no nature-of-report marker on any page"
                             if r["basis"] == AMBIG_NO_MARKER
                             else "both consolidated and standalone markers present")}
                 for r in records
                 if r["basis"] in (AMBIG_NO_MARKER, AMBIG_MIXED)]
    out = {
        "generated_at": classification["generated_at"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "ambiguous_count": len(ambiguous),
        "ambiguous_in_freeze_v2": sum(1 for a in ambiguous if a["in_freeze_v2"]),
        "ambiguous_docs": ambiguous,
    }
    OUT_AMBIG.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("quarterly ledger docs:", len(records))
    print("basis counts:", counts)
    print("freeze mismatches:", len(mismatches))
    print("ambiguous:", len(ambiguous), "| in freeze v2:", out["ambiguous_in_freeze_v2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
