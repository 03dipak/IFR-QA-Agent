"""One-off: convert data/gold/candidates.csv (100 draft rows) to JSONL drafts.

EDGAR row schema adapted to page-level evidence:
  * evidence entries carry document_id + 1-indexed pdf_page + verbatim excerpt
    (+ snippet_hash via src.eval.textnorm.snippet_hash); no item/fact/XBRL.
  * period = fiscal_year of the last-listed evidence document (annual FY or
    quarterly label, e.g. FY2026 / Q2FY2027).
  * metric = first Ind AS statement line found in the question (case-insens
    substring over the INDAS lists), else "".
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from src.eval.textnorm import snippet_hash

FREEZE = "data/corpus/freeze_v1.json"
INDAS = [
    "revenue from operations", "total income", "total expenses",
    "profit before tax", "profit for the year", "profit for the period",
    "total assets", "total equity", "total liabilities",
]
MONEY_WORDS = ("crore", "lakh", "₹", "rs", "inr")


def _load_freeze():
    docs = json.load(open(FREEZE, encoding="utf-8"))["documents"]
    return {d["document_id"]: d for d in docs}


def _metric(question: str) -> str:
    q = question.lower()
    for m in INDAS:
        if m in q:
            return m
    return ""


def _answer_value(answer: str):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", answer.replace("(", "-").replace(")", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def convert(inp="data/gold/candidates.csv", oup="data/eval/datasets/packs/drafts.jsonl") -> int:
    meta = _load_freeze()
    rows_out = []
    with open(inp, newline="", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=1):
            ids = r["doc_id"].split("|")
            page_s = r["page"].split("|")
            excerpts = [e.strip() for e in r["evidence_excerpt"].split("|||")]
            try:
                pages = [int(p) for p in page_s]
            except ValueError:
                pages = []
            evidence = []
            versions = []
            for j, (did, p) in enumerate(zip(ids, pages)):
                exc = excerpts[j] if j < len(excerpts) else (excerpts[0] if excerpts else "")
                ev = {
                    "document_id": did,
                    "pdf_page": p,
                    "evidence_role": "primary" if j == 0 else "comparison",
                    "excerpt": exc,
                    "snippet_hash": snippet_hash(exc) if exc else None,
                    "version": (meta.get(did) or {}).get("version"),
                }
                evidence.append(ev)
                versions.append(ev["version"])
            versions = [v for v in versions if v is not None]
            row_version = versions[0] if len(set(versions)) == 1 else None
            last = meta.get(ids[-1]) or {}
            cat = r["category"]
            is_abstain = cat == "unanswerable_abstain"
            primary = evidence[0] if evidence else {}
            row = {
                "row_id": f"G-{i:03d}",
                "question_id": None,
                "category": cat,
                "company_id": (meta.get(ids[0]) or {}).get("company_id", ""),
                "doc_type": (meta.get(ids[0]) or {}).get("doc_type", "single"),
                "period": last.get("fiscal_year", ""),
                "fiscal_year": last.get("fiscal_year", ""),
                "variant": (meta.get(ids[0]) or {}).get("variant"),
                "metric": _metric(r["question"]),
                "question_text": r["question"],
                "question": r["question"],
                "expected_answer": r["proposed_answer"],
                "expected_value": _answer_value(r["proposed_answer"]),
                "answer_text": r["proposed_answer"],
                "answer_value": _answer_value(r["proposed_answer"]),
                "expected_unit": ("money" if any(w in (r["units"] or "").lower()
                                                for w in MONEY_WORDS) else None),
                "expected_unit_raw": r["units"],
                "evidence": evidence,
                "must_include": [],
                "must_not_include": [],
                "abstention_subtype": ("not_in_corpus" if cat == "unanswerable_abstain" else None),
                "expected_behaviour": ("abstain" if cat == "unanswerable_abstain" else "answer"),
                "expected_formula": None,
                "depends_on": None,
                "standalone_question": None,
                "sub_answers": None,
                "version": row_version,
                "status": "active",
                "review_status": "unreviewed",
                "reviewer": None,
                "reviewers": [],
                "row_revision": 0,
                "notes": "",
                "split": "dev",
                "annotated_at": None,
                "re_annotated_at": None,
                "annotator": None,
                "gold_schema_version": "2.0",
                "gold_set_version": "gold_v1",
                "printed_page_label": None,
                "table_refs": None,
                "extraction_checksum": None,
                "relevance_rule": "none" if cat == "unanswerable_abstain" else "primary",
                "owner": None,
                "reviewers": [],
                "signed_sha256": None,
                "lifecycle_status": "active",
                "review_date": None,
                "review_status": "unreviewed",
                "status": "active",
                "version": row_version,
            }
            # LLD 33 top-level scalars (null if unknown) from primary evidence
            if is_abstain:
                row.update(evidence_span_text=None, document_id=None,
                           pdf_page_index=None, extraction_checksum=None)
            else:
                import hashlib as _h
                from src.eval.textnorm import normalise_for_hash as _n
                try:
                    with open(f"data/corpus/extract/page_extracts/{primary['document_id']}.pages.jsonl",
                              encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            if not line.strip():
                                continue
                            rec = json.loads(line)
                            if rec.get("pdf_page_index") == primary_page:
                                row["extraction_checksum"] = _h.sha256(
                                    _n(rec.get("page_text") or "").encode()).hexdigest()
                                break
                except OSError:
                    row["extraction_checksum"] = None
                row.update(evidence_span_text=row_evidence_span,
                           document_id=primary.get("document_id"),
                           pdf_page_index=primary_page)
            row["printed_page_label"] = None
            row["table_refs"] = None
            row["row_revision"] = 0
            rows_out.append(row)
    out = Path(oup)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n")
    print(f"wrote {len(rows_out)} drafts to {out}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="src.eval.gold_convert")
    ap.add_argument("--input", default="data/gold/candidates.csv")
    ap.add_argument("--output", default="data/eval/datasets/packs/drafts.jsonl")
    args = ap.parse_args()
    sys.exit(convert(args.input, args.output))
