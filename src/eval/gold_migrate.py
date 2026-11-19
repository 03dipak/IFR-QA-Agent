"""One-off migration: gold_data.jsonl schema v1 -> v2 (lld_eval.txt section 3).

33-field row. Fields conditional-null, never absent. Extra non-LLD fields are
kept as extras (schema change => new gold_schema_version; row ids preserved;
reviewers list/signed_sha256 re-stamped under the new schema with a migration
note; review_status/revision untouched).

Run: python -m src.eval.gold_migrate
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from .gold_rows import load_rows
from .gold_schema import canonical, signed_content_hash
from .textnorm import normalise_for_hash

GOLD = "data/eval/datasets/gold_data.jsonl"
PAGE_DIR = "data/corpus/extract/page_extracts"
NOTE = "[migrated to schema 2.0 2026-11-08; signed fields unchanged, re-stamped]"


def page_checksum(doc_id: str, page: int) -> str | None:
    path = Path(PAGE_DIR) / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pdf_page_index") == page:
                return hashlib.sha256(
                    normalise_for_hash(rec.get("page_text") or "").encode()).hexdigest()
    return None


def migrate_row(r: dict) -> dict:
    ev = (r.get("evidence") or [])
    primary = ev[0] if ev else {}
    primary_page = primary.get("pdf_page")
    is_abstain = r.get("category") == "unanswerable_abstain"

    row = {
        # identity (5)
        "row_id": r.get("id"),
        "question_id": None,
        "category": r.get("category"),
        "company_id": r.get("company_id"),
        "period": r.get("fiscal_year") or r.get("period") or "",
        # behaviour (3)
        "expected_behaviour": r.get("expected_behaviour"),
        "abstention_subtype": r.get("abstention_subtype"),
        "depends_on": r.get("depends_on"),
        # question (2)
        "question_text": r.get("question"),
        "expected_answer": r.get("expected_answer"),
        # answer (4)
        "expected_value": r.get("answer_value"),
        "expected_unit": r.get("expected_unit"),
        "expected_formula": None,
        "standalone_question": r.get("standalone_question"),
        # retrieval (1)
        "relevance_rule": "none" if is_abstain else "primary",
        # scoring (2)
        "must_include": r.get("must_include") or [],
        "must_not_include": r.get("must_not_include") or [],
        # provenance (10)
        "owner": r.get("annotator"),
        "annotated_at": r.get("annotated_at"),
        "re_annotated_at": r.get("re_annotated_at"),
        "review_date": r.get("re_annotated_at") or r.get("annotated_at"),
        "review_status": r.get("review_status"),
        "lifecycle_status": {"active": "active", "rejected": "retired",
                             "superseded": "superseded"}.get(r.get("status"), "retired"),
        "notes": (r.get("notes") or "") + (" " if r.get("notes") else "") + NOTE,
        "gold_schema_version": "2.0",
        "gold_set_version": r.get("gold_set_version") or "gold_v1",
        "row_revision": r.get("row_revision") or 0,
        # extras (kept)
        "id": r.get("id"),
        "question": r.get("question"),
        "answer_text": r.get("answer_text") or r.get("expected_answer"),
        "answer_value": r.get("answer_value"),
        "expected_unit_raw": r.get("expected_unit_raw"),
        "fiscal_year": r.get("fiscal_year") or r.get("period"),
        "doc_type": r.get("doc_type"),
        "variant": r.get("variant"),
        "version": r.get("version"),
        "reviewer": r.get("reviewer"),
        "reviewers": r.get("reviewers") or [],
        "signed_sha256": "",
        "sub_answers": r.get("sub_answers"),
        "evidence": ev,
    }
    if is_abstain:
        row.update(evidence_span_text=None, document_id=None, pdf_page_index=None,
                   printed_page_label=None, table_refs=None, extraction_checksum=None)
    else:
        checksum = page_checksum(primary.get("document_id"), primary_page) if (
            primary.get("document_id") and isinstance(primary_page, int)) else None
        row.update(evidence_span_text=(primary.get("excerpt") or "").strip() or None,
                   document_id=primary.get("document_id"),
                   pdf_page_index=primary_page,
                   printed_page_label=None,
                   table_refs=None,
                   extraction_checksum=checksum)
    row["signed_sha256"] = signed_content_hash(row)
    return row


def main() -> int:
    rows = load_rows(GOLD)
    backup = Path(GOLD).with_suffix(".jsonl.bak-migrate-20261108T1000Z")
    backup.write_bytes(Path(GOLD).read_bytes())
    out = [migrate_row(r) for r in rows]
    with open(GOLD, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n")
    print(f"migrated {len(out)} rows to schema 2.0; backup={backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
