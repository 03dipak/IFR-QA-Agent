"""Gold-set row schema + canonical hashing (EDGAR format, adapted to page-level evidence).

Canonical conventions (identical to the EDGAR capstone's gold tooling):
  * canonical JSONL: keys sorted, LF line endings, separators (",", ":"),
    ensure_ascii=False, trailing newline, one object per line;
  * row/file hash: sha256 over that canonical form;
  * last-wins on duplicate ids (replay semantics), so review stamps survive.

Adapted from EDGAR's schema (docs/decisions/005-evaluation-protocol-and-gold-schema.md
and src/collector/gold/schema.py + rows.py) to page-level evidence:
  * evidence entries carry document_id + 1-indexed pdf_page + verbatim excerpt
    (+ snippet_hash); no item, fact/XBRL context, or extraction checksum.
"""

from __future__ import annotations

import hashlib
import json

CATEGORIES = ("cross_company_comparison", "unanswerable_abstain",
              "single_fact_lookup", "numeric_from_table",
              "same_company_cross_period", "multi_hop")
ALLOCATION = {"cross_company_comparison": 0.20, "unanswerable_abstain": 0.15,
              "single_fact_lookup": 0.20, "numeric_from_table": 0.15,
              "same_company_cross_period": 0.15, "multi_hop": 0.15}
MIN_REVIEWED = 60
HOLDOUT_FRAC = 0.25
ABSTAIN_SUBTYPES = ("not_in_corpus", "company_not_in_corpus",
                     "near_miss_absent", "out_of_scope")
SPLITTER_VERSION = "gold_freeze.v1"
GOLD_SET_VERSION = "gold_v1"
GOLD_SCHEMA_VERSION = "2.0"
FREEZE_SHA_V1 = "899ac206a760af531e102e01daf99e4ff61d36cb6fc1e2e7a28cf01b094b5edd"
FREEZE_SHA_V2 = "fb2f85ae5c071cad676e42c5a1650dc5701c58fd706bdd0e663d59601827a7b5"
FREEZE_SHAS = (FREEZE_SHA_V1, FREEZE_SHA_V2)

REQUIRED_KEYS = (
    "row_id", "question_id", "category", "company_id", "period",
    "expected_behaviour", "abstention_subtype", "depends_on",
    "question_text", "expected_answer",
    "expected_value", "expected_unit", "expected_formula", "standalone_question",
    "evidence_span_text", "document_id", "pdf_page_index", "printed_page_label",
    "table_refs", "extraction_checksum",
    "relevance_rule", "must_include", "must_not_include",
    "owner", "annotated_at", "re_annotated_at", "review_date",
    "review_status", "lifecycle_status", "notes",
    "gold_schema_version", "gold_set_version", "row_revision", "freeze_sha",
)
_STATUS = ("active", "needs_review", "superseded", "retired")
_REVIEW = ("unreviewed", "self_reviewed", "peer_reviewed")
_SPLIT = ("dev", "holdout")
# Fields whose change after signing invalidates the signature.
SIGNED_FIELDS = ("question_text", "expected_answer", "answer_text",
                 "expected_value", "expected_unit", "evidence", "category",
                 "metric", "company_id", "period")


def canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8") + b"\n"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_hash(row: dict) -> str:
    return sha256_hex(canonical(row))


def signed_content_hash(row: dict) -> str:
    return sha256_hex(json.dumps({k: row.get(k) for k in SIGNED_FIELDS},
                                 ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))


def freeze_sha_of(row: dict) -> str:
    """Rows authored before the field existed belong to freeze v1."""
    return row.get("freeze_sha") or FREEZE_SHA_V1


def validate_row(row: dict) -> list[str]:
    problems = []
    row = dict(row)
    row.setdefault("freeze_sha", FREEZE_SHA_V1)
    for k in REQUIRED_KEYS:
        if k not in row:
            problems.append(f"missing key {k}")
    if row.get("category") not in CATEGORIES:
        problems.append(f"bad category {row.get('category')!r}")
    if row.get("status") not in _STATUS:
        problems.append(f"bad status {row.get('status')!r}")
    if row.get("review_status") not in _REVIEW:
        problems.append(f"bad review_status {row.get('review_status')!r}")
    if row.get("split") not in _SPLIT:
        problems.append(f"bad split {row.get('split')!r}")
    ev = row.get("evidence")
    if not isinstance(ev, list):
        problems.append("evidence empty")
    else:
        if not ev:
            # lld_eval.txt section ROW TYPE: absence-verified abstention rows
            # (not_in_corpus, company_not_in_corpus, near_miss_absent) are
            # evidence empty; out_of_scope rows carry >=1 near_miss span.
            if not (row.get("category") == "unanswerable_abstain"
                    and row.get("abstention_subtype") in ("not_in_corpus",
                                                          "company_not_in_corpus",
                                                          "near_miss_absent")):
                problems.append("evidence empty")
        for i, e in enumerate(ev):
            if not isinstance(e, dict):
                problems.append(f"evidence[{i}] not an object")
                continue
            did = e.get("document_id")
            if not (isinstance(did, str) and len(did) == 64):
                problems.append(f"evidence[{i}].document_id not 64-hex")
            if not isinstance(e.get("pdf_page"), int) or e["pdf_page"] < 1:
                problems.append(f"evidence[{i}].pdf_page not a 1-indexed int")
            if not (e.get("excerpt") or "").strip():
                problems.append(f"evidence[{i}].excerpt empty")
    ab = row.get("abstention_subtype")
    if row.get("category") == "unanswerable_abstain":
        if ab not in ABSTAIN_SUBTYPES:
            problems.append(f"abstain row with bad abstention_subtype {ab!r}")
    elif ab is not None:
        problems.append("non-abstain row carries abstention_subtype")
    return problems
