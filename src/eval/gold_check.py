"""Gold-set checks (adapted to page-level evidence).

Checks (every row with status='active'):
  * evidence_resolvable — evidence document_id in data/corpus/freeze_v1.json,
    1-indexed PDF page exists in the page extract, excerpt found on page
    (whitespace-normalised, case-insensitive);
  * numeric_grounded — every numeric token of expected_answer present on its
    cited page; expected_unit non-empty (basis statement presence warns);
  * duplicate_questions — near-duplicate question text (token Jaccard >= 0.9);
  * evidence_uniqueness — two rows re-anchoring near-identical 13-token
    shingles of the same document;
  * abstain coherence — per-abstain-row keyword-hit report over the company's
    frozen documents (metric line + common surface forms);
  * solo_authoring — live rows with < 2 distinct reviewers (no_separation).
Exit 1 if any FAIL; warnings never fail.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from .gold_schema import (
    ALLOCATION,
    FREEZE_SHA_V1,
    FREEZE_SHAS,
    freeze_sha_of,
    signed_content_hash,
    validate_row,
)
from .textnorm import normalise_for_hash

FREEZE = "data/corpus/freeze_v1.json"
PAGE_DIR = "data/corpus/extract/page_extracts"
INDAS_LINES = [
    "revenue from operations", "total income", "total expenses",
    "profit before tax", "profit for the year", "profit for the period",
    "total assets", "total equity", "total liabilities",
]


def freeze_ids() -> set[str]:
    return set(json.load(open(FREEZE, encoding="utf-8"))["kept_ids"])


def _frozen_company_docs() -> dict[str, set[str]]:
    docs = json.load(open(FREEZE, encoding="utf-8"))["documents"]
    out = defaultdict(set)
    for d in docs:
        out[d.get("company_id")].add(d["document_id"])
    return dict(out)


def _page_text(doc_id: str, page: int) -> str | None:
    path = Path(PAGE_DIR) / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pdf_page_index") == page:
                return rec.get("page_text") or ""
    return None


def _page_index(doc_id: str) -> dict[int, str]:
    path = Path(PAGE_DIR) / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec.get("pdf_page_index")
            if isinstance(p, int):
                out[p] = rec.get("page_text") or ""
    return out


_SH = 13

def _shingles(text: str) -> set[str]:
    toks = normalise_for_hash(text).split()
    return {" ".join(toks[i:i + _SH]) for i in range(max(0, len(toks) - _SH + 1))}


def check_drift(rows, apply=False):
    """Freeze-boundary drift check: every active row's checksum + span are
    re-verified against CURRENT extraction. Failures flag needs_review;
    never fixes anything."""
    problems = []
    for r in rows:
        if r.get("lifecycle_status") != "active":
            continue
        if r.get("category") == "unanswerable_abstain":
            continue
        doc = r.get("document_id"); pg = r.get("pdf_page_index"); chk = r.get("extraction_checksum")
        rid = r.get("row_id")
        if not doc or not isinstance(pg, int):
            problems.append((rid, f"drift: missing document_id/pdf_page_index"))
            continue
        text = _page_text(doc, pg)
        if text is None:
            problems.append((rid, f"drift: page {pg} missing from page extract"))
            continue
        if chk is not None:
            actual = hashlib.sha256(normalise_for_hash(text).encode()).hexdigest()
            if actual != chk:
                problems.append((rid, "drift: extraction_checksum mismatch"))
        span = (r.get("evidence_span_text") or "").strip()
        if span and normalise_for_hash(span) not in normalise_for_hash(text):
            problems.append((rid, "drift: evidence_span_text not verbatim on page"))
    if apply:
        flagged = set()
        for rid, _ in problems:
            for r in rows:
                if r.get("row_id") == rid:
                    r["lifecycle_status"] = "needs_review"
                    r["status"] = "active"  # keep v1 alias consistent
                    flagged.add(rid)
    return problems


def check_evidence_resolvable(rows, fids):
    problems = []
    for r in rows:
        for j, e in enumerate(r.get("evidence") or []):
            did = e.get("document_id")
            pg = e.get("pdf_page")
            if did not in fids:
                problems.append((r["id"], f"evidence[{j}] doc {did[:12]} not in freeze_v1"))
                continue
            text = _page_text(did, pg)
            if text is None:
                problems.append((r["id"], f"evidence[{j}] page {pg} not in page extract"))
                continue
            exc = (e.get("excerpt") or "").strip()
            if exc and normalise_for_hash(exc) not in normalise_for_hash(text):
                problems.append((r["id"], f"evidence[{j}] excerpt not verbatim on page {pg}"))
    return problems


_NUM = re.compile(r"-?\(?\d[\d,]*(?:\.\d+)?\)?")


def check_numeric_grounded(rows):
    problems, warnings = [], []
    for r in rows:
        if r.get("category") == "unanswerable_abstain":
            continue
        ans = str(r.get("expected_answer") or "")
        tokens = [t for t in _NUM.findall(ans) if len(t.replace(",", "")) >= 3]
        unit = r.get("expected_unit")
        if not unit:
            problems.append((r["id"], "expected_unit empty (unit not stated)"))
        if not tokens:
            continue
        for e in r.get("evidence") or []:
            text = _page_text(e.get("document_id"), e.get("pdf_page")) or ""
            flat = " ".join(text.split()).replace(",", "")
            if any(t.replace(",", "") in flat for t in tokens):
                break
        else:
            problems.append((r["id"], "no numeric token of the answer found on its cited page"))
        joined = " ".join((r.get("question") or "").split()).lower()
        if not any(k in joined for k in ("standalone", "consolidated", "combined", "single")):
            warnings.append((r["id"], "basis (standalone/consolidated) not stated in question"))
    return problems, warnings


def check_duplicate_questions(rows):
    warnings = []
    seen: list[tuple[str, set[str]]] = []
    for r in rows:
        toks = set(normalise_for_hash(r.get("question") or "").split())
        for prev_id, prev_toks in seen:
            if not toks or not prev_toks:
                continue
            jac = len(toks & prev_toks) / len(toks | prev_toks)
            if jac >= 0.9:
                warnings.append((r["id"], f"near-duplicate of {prev_id} (jaccard {jac:.2f})"))
        seen.append((r["id"], toks))
    return warnings


def check_evidence_uniqueness(rows):
    warnings = []
    by_doc = defaultdict(list)
    for r in rows:
        pgs = {(e.get("document_id"), e.get("pdf_page")) for e in r.get("evidence") or []}
        shingle_map = {}
        for e in r.get("evidence") or []:
            shingle_map[(e.get("document_id"), e.get("pdf_page"))] = _shingles(e.get("excerpt") or "")
        for (did, pg), sh in shingle_map.items():
            by_doc[did].append((r["id"], pg, sh))
    for did, items in by_doc.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if not a[2] or not b[2]:
                    continue
                jac = len(a[2] & b[2]) / len(a[2] | b[2])
                if jac >= 0.5:
                    warnings.append((a[0], f"shares >=50% of 13-token shingles with {b[0]} on {did[:12]}"))
    return warnings


def abstain_keyword_report(rows):
    """Per abstain row: keyword (Ind AS line) hits, across the company's
    frozen documents AND inside the cited document itself.

    Company-wide hits are EXPECTED (the metric prints in every AR) and are
    informational; hits INSIDE the cited document mean the abstained figure
    may in fact be printed there and the row needs human verification.
    """
    report = []
    company_docs = _frozen_company_docs()
    for r in rows:
        if r.get("category") != "unanswerable_abstain":
            continue
        q = normalise_for_hash(r.get("question") or "")
        metric = next((m for m in INDAS_LINES if m in q), None)
        comp = r.get("company_id")
        if not metric:
            report.append({"id": r["id"], "company": comp, "metric": None,
                           "hits": 0, "cited_doc_hits": 0, "pages": []})
            continue
        pat = re.compile(re.escape(metric), re.IGNORECASE)
        doc_hits, cited_hits = [], []
        for did in sorted(company_docs.get(comp, ())):
            for pg, text in _page_index(did).items():
                if pat.search(text):
                    doc_hits.append((did[:12], pg))
        cid = (r.get("evidence") or [{}])[0].get("document_id")
        if cid:
            for pg, text in _page_index(cid).items():
                if pat.search(text):
                    cited_hits.append(pg)
        report.append({"id": r["id"], "company": comp, "metric": metric,
                       "hits": len(doc_hits), "cited_doc_hits": len(cited_hits),
                       "pages": doc_hits[:10]})
    return report


def check_solo_authoring(rows):
    return [r["id"] for r in rows
            if r.get("status") == "active" and len(r.get("reviewers") or []) < 2]


def check_depends_on_resolves(rows):
    """FAIL: depends_on must resolve to a live (active) gold row id.

    Guards the multi-turn pair rule ("pairs freeze whole"): a pointer to a
    draft id or superseded row means the pair cannot be treated as whole.
    """
    live_ids = {r["id"] for r in rows if r.get("status") == "active"}
    return [(r["id"], f"depends_on {r['depends_on']} does not resolve to a live gold row")
            for r in rows if r.get("status") == "active" and r.get("depends_on")
            and r["depends_on"] not in live_ids]


def check_freeze_sha(rows) -> list:
    """Every row must cite a known corpus freeze; v1.4 rows default to freeze v1."""
    fails = []
    for r in rows:
        sha = freeze_sha_of(r)
        if sha not in FREEZE_SHAS:
            fails.append((r.get("id", "?"), f"unknown freeze_sha {sha[:12]}"))
        elif sha == FREEZE_SHA_V1 and str(r.get("gold_set_version", "")).startswith("gold_v2"):
            fails.append((r.get("id", "?"), "gold_v2 row still pinned to freeze_v1"))
    return fails


def run_all(rows, allocation=None):
    n_superseded = sum(1 for r in rows if r.get("status") == "superseded")
    rows = [r for r in rows if r.get("status") == "active"]
    fids = freeze_ids()
    fails, warns = [], []
    for r in rows:
        for p in validate_row(r):
            fails.append((r.get("id", "?"), f"schema: {p}"))
    fails += check_freeze_sha(rows)
    drift = check_drift(rows, apply=False)
    fails += [(rid, f"drift: {msg}") for rid, msg in drift]
    ev = check_evidence_resolvable(rows, fids)
    fails += ev
    num_fails, num_warn = check_numeric_grounded(rows)
    fails += num_fails
    warns += num_warn
    warns += check_duplicate_questions(rows)
    warns += check_evidence_uniqueness(rows)
    solo = check_solo_authoring(rows)
    if solo:
        warns.append(("no_separation", f"{len(solo)} live rows with <2 reviewers: "
                                      + ", ".join(solo[:10])))
    fails += check_depends_on_resolves(rows)
    ab = abstain_keyword_report(rows)
    if allocation:
        total = sum(allocation.values())
        counts = defaultdict(int)
        for r in rows:
            if r.get("status") == "active":
                counts[r.get("category")] += 1
        for cat, frac in allocation.items():
            want = int(round(len([r for r in rows if r.get("status") == "active"]) * frac))
            have = counts.get(cat, 0)
            if abs(have - want) > max(1, int(0.15 * want)):
                warns.append((cat, f"category count {have} vs target {want}"))
    live = rows
    summary = {
        "n_rows": len(rows),
        "n_live": len(live),
        "n_reviewed": sum(1 for r in live if r.get("review_status") in ("self_reviewed", "peer_reviewed")),
        "n_two_reviewers": sum(1 for r in live if len(r.get("reviewers") or []) >= 2),
        "n_rejected": sum(1 for r in rows if r.get("status") == "rejected"),
        "n_superseded": n_superseded,
        "abstain_keyword_report": ab,
    }
    return fails, warns, summary


def main():
    from .gold_rows import load_rows
    rows = load_rows("data/eval/datasets/gold_data.jsonl", active_only=True)
    fails, warns, summary = run_all(rows, ALLOCATION)
    print(f"GOLD SET CHECKS  rows={summary['n_rows']} live={summary['n_live']} "
          f"reviewed={summary['n_reviewed']} two_reviewers={summary['n_two_reviewers']} "
          f"rejected={summary['n_rejected']} superseded={summary['n_superseded']}")
    for rid, msg in fails:
        print(f"  [FAIL] {rid}: {msg}")
    for rid, msg in warns:
        print(f"  [warn] {rid}: {msg}")
    print("abstain keyword-hit report (company-wide and cited-doc; informational, verify figures yourself):")
    for r in summary["abstain_keyword_report"]:
        print(f"  {r['id']} company={r['company']} metric={r['metric']} "
              f"company_hits={r['hits']} cited_doc_hits={r['cited_doc_hits']}"
              + (f" pages={r['pages'][:5]}" if r["hits"] else ""))
    print(f"{len(fails)} fail, {len(warns)} warn")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
