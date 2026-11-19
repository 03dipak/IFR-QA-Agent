"""Gold authoring tooling (candidate generation + verification of drafts).

Contract: docs/llds/lld_eval.txt section 5 (gold-row contract) and the gold
authoring recipe. Generates DRAFT candidate questions from table_records and
page_extracts only. Cites document id + 1-indexed PDF page + row label; never
a chunk_id. Every produced row is status=draft with verified_by blank - a
human must verify each against the PDF before anything becomes gold.

CSV columns (data/gold/candidates.csv):
    question, category, proposed_answer, doc_id, page, evidence_excerpt,
    units, verified_by, status

Categories: single_fact_lookup, numeric_from_table, same_company_cross_period,
cross_company_comparison, multi_hop, unanswerable_abstain.
Templates + category mix: src/eval/gold_templates.py (NSE/Ind AS port,
provenance noted there). Excerpt normalisation: src/eval/textnorm.py
(copied verbatim from the EDGAR capstone, see src/eval/textnorm.py).

    python -m src.eval.gold_candidates --generate \
        --output data/gold/candidates.csv
    python -m src.eval.gold_candidates --verify data/gold/candidates.csv

Verify checks every row with verified_by non-blank: page exists in the
document's page extract, the excerpt is found verbatim on that page, and a
numeric proposed_answer appears on that page. Failures are printed; exit != 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from .gold_templates import (
    INDAS_BALANCE_LINES,
    INDAS_CASHFLOW_LINES,
    INDAS_INCOME_LINES,
    canonicalise_unit,
    period_phrase,
    question,
)
from .textnorm import normalise_for_hash

PAGE_EXTRACT_DIR = "data/corpus/extract/page_extracts"
LEDGER = "data/corpus/ledger.jsonl"

DENSE_COMPANIES = ("bhel", "maruti_suzuki", "deepak_nitrite", "srf",
                   "berger_paints", "dabur", "bosch")
DENSE_YEARS = ("FY2024", "FY2025", "FY2026")


def _in_slice(doc_type, fiscal_year, company_id) -> bool:
    if doc_type == "annual" and fiscal_year in DENSE_YEARS:
        return True
    if doc_type == "quarterly" and company_id in DENSE_COMPANIES:
        return True
    return False

COLUMNS = ["question", "category", "proposed_answer", "doc_id", "page",
           "evidence_excerpt", "units", "verified_by", "status"]

_NUM_RE = re.compile(r"-?\(?\d[\d,]*(?:\.\d+)?\)?")


def _kept_docs(ledger_path: str) -> dict[str, dict]:
    """checksum -> doc meta for gate-kept + registered pre-manual rows."""
    merged: dict[str, dict] = {}
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        d = row.get("data", {})
        if row.get("kind") != "document":
            continue
        cur = merged.setdefault(d["checksum"], {})
        for key in ("company_id", "doc_type", "fiscal_year", "gate_decision",
                    "variant", "extraction_status"):
            if cur.get(key) in (None, "", []) and d.get(key) is not None:
                cur[key] = d[key]
    return {c: d for c, d in merged.items()
            if d.get("gate_decision") in (None, "keep")}


def _pages(doc_id: str) -> dict[int, str]:
    path = Path(PAGE_EXTRACT_DIR) / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return {}
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["pdf_page_index"]] = rec.get("page_text") or ""
    return out


def _excerpt(page_text: str, label: str) -> str:
    """Verbatim excerpt from the page carrying the row label (+ numbers)."""
    if not label:
        return ""
    needle = " ".join(label.split())[:24]
    for line in page_text.splitlines():
        flat = " ".join(line.split())
        if needle.lower() in flat.lower():
            return flat[:240]
    return ""


_METRIC_ROWS = {
    "income_statement": [(m, m) for m in INDAS_INCOME_LINES],
    "balance_sheet": [(m, m) for m in INDAS_BALANCE_LINES],
    "cash_flow": INDAS_CASHFLOW_LINES,  # parked: variable labels are noisy
}


def _row_value(cells: list[dict]) -> str | None:
    """First parseable numeric cell (skip the label column)."""
    for cell in cells:
        raw = (cell.get("raw") or "").strip()
        if _NUM_RE.fullmatch(raw):
            return raw
    return None


def _candidate_pool(ledger_path=LEDGER) -> list[dict]:
    """Evidence packs: one per (document, statement table, metric row).
    Consolidated or standalone only; carries unit canon derived from the
    cited page, plus variant/version from the freeze manifest. Units with
    no page evidence are dropped upstream — never guessed.
    """
    kept = _kept_docs(ledger_path)
    freeze_docs = json.load(open("data/corpus/freeze_v1.json", encoding="utf-8"))["documents"]
    fmeta = {d["document_id"]: d for d in freeze_docs}
    pool: list[dict] = []
    rec_root = Path("data/corpus/extract/table_records")
    pages_cache: dict[str, dict[int, str]] = {}
    for path in sorted(rec_root.glob("*.tables.jsonl")):
        doc_id = path.name[: -len(".tables.jsonl")]
        meta = kept.get(doc_id)
        if meta is None:
            continue
        if not _in_slice(meta.get("doc_type"), meta.get("fiscal_year"),
                         meta.get("company_id")):
            continue
        fdoc = fmeta.get(doc_id, {})
        doc_pages = pages_cache.get(doc_id)
        if doc_pages is None:
            doc_pages = pages_cache[doc_id] = _pages(doc_id)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("statement_kind")
            metrics = _METRIC_ROWS.get(kind)
            if not metrics:
                continue
            basis = rec.get("reporting_basis")
            if basis not in ("consolidated", "standalone"):
                continue
            page_no = rec.get("page_start")
            page_text = doc_pages.get(page_no, "")
            units = canonicalise_unit((rec.get("unit_header") or "").strip(),
                                      page_text)
            if units is None:
                continue
            for needle, label in metrics:
                for row in rec.get("rows", []):
                    label_raw = (row.get("row_label_raw") or "").strip()
                    if not label_raw or needle not in label_raw.lower():
                        continue
                    value = _row_value(row.get("cells", []))
                    if value is None:
                        continue
                    excerpt = _excerpt(page_text, label_raw)
                    if not excerpt:
                        continue
                    pool.append({
                        "company": meta.get("company_id", "?"),
                        "fiscal_year": meta.get("fiscal_year", "?"),
                        "doc_type": meta.get("doc_type", "?"),
                        "doc_id": doc_id,
                        "page": page_no,
                        "statement_kind": kind,
                        "basis": basis,
                        "variant": fdoc.get("variant"),
                        "version": fdoc.get("version"),
                        "row_label": label_raw,
                        "metric": label,
                        "value": value,
                        "units": units,
                        "excerpt": excerpt,
                    })
                    break
    return pool


def _comp(company: str) -> str:
    return company.replace("_", " ").title()


def _period(meta_or_pack: dict) -> str:
    return period_phrase(meta_or_pack.get("doc_type", "annual"),
                         meta_or_pack.get("fiscal_year") or meta_or_pack.get("period", ""))


def generate(output: str = "data/gold/candidates.csv", limit: int = 100,
             ledger_path=LEDGER) -> int:
    pool = _candidate_pool(ledger_path)
    used_keys: set[str] = set()
    used_cm: set[tuple] = set()
    accepted_questions: list[str] = []
    drafts: list[dict] = []

    def key(metric: str, fy: str, company: str, variant) -> str:
        return f"{metric}|{fy}|{company}|{variant}"

    def register(metric: str, fy: str, company: str, variant) -> bool:
        k = key(metric, fy, company, variant)
        if k in used_keys:
            return False
        used_keys.add(k)
        return True

    def register_cm(company: str, metric: str) -> bool:
        k = (company, metric)
        if k in used_cm:
            return False
        used_cm.add(k)
        return True

    bykey = defaultdict(dict)
    for p in pool:
        bykey[(p["statement_kind"], p["metric"], p["fiscal_year"],
                p["basis"], p["company"])][p["doc_id"]] = p

    # ---- cross-company_comparison (20): all six sector pairs covered ----
    pairs = [("deepak_nitrite", "srf"), ("maruti_suzuki", "bosch"),
             ("dabur", "berger_paints"), ("aurobindo_pharma", "dr_reddys"),
             ("hcl", "tech_mahindra"), ("tvs_motor", "ashok_leyland")]
    combos = []
    for fy in ("FY2025", "FY2026", "FY2024"):
        for a, b in pairs:
            for metric, kind in [("revenue from operations", "income_statement"),
                                 ("total income", "income_statement"),
                                 ("profit before tax", "income_statement"),
                                 ("profit for the year", "income_statement"),
                                 ("total assets", "balance_sheet"),
                                 ("total equity", "balance_sheet"),
                                 ("total expenses", "income_statement"),
                                 ("total liabilities", "balance_sheet")]:
                combos.append((metric, kind, fy, a, b))
    for metric, kind, fy, a, b in combos:
        if sum(1 for d in drafts if d["category"] == "cross_company_comparison") >= 20:
            break
        pa = bykey.get((kind, metric, fy, "consolidated", a))
        pb = bykey.get((kind, metric, fy, "consolidated", b))
        if not pa or not pb:
            continue
        a_doc, a_p = next(iter(pa.items()))
        b_doc, b_p = next(iter(pb.items()))
        if not register(metric, fy, a, a_p.get("variant")):
            continue
        if not register(metric, fy, b, b_p.get("variant")):
            continue
        q = question("cross_company_comparison", a=_comp(a), b=_comp(b),
                     fy=fy, metric=metric, unit=a_p["units"],
                     period=_period(a_p), basis="consolidated",
                     _avoid=accepted_questions)
        if q is None:
            used_keys.discard(key(metric, fy, a, a_p.get("variant")))
            used_keys.discard(key(metric, fy, b, b_p.get("variant")))
            continue
        drafts.append({
            "question": q, "category": "cross_company_comparison",
            "proposed_answer": f"{a}={a_p['value']}; {b}={b_p['value']}",
            "doc_id": f"{a_doc}|{b_doc}", "page": f"{a_p['page']}|{b_p['page']}",
            "evidence_excerpt": a_p["excerpt"] + " ||| " + b_p["excerpt"],
            "units": a_p["units"], "version": a_p.get("version"),
        })
        accepted_questions.append(q)

    # ---- unanswerable_abstain (15), distinct companies ----
    have = defaultdict(set)
    for p in pool:
        have[p["doc_id"]].add((p["statement_kind"], p["metric"]))
    seen_cos = set()
    for doc_id in sorted({p["doc_id"] for p in pool}):
        if sum(1 for d in drafts if d["category"] == "unanswerable_abstain") >= 15:
            break
        meta = _kept_docs(ledger_path).get(doc_id)
        if meta is None:
            continue
        comp = meta.get("company_id")
        fy = meta.get("fiscal_year")
        variant = None
        if comp in seen_cos:
            continue
        kinds = {k for k, _ in have.get(doc_id, set())}
        missing = [k for k in ("balance_sheet", "income_statement") if k not in kinds]
        target_kind = missing[0] if missing else None
        target_metric = _METRIC_ROWS[target_kind][0][1] if target_kind else None
        if target_metric is None:
            for k2, m2 in have.get(doc_id, set()):
                unused = [mm for _, mm in _METRIC_ROWS[k2]
                          if mm != m2 and all(mm != m3 for _, m3 in have[doc_id])]
                if unused:
                    target_metric = unused[0]
                    break
        if target_metric is None:
            continue
        anchor = next((p for p in pool if p["doc_id"] == doc_id
                       and p["metric"] != target_metric), None)
        if anchor is None:
            continue
        if not register(target_metric, fy or "?", comp, anchor.get("variant")):
            continue
        if not register_cm(comp, target_metric):
            continue
        seen_cos.add(comp)
        q = question("unanswerable_abstain", comp=_comp(comp),
                     metric=target_metric, period=_period(anchor),
                     basis=anchor["basis"], _avoid=accepted_questions)
        if q is None:
            used_keys.discard(key(target_metric, fy or "?", comp, anchor.get("variant")))
            used_cm.discard((comp, target_metric))
            continue
        drafts.append({
            "question": q, "category": "unanswerable_abstain",
            "proposed_answer": "ABSTAIN",
            "doc_id": doc_id, "page": anchor["page"],
            "evidence_excerpt": anchor["excerpt"],
            "units": "", "version": anchor.get("version"),
        })
        accepted_questions.append(q)

    # ---- single_fact_lookup (20) ----
    sf = 0
    for p in sorted(pool, key=lambda p: (p["company"], p["fiscal_year"], p["metric"])):
        if sf >= 20:
            break
        if p["fiscal_year"] not in ("FY2024", "FY2025", "FY2026"):
            continue
        if p["metric"] not in ("total assets", "total equity", "total liabilities"):
            continue
        if not register(p["metric"], p["fiscal_year"], p["company"], p["variant"]):
            continue
        if not register_cm(p["company"], p["metric"]):
            continue
        q = question("single_fact_lookup", comp=_comp(p["company"]),
                     metric=p["metric"], period=_period(p),
                     basis=p["basis"], unit=p["units"],
                     _avoid=accepted_questions)
        if q is None:
            used_keys.discard(key(p["metric"], p["fiscal_year"], p["company"], p["variant"]))
            used_cm.discard((p["company"], p["metric"]))
            continue
        drafts.append({
            "question": q, "category": "single_fact_lookup",
            "proposed_answer": p["value"],
            "doc_id": p["doc_id"], "page": p["page"],
            "evidence_excerpt": p["excerpt"], "units": p["units"],
            "version": p.get("version"),
        })
        accepted_questions.append(q)
        sf += 1

    # ---- numeric_from_table (15) ----
    nf = 0
    for p in sorted(pool, key=lambda p: (p["company"], p["fiscal_year"], p["metric"])):
        if nf >= 15:
            break
        if p["metric"] not in ("profit before tax", "profit for the year",
                               "total income", "revenue from operations",
                               "total expenses"):
            continue
        if not register(p["metric"], p["fiscal_year"], p["company"], p["variant"]):
            continue
        if not register_cm(p["company"], p["metric"]):
            continue
        q = question("numeric_from_table", comp=_comp(p["company"]),
                     metric=p["metric"], period=_period(p),
                     basis=p["basis"], unit=p["units"],
                     _avoid=accepted_questions)
        if q is None:
            used_keys.discard(key(p["metric"], p["fiscal_year"], p["company"], p["variant"]))
            used_cm.discard((p["company"], p["metric"]))
            continue
        drafts.append({
            "question": q, "category": "numeric_from_table",
            "proposed_answer": p["value"],
            "doc_id": p["doc_id"], "page": p["page"],
            "evidence_excerpt": p["excerpt"], "units": p["units"],
            "version": p.get("version"),
        })
        accepted_questions.append(q)
        nf += 1

    # ---- same_company_cross_period (15) ----
    comp_metric: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for p in pool:
        if p["fiscal_year"] not in ("FY2024", "FY2025", "FY2026"):
            continue
        if p["metric"] not in ("total income", "revenue from operations",
                               "profit for the year", "total assets",
                               "total equity", "profit before tax"):
            continue
        comp_metric[(p["company"], p["metric"], p["basis"], p["variant"])][p["fiscal_year"]] = p
    scp = 0
    for (comp, metric, basis, variant), by_fy in sorted(
            comp_metric.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if scp >= 15:
            break
        ys = sorted(by_fy)
        if len(ys) < 2:
            continue
        y1, y2 = ys[0], ys[-1]
        if not register(metric, y1, comp, variant):
            continue
        if not register(metric, y2, comp, variant):
            continue
        if not register_cm(comp, metric):
            continue
        a, b = by_fy[y1], by_fy[y2]
        if a["doc_id"] == b["doc_id"]:
            continue
        q = question("same_company_cross_period", comp=_comp(comp),
                     metric=metric, y1=y1, y2=y2,
                     basis=basis, unit=a["units"],
                     _avoid=accepted_questions)
        if q is None:
            used_keys.discard(key(metric, y1, comp, variant))
            used_keys.discard(key(metric, y2, comp, variant))
            used_cm.discard((comp, metric))
            continue
        drafts.append({
            "question": q, "category": "same_company_cross_period",
            "proposed_answer": f"{y1}={a['value']}; {y2}={b['value']}",
            "doc_id": f"{a['doc_id']}|{b['doc_id']}",
            "page": f"{a['page']}|{b['page']}",
            "evidence_excerpt": a["excerpt"] + " ||| " + b["excerpt"],
            "units": a["units"], "version": a.get("version"),
        })
        accepted_questions.append(q)
        scp += 1

    # ---- multi_hop (15), two DIFFERENT metrics per row ----
    page_metrics = defaultdict(list)
    for p in pool:
        if p["metric"] in ("profit for the year", "profit for the period"):
            continue
        pair = (p["doc_id"], p["page"])
        if not any(x["metric"] == p["metric"] for x in page_metrics[pair]):
            page_metrics[pair].append(p)
    mh = 0
    for (doc_id, page), mps in sorted(page_metrics.items()):
        if mh >= 15:
            break
        if len(mps) < 2:
            continue
        a, b = mps[0], mps[1]
        fy = a["fiscal_year"]
        if not register(a["metric"], fy, a["company"], a.get("variant")):
            continue
        if not register(b["metric"], fy, b["company"], b.get("variant")):
            continue
        if not register_cm(a["company"], a["metric"]):
            continue
        if not register_cm(a["company"], b["metric"]):
            continue
        q = question("multi_hop", comp=_comp(a["company"]),
                     metric=f"{a['metric']} and {b['metric']}",
                     metric_a=a["metric"], metric_b=b["metric"],
                     period=_period(a), basis=a["basis"],
                     unit=a["units"], fy=fy, _avoid=accepted_questions)
        if q is None:
            used_keys.discard(key(a["metric"], fy, a["company"], a.get("variant")))
            used_keys.discard(key(b["metric"], fy, b["company"], b.get("variant")))
            used_cm.discard((a["company"], a["metric"]))
            used_cm.discard((a["company"], b["metric"]))
            continue
        drafts.append({
            "question": q, "category": "multi_hop",
            "proposed_answer": f"{a['metric']}={a['value']}; {b['metric']}={b['value']}",
            "doc_id": doc_id, "page": str(page),
            "evidence_excerpt": a["excerpt"] + " ||| " + b["excerpt"],
            "units": a["units"], "version": a.get("version"),
        })
        accepted_questions.append(q)
        mh += 1

    for d in drafts:
        d["verified_by"] = ""
        d["status"] = "draft"
    columns = COLUMNS + ["version"]
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(drafts)
    print(f"gold_candidates: wrote {len(drafts)} draft rows to {out_path}")
    return 0


def verify(path: str, ledger_path=LEDGER) -> int:
    """Check verified rows: page exists, excerpt on that page, numeric answer
    on that page. Multi-document rows (doc_id/page joined with '|') are
    checked per document-page pair; excerpts split on ' ||| ' are checked
    against the matching page. ABSTAIN rows skip the numeric check."""
    with Path(path).open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    checked = failed = 0
    for i, row in enumerate(rows, start=2):
        if not (row.get("verified_by") or "").strip():
            continue  # draft, human verification pending
        checked += 1
        problems: list[str] = []
        doc_ids = row["doc_id"].split("|")
        page_s = row["page"].split("|")
        if len(doc_ids) != len(page_s):
            problems.append(f"doc_id/page arity mismatch "
                            f"({len(doc_ids)} vs {len(page_s)})")
        else:
            pairs = list(zip(doc_ids, page_s))
            try:
                pairs = [(d, int(p)) for d, p in pairs]
            except ValueError:
                problems.append("non-numeric page in row")
                pairs = []
            excerpt_parts = [p.strip() for p in
                             (row.get("evidence_excerpt") or "").split("|||")]
            if len(excerpt_parts) != len(pairs):
                excerpt_parts = [None] * len(pairs)  # arity flagged above
            num_tokens = _NUM_TOKEN_RE.findall(row.get("proposed_answer") or "")
            is_abstain = "abstain" in (row.get("proposed_answer") or "").lower()
            for (doc_id, page_no), excerpt in zip(pairs, excerpt_parts):
                pages = _pages(doc_id)
                if page_no not in pages:
                    problems.append(f"page {page_no} not in page extract of "
                                    f"{doc_id[:12]}")
                    continue
                text = pages[page_no]
                if excerpt:
                    if normalise_for_hash(excerpt) not in \
                            normalise_for_hash(text):
                        problems.append(
                            f"excerpt not found verbatim on page {page_no}")
                if not is_abstain and num_tokens:
                    if not any(_page_has_number(text, t) for t in num_tokens):
                        problems.append(
                            f"no proposed-answer number on page {page_no}")
        if problems:
            failed += 1
            print(f"FAIL row {i} ({row['question'][:60]!r}): "
                  + "; ".join(problems))
    print(f"verify: checked={checked} failed={failed} "
          f"drafts={len(rows) - checked}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.eval.gold_candidates")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", metavar="CSV")
    parser.add_argument("--output", default="data/gold/candidates.csv")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args(argv)
    if args.verify:
        return verify(args.verify)
    return generate(args.output, args.limit)


if __name__ == "__main__":
    sys.exit(main())
