"""Check quarterly statement-map readiness for 2 companies (read-only; no fixes)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.tables.statement_map import load_pages, propose_spans

PAGES_DIR = Path("data/corpus/extract/page_extracts")
COMPANIES = {
    "deepak_nitrite": [
        "DEEPAKNTR_quarterly_Q1FY2026_consolidated_U_printed.pdf",
        "DEEPAKNTR_quarterly_Q1FY2026_standalone_U_printed.pdf",
    ],
    "adani_green": [
        "ADANIGREEN_quarterly_Q1FY2026_consolidated_U_printed.pdf",
        "ADANIGREEN_quarterly_Q1FY2026_standalone_U_printed.pdf",
    ],
}

FIELD_LABELS = {
    "revenue": re.compile(r"revenue from operations", re.I),
    "pat": re.compile(r"total profit\s*\(loss\)\s*for period", re.I),
    "eps": re.compile(r"earnings per share|basic & diluted|basic and diluted", re.I),
}


def checksum_for(company: str, filename: str) -> str:
    variant = "consolidated" if "consolidated" in filename else "standalone"
    wanted_period = "Q1FY2026"
    best = ""
    for line in open("data/corpus/ledger.jsonl", encoding="utf-8"):
        record = json.loads(line)
        if record.get("kind") != "document":
            continue
        data = record.get("data") or {}
        if (data.get("company_id") == company
                and data.get("doc_type") == "quarterly"
                and data.get("fiscal_year") == wanted_period
                and data.get("variant") == variant):
            best = data["checksum"]
    return best


def numeric_tokens(text: str) -> int:
    return len(re.findall(r"\d[\d,\.]*", text))


def check() -> list[dict]:
    results = []
    for company, files in COMPANIES.items():
        for filename in files:
            checksum = checksum_for(company, filename)
            if not checksum:
                results.append({"file": f"{company}/{filename}", "error": "not in ledger"})
                continue
            pages = load_pages(PAGES_DIR.parent, checksum)
            spans = propose_spans(company, checksum, pages)
            page_map = dict(pages)
            row = {"file": f"{company}/{filename}", "checksum": checksum[:12],
                   "spans": [(s.basis, s.statement, s.start_pdf_page, s.end_pdf_page, s.status)
                             for s in spans],
                   "fields": {}}
            for field, pattern in FIELD_LABELS.items():
                hits = []
                for page_no in sorted(page_map):
                    text = page_map[page_no]
                    if pattern.search(text) and numeric_tokens(text) >= 5:
                        hits.append(f"p{page_no}")
                row["fields"][field] = "PASS " + ",".join(hits[:6]) if hits else "FAIL"
            results.append(row)
    return results


def main() -> int:
    for row in check():
        print(json.dumps(row, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
