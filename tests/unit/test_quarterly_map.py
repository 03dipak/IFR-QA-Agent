"""Unit tests for the NSE iXBRL quarterly statement layout."""

from __future__ import annotations

import json
from pathlib import Path

from src.tables.quarterly_map import quarterly_figures, quarterly_spans
from src.tables.statement_map import load_pages, propose_spans

PAGES_DIR = Path("data/corpus/extract")
CASES = [
    ("deepak_nitrite", "consolidated", "Q1FY2026",
     {"revenue": "18898800000", "pat": "1122500000", "eps": "8.23"}),
    ("adani_green", "standalone", "Q1FY2026",
     {"revenue": "47890000000", "pat": "340000000", "eps": "0.11"}),
]


def _checksum(company: str, variant: str, fiscal_year: str) -> str:
    """Last ledger row for the company/variant/period."""
    checksum = ""
    with open("data/corpus/ledger.jsonl", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("kind") != "document":
                continue
            data = record["data"]
            if (data.get("company_id") == company
                    and data.get("doc_type") == "quarterly"
                    and data.get("fiscal_year") == fiscal_year
                    and data.get("variant") == variant):
                checksum = data["checksum"]
    assert checksum, f"no ledger row for {company}/{variant}/{fiscal_year}"
    return checksum


def test_quarterly_spans_and_figures() -> None:
    for company, variant, period, expected in CASES:
        checksum = _checksum(company, variant, period)
        pages = load_pages(PAGES_DIR, checksum)
        assert pages, f"no page extracts for {checksum}"

        spans = quarterly_spans(company, checksum, pages)
        pl = [span for span in spans if span.basis == variant and span.statement == "PL"]
        assert len(pl) == 1, f"{company}: expected one PL span, got {spans}"
        assert pl[0].start_pdf_page >= 1

        figures = quarterly_figures(pages, variant)
        for field, value in expected.items():
            assert field in figures, f"{company}: missing {field}"
            assert figures[field][0] == value, f"{company}: {field}={figures[field][0]}"


def test_propose_spans_falls_back_to_quarterly() -> None:
    company, variant, period = CASES[0][0], CASES[0][1], CASES[0][2]
    checksum = _checksum(company, variant, period)
    pages = load_pages(PAGES_DIR, checksum)
    spans = propose_spans(company, checksum, pages)
    assert any(span.statement == "PL" and span.basis == variant for span in spans), spans
