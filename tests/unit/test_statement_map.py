"""Focused tests for statement-span proposal."""

from __future__ import annotations

from src.tables.statement_map import propose_spans


def _page(label: str, values: str = "1 2 3 4 5 6") -> str:
    return f"Consolidated Financial Statements {label} As at March 31 {values}"


def test_chooses_smallest_heading_triple():
    pages = {
        10: _page("Balance Sheet"),
        11: _page("Statement of Profit and Loss"),
        13: _page("Statement of Cash Flows"),
        30: _page("Balance Sheet"),
    }
    spans = propose_spans("c", "d.pdf", list(pages.items()))
    by_key = {(s.basis, s.statement): s for s in spans}
    assert (by_key[("consolidated", "BS")].start_pdf_page,
            by_key[("consolidated", "PL")].start_pdf_page,
            by_key[("consolidated", "CF")].start_pdf_page) == (10, 11, 13)
    assert all(s.status == "proposed" for s in spans)


def test_footer_heading_detected_when_not_in_opening_text():
    body = "Notes Year ended March 31 2025 INCOME 1 2 3 4 5 6"
    pages = {
        7: _page("Balance Sheet"),
        8: body + " Consolidated Statement of Profit and Loss",
        10: body + " Consolidated Statement of Cash Flows",
    }
    spans = propose_spans("c", "d.pdf", list(pages.items()))
    by_key = {(s.basis, s.statement): s for s in spans}
    assert by_key[("consolidated", "PL")].start_pdf_page == 8
    assert by_key[("consolidated", "CF")].start_pdf_page == 10


def test_wide_spread_is_flagged_check():
    pages = {
        1: _page("Balance Sheet"),
        5: _page("Statement of Profit and Loss"),
        11: _page("Statement of Cash Flows"),
    }
    spans = propose_spans("c", "d.pdf", list(pages.items()))
    assert all(s.status == "check" for s in spans)
