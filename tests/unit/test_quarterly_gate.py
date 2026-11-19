"""Quarterly results gate: statement-like detection for native results PDFs."""

from __future__ import annotations

from src.ingestion.extractor import (
    extract_pages,
    is_scanned_document,
    is_statement_like,
    scanned_document_rules,
)
from src.ingestion.models import DocType
from tests.unit.test_ingestion import _make_text_pdf

_HEADER = "LARSEN AND TOUBRO LIMITED  Mumbai 400 001  CIN: U00000MH0000PLC000000"
_RESULTS_TITLE_FULL = ("STATEMENT OF UNAUDITED CONSOLIDATED FINANCIAL RESULTS "
                       "FOR THE QUARTER ENDED JUNE 30, 2026")
_RESULTS_TITLE_STANDALONE = ("STATEMENT OF UNAUDITED STANDALONE FINANCIAL "
                             "RESULTS FOR THE QUARTER ENDED JUNE 30, 2026")


def _results_pages() -> list[list[str]]:
    return [
        [_HEADER, _RESULTS_TITLE_FULL,
         "1 Income: a) Revenue from operations 67941.74 82762.16 63678.92",
         "b) Other income (net) 2376.73 1578.59 1356.78",
         "Total Income 70318.47 84340.75 65035.70",
         "2 Expenses: ii) Construction materials consumed 11729.13 18352.95 13951.77",
         "Profit before exceptional items and tax 5180.05 5297.99 4040.09"],
        ["June 30, March 31, June 30, March 31",
         "1 Debt equity ratio 1.01 0.95 1.13 0.95",
         "2 Debt service coverage ratio 10.72 9.00 0.92 2.27",
         "3 Interest service coverage ratio 11.42 11.55 7.15 9.19",
         "4 Current ratio 1.25 1.25 1.21 1.25",
         "5 Long term debt to working capital ratio 1.07 0.95 1.13 0.95"],
        ["June 30, March 31, June 30, March 31",
         "a) Revenue from operations 36024.07 47190.86 33470.73 153680.17",
         "b) Profit before tax (after exceptional items) 5180.05 4697.99 4040.09",
         "d) Net profit after tax (after exceptional items) 4455.26 3560.92 3485.30",
         "Mumbai July 28, 2026",
         "The specified items of the standalone financial results of the Company "
         "for the quarter ended June 30, 2026 are given below"],
        ["Gross segment revenue June 30 2026",
         "1 Infrastructure and Utilities 22143.53 29221.35 22696.07",
         "2 Energy 14246.31 16113.52 12467.57",
         "3 Energy - Green 5607.09 11426.16 6335.48",
         "Total 68613.28 83753.18 64260.36",
         "Net segment revenue 67941.74 82762.16 63678.92"],
        [_HEADER, _RESULTS_TITLE_STANDALONE,
         "1 Income: a) Revenue from operations 36024.07 47190.86 33470.73",
         "b) Other income (net) 4094.41 1502.84 2606.00",
         "Total Income 40118.48 48693.70 36076.73",
         "Total Expenses 34938.43 43395.71 32036.64",
         "Net profit after tax 4455.26 3560.92 3485.30"],
        ["June 30 2026 Financial Statements",
         "1 Debt equity ratio 0.23 0.16 0.31 0.16",
         "2 Debt service coverage ratio 15.70 15.15 0.78 1.81",
         "3 Interest service coverage ratio 15.70 15.15 9.52 10.71",
         "4 Current ratio 1.27 1.29 1.26 1.29",
         "15 The Company has not issued any secured listed non-convertible debt securities."],
        ["June 30 2026 Financial Statements",
         "1 Infrastructure and Utilities 19122.59 25559.92 19763.06",
         "2 Energy - Conventional 8566.31 8403.55 6333.32",
         "3 Energy - Green 3910.50 7563.82 3290.41",
         "4 Manufacturing and Products 4079.27 5623.19 3752.27",
         "Total 36217.90 47509.34 33670.26"],
    ]


def test_quarterly_results_pdf_keeps_and_detects_statement():
    pages = extract_pages(_make_text_pdf(_results_pages()), "d1", "run1")
    assert scanned_document_rules(pages, DocType.QUARTERLY) == []
    assert is_scanned_document(pages, DocType.QUARTERLY)[0] is False
    assert any(is_statement_like(p.page_text) for p in pages)


def test_quarterly_results_titles_are_statement_like():
    pages = extract_pages(_make_text_pdf(_results_pages()), "d2", "run2")
    titles = [p.page_text for p in pages
              if "FINANCIAL RESULTS FOR THE QUARTER ENDED" in p.page_text.upper()]
    assert len(titles) == 2
    assert all(is_statement_like(t) for t in titles)
