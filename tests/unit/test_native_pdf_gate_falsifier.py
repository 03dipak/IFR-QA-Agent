"""Falsifier for LLD-45: a native report with a blank, a divider and a short
chart page must not be excluded as scanned_pdf."""

from __future__ import annotations

from src.ingestion import pipeline as pipeline_module
from src.ingestion.models import PageExtract, PageType
from src.ingestion.pipeline import process_candidate
from tests.unit.test_ingestion import FakeFetcher


def test_native_with_blank_divider_short_chart_not_excluded(monkeypatch, tmp_path):
    stmt = "Balance Sheet total assets 100 revenue 200 liabilities 50 cash flow 10 profit 5"

    def fake_extract(_data, document_id, run_id):
        return [
            PageExtract(document_id=document_id, extraction_run_id=run_id,
                        pdf_page_index=1, page_text=stmt,
                        extraction_hash="h1", page_type=PageType.STATEMENT,
                        character_count=len(stmt)),
            PageExtract(document_id=document_id, extraction_run_id=run_id,
                        pdf_page_index=2, page_text="",
                        extraction_hash="h2", page_type=PageType.SCAN,
                        character_count=0),
            PageExtract(document_id=document_id, extraction_run_id=run_id,
                        pdf_page_index=3, page_text="Standalone Financial Statements",
                        extraction_hash="h3", page_type=PageType.SCAN,
                        character_count=31),
            PageExtract(document_id=document_id, extraction_run_id=run_id,
                        pdf_page_index=4, page_text="Revenue chart",
                        extraction_hash="h4", page_type=PageType.SCAN,
                        character_count=13),
            PageExtract(document_id=document_id, extraction_run_id=run_id,
                        pdf_page_index=5, page_text=stmt,
                        extraction_hash="h5", page_type=PageType.STATEMENT,
                        character_count=len(stmt)),
            PageExtract(document_id=document_id, extraction_run_id=run_id,
                        pdf_page_index=6, page_text=stmt,
                        extraction_hash="h6", page_type=PageType.STATEMENT,
                        character_count=len(stmt)),
        ]

    monkeypatch.setattr(pipeline_module, "extract_pages", fake_extract)
    pdf = b"%PDF-1.4\n"
    from src.ingestion.ledger import Ledger
    from src.ingestion.models import Candidate, DocType
    from src.ingestion.storage import LocalRawStorage
    ledger = Ledger()
    candidate = Candidate(candidate_id="c", company_id="C", fiscal_year="FY",
                          reporting_period="annual", doc_type=DocType.ANNUAL,
                          source_url="https://x/y.pdf")
    state = process_candidate(candidate, FakeFetcher({"https://x/y.pdf": pdf}),
                             ledger, LocalRawStorage(tmp_path))
    assert state == "extracted"
