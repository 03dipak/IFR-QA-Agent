"""Document-type scoping for the native-PDF gate (LLD-45 regression)."""

from __future__ import annotations

from src.ingestion import pipeline as pipeline_module
from src.ingestion.ledger import Ledger
from src.ingestion.models import Candidate, DocType, PageExtract, PageType
from src.ingestion.pipeline import process_candidate
from src.ingestion.storage import LocalRawStorage
from tests.unit.test_ingestion import FakeFetcher


def _page(idx, text, page_type=PageType.PROSE):
    return PageExtract(
        document_id="d",
        extraction_run_id="r",
        pdf_page_index=idx,
        page_text=text,
        extraction_hash=f"h{idx}",
        page_type=page_type,
        character_count=len(text),
    )


def test_disclosure_with_no_statements_not_excluded(monkeypatch, tmp_path):
    def fake_extract(_data, document_id, run_id):
        return [_page(1, "Disclosure of voting results of the general meeting.")]

    monkeypatch.setattr(pipeline_module, "extract_pages", fake_extract)
    pdf = b"%PDF-1.4\n"
    ledger = Ledger()
    candidate = Candidate(
        candidate_id="disc", company_id="C", fiscal_year="FY",
        reporting_period="disclosure", doc_type=DocType.DISCLOSURE,
        source_url="https://x/d.pdf",
    )
    state = process_candidate(candidate, FakeFetcher({"https://x/d.pdf": pdf}),
                             ledger, LocalRawStorage(tmp_path))
    assert state == "extracted"


def test_quarterly_with_one_statement_page_not_excluded(monkeypatch, tmp_path):
    stmt = "Statement of Profit and Loss revenue 100 expenses 50 profit 25 tax 5 net 20"
    def fake_extract(_data, document_id, run_id):
        return [_page(1, stmt, PageType.STATEMENT)]

    monkeypatch.setattr(pipeline_module, "extract_pages", fake_extract)
    pdf = b"%PDF-1.4\n"
    ledger = Ledger()
    candidate = Candidate(
        candidate_id="q", company_id="C", fiscal_year="FY",
        reporting_period="quarterly", doc_type=DocType.QUARTERLY,
        source_url="https://x/q.pdf",
    )
    state = process_candidate(candidate, FakeFetcher({"https://x/q.pdf": pdf}),
                             ledger, LocalRawStorage(tmp_path))
    assert state == "extracted"
