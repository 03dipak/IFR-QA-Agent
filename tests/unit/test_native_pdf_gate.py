"""Tests for the revised native-PDF document gate (LLD-45)."""

from __future__ import annotations

from src.ingestion import pipeline as pipeline_module
from src.ingestion.extractor import (
    PageClass,
    classify_page,
    is_scanned_document,
    is_statement_like,
)
from src.ingestion.models import PageExtract, PageType


def _mk_page(idx, page_class, text="", char_count=None, usable=True, uffd=0):
    if char_count is None:
        char_count = len(text)
    return PageExtract(
        document_id="d",
        extraction_run_id="r",
        pdf_page_index=idx,
        page_text=text,
        extraction_hash=f"h{idx}",
        page_type=PageType.SCAN if page_class != PageClass.TEXT else PageType.STATEMENT,
        page_class=page_class,
        usable_as_evidence=usable,
        ufffd_count=uffd,
        character_count=char_count,
    )


def _patch_extract(monkeypatch, pages):
    def fake_extract(_data, document_id, run_id):
        out = []
        for p in pages:
            p.document_id = document_id
            p.extraction_run_id = run_id
            out.append(p)
        return out
    monkeypatch.setattr(pipeline_module, "extract_pages", fake_extract)


def test_empty_page_is_blank():
    assert classify_page("", 0, 0, 0, 612.0, 792.0) == PageClass.BLANK


def test_low_text_boundary_39_chars():
    text = "a" * 39
    assert classify_page(text, 0, 0, 0, 612.0, 792.0) == PageClass.LOW_TEXT


def test_image_dominant_with_live_header_not_excluded(monkeypatch, tmp_path):
    from src.ingestion.pipeline import process_candidate
    from tests.unit.test_ingestion import FakeFetcher

    stmt = "Balance Sheet total assets 100 revenue 200 liabilities 50 cash flow 10 profit 5"
    pages = [
        _mk_page(1, PageClass.TEXT, stmt),
        _mk_page(2, PageClass.TEXT, stmt),
        _mk_page(3, PageClass.TEXT, stmt),
        _mk_page(4, PageClass.IMAGE_DOMINANT, "Assurance Statement", char_count=23,
                 usable=False),
    ]
    _patch_extract(monkeypatch, pages)
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


def test_mixed_image_pages_kept_with_flagged_pages(monkeypatch, tmp_path):
    from src.ingestion.pipeline import process_candidate
    from tests.unit.test_ingestion import FakeFetcher

    stmt = "Balance Sheet total assets 100 revenue 200 liabilities 50 cash flow 10 profit 5"
    pages = [
        _mk_page(1, PageClass.TEXT, stmt),
        _mk_page(2, PageClass.TEXT, stmt),
        _mk_page(3, PageClass.TEXT, stmt),
        _mk_page(4, PageClass.IMAGE_ONLY, "", char_count=0, usable=False),
        _mk_page(5, PageClass.IMAGE_ONLY, "", char_count=0, usable=False),
    ]
    _patch_extract(monkeypatch, pages)
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
    assert len(ledger.rows) == 1
    assert not is_scanned_document(pages)[0]
    assert pages[3].usable_as_evidence is False


def test_image_only_document_excluded(monkeypatch, tmp_path):
    from src.ingestion.pipeline import process_candidate
    from tests.unit.test_ingestion import FakeFetcher

    pages = [_mk_page(i, PageClass.IMAGE_ONLY, "", char_count=0, usable=False)
             for i in range(1, 6)]
    _patch_extract(monkeypatch, pages)
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
    assert state == "excluded"
    assert ledger.rows == {}
    assert ledger.exclusions[0].reason.value == "scanned_pdf"


def test_statement_pages_all_image_only_excluded(monkeypatch, tmp_path):
    from src.ingestion.pipeline import process_candidate
    from tests.unit.test_ingestion import FakeFetcher

    pages = [_mk_page(i, PageClass.IMAGE_ONLY, "", char_count=0, usable=False)
             for i in range(1, 4)]
    _patch_extract(monkeypatch, pages)
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
    assert state == "excluded"


def test_garbled_font_document_excluded(monkeypatch, tmp_path):
    from src.ingestion.pipeline import process_candidate
    from tests.unit.test_ingestion import FakeFetcher

    pages = [_mk_page(1, PageClass.GARBLED, "\ufffd" * 80, char_count=80,
                      usable=False)]
    _patch_extract(monkeypatch, pages)
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
    assert state == "excluded"


def test_native_with_blank_divider_short_chart_not_excluded(monkeypatch, tmp_path):
    from src.ingestion.pipeline import process_candidate
    from tests.unit.test_ingestion import FakeFetcher

    stmt = "Balance Sheet total assets 100 revenue 200 liabilities 50 cash flow 10 profit 5"
    pages = [
        _mk_page(1, PageClass.TEXT, stmt),
        _mk_page(2, PageClass.TEXT, stmt),
        _mk_page(3, PageClass.TEXT, stmt),
        _mk_page(4, PageClass.BLANK, "", char_count=0, usable=False),
        _mk_page(5, PageClass.LOW_TEXT, "Standalone Financial Statements",
                 char_count=31, usable=False),
        _mk_page(6, PageClass.LOW_TEXT, "Revenue chart", char_count=13,
                 usable=False),
    ]
    _patch_extract(monkeypatch, pages)
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


def test_ufffd_count_diagnostic(monkeypatch):
    page = _mk_page(1, PageClass.TEXT, "x", uffd=3)
    assert page.ufffd_count == 3
    assert page.usable_as_evidence is True


def test_is_statement_like():
    assert is_statement_like("Balance Sheet 1 2 3 4 5") is True
    assert is_statement_like("Balance Sheet") is False
