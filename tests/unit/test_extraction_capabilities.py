"""Focused tests for table, image and page-extract persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.ingestion.extractor import extract_images
from src.ingestion.models import Candidate, DocType
from src.ingestion.pipeline import process_candidate
from src.ingestion.storage import (
    LocalPageExtractStore,
    LocalRawStorage,
    LocalTableRecordStore,
)
from src.tables.extractor import extract_tables
from tests.unit.test_ingestion import FakeFetcher, _make_text_pdf


def test_page_extracts_persisted():
    tmp = Path(tempfile.mkdtemp())
    stmt = "Balance Sheet total assets 100 revenue 200 liabilities 50 cash flow 10 profit 5"
    pdf = _make_text_pdf([[stmt], [stmt], [stmt]])
    from src.ingestion.ledger import Ledger

    ledger = Ledger()
    candidate = Candidate(
        candidate_id="c", company_id="C", fiscal_year="FY",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://x/y.pdf",
    )
    state = process_candidate(
        candidate, FakeFetcher({"https://x/y.pdf": pdf}), ledger, LocalRawStorage(tmp)
    )
    assert state == "extracted"
    doc_id = next(iter(ledger.rows))
    pages = LocalPageExtractStore(tmp.parent / "extract" / "page_extracts").get(doc_id)
    assert len(pages) == 3


def test_table_records_persisted():
    tmp = Path(tempfile.mkdtemp())
    page = ["Balance Sheet", "Total assets 100 200", "Revenue 10 20", "Liabilities 5 6"]
    pdf = _make_text_pdf([page, page, page])
    from src.ingestion.ledger import Ledger

    ledger = Ledger()
    candidate = Candidate(
        candidate_id="c", company_id="C", fiscal_year="FY",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://x/y.pdf",
    )
    state = process_candidate(
        candidate, FakeFetcher({"https://x/y.pdf": pdf}), ledger, LocalRawStorage(tmp)
    )
    assert state == "extracted"
    doc_id = next(iter(ledger.rows))
    records = LocalTableRecordStore(tmp.parent / "extract" / "table_records").get(doc_id)
    assert len(records) == 3
    assert records[0].statement_kind.value == "balance_sheet"


def test_extract_tables_direct():
    from src.ingestion.models import PageExtract

    page = PageExtract(
        document_id="d", extraction_run_id="r", pdf_page_index=1,
        page_text="Balance Sheet\nTotal assets 100 200\nRevenue 10 20\nLiabilities 5 6",
        extraction_hash="h", character_count=40,
    )
    records = extract_tables([page], "d", "r", "cfg")
    assert len(records) == 1
    assert len(records[0].rows) == 2


def test_extract_images_skips_unsupported_image(monkeypatch):
    class _Page:
        images = None

    class _Images:
        def __getitem__(self, index):
            if index == 0:
                from pypdf.errors import DependencyError

                raise DependencyError("jbig2dec binary is not available.")
            if index == 1:
                return SimpleNamespace(name="ok.png", data=b"png")
            raise IndexError(index)

    page = _Page()
    page.images = _Images()

    class _Reader:
        pages = [page]

    monkeypatch.setattr("pypdf.PdfReader", lambda *_a, **_k: _Reader())
    images = extract_images(b"%PDF-1.4 fake")
    assert len(images) == 1
    assert images[0]["name"] == "ok.png"


def test_extract_images_no_ocr():
    data = open(
        "data/corpus/raw/b57b3e08787d5c0cadbe031288cfb2e67f59466cb8e4c06c776180b0c6fb36d4.pdf",
        "rb",
    ).read()
    images = extract_images(data)
    assert len(images) > 0
    assert all("data" in image and len(image["data"]) > 0 for image in images)
