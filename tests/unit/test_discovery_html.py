"""Unit tests for discovery HTML listing mode: link harvest + bounded probe.

No network: listing pages come from saved fixtures; probes monkeypatch
urllib.request.urlopen with in-memory fakes.
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path

from src.ingestion import discovery

FIXTURES = Path(__file__).parent / "fixtures"


def _listing(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int = 206, content_type: str = "application/pdf"):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_extract_pdf_links_order_dedupe_and_case(monkeypatch):
    links = discovery.extract_pdf_links(_listing("listing_wipro.html"))
    assert links == [
        "/content/dam/nexus/en/investor/annual-reports/2022-2023/"
        "integrated-annual-report-2022-23.pdf",
        "/content/dam/nexus/en/investor/annual-reports/2023-2024/"
        "integrated-annual-report-2023-24.pdf",
        "/content/dam/nexus/en/investor/annual-reports/2024-2025/"
        "Integrated-annual-report-2024-2025.PDF",
        "https://www.wipro.com/content/dam/nexus/en/investor/quarterly-results/"
        "2025-2026/q1fy26/regulation-33-financial-results-q1fy26.pdf",
    ]


def test_extract_pdf_links_ignores_non_pdf_links():
    links = discovery.extract_pdf_links(_listing("listing_lt.html"))
    assert len(links) == 3
    assert all(url.lower().endswith(".pdf") for url in links)


def test_candidates_from_listing_html_annotates_rows():
    rows = discovery.candidates_from_listing_html(
        _listing("listing_lt.html"),
        company_id="l_and_t",
        scrip_code="500510",
        document_type="quarterly",
        fiscal_period="Q1FY2027",
        source_page="https://investors.larsentoubro.com/Quarterly-Results-Archives.aspx",
    )
    assert len(rows) == 3
    first = rows[0]
    assert first["company_id"] == "l_and_t"
    assert first["document_type"] == "quarterly"
    assert first["fiscal_period"] == "Q1FY2027"
    assert first["filing_date"] == ""
    assert first["url"].endswith(".pdf")
    assert first["discovered_at"]


def test_probe_pdf_url_reports_pdf_magic(monkeypatch):
    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda req, timeout: FakeResponse(b"%PDF-1.7\n...", status=206),
    )
    result = discovery.probe_pdf_url("https://example.invalid/report.pdf")
    assert result.ok is True
    assert result.status == 206
    assert result.pdf_magic is True


def test_probe_pdf_url_rejects_html_masquerading_as_pdf(monkeypatch):
    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda req, timeout: FakeResponse(b"<!doctype html><html>", status=200,
                                          content_type="text/html"),
    )
    result = discovery.probe_pdf_url("https://example.invalid/fake.pdf")
    assert result.ok is False
    assert result.pdf_magic is False


def test_probe_pdf_url_captures_errors(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.URLError("403 Forbidden")

    monkeypatch.setattr(discovery.urllib.request, "urlopen", boom)
    result = discovery.probe_pdf_url("https://example.invalid/blocked.pdf")
    assert result.ok is False
    assert result.status is None
    assert "403" in (result.error or "")


def test_probe_sends_honest_user_agent_and_range(monkeypatch):
    captured: dict[str, str] = {}

    def fake_urlopen(req, timeout):
        captured.update(req.headers)
        return FakeResponse(b"%PDF-1.6", status=206)

    monkeypatch.setattr(discovery.urllib.request, "urlopen", fake_urlopen)
    discovery.probe_pdf_url("https://example.invalid/report.pdf")
    ua = captured.get("User-agent") or captured.get("User-Agent") or ""
    assert "IFR-QA-Agent" in ua
    assert "Mozilla" not in ua
    rng = captured.get("Range") or ""
    assert rng.startswith("bytes=0-")
