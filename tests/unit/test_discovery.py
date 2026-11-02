"""Unit tests for LLD-35 Phase 1 discovery."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.ingestion.discovery import (
    CANDIDATES_COLUMNS,
    Company,
    DiscoveryConfig,
    build_candidates,
    load_companies,
    load_discovery_config,
    main,
    validate_companies,
    write_candidates,
)
from src.ingestion.fetcher import Fetcher
from src.ingestion.models import FetchAttempt


class FakeFetcher(Fetcher):
    def __init__(self, pages: dict[str, bytes]) -> None:
        super().__init__(min_interval_seconds=0.0)
        self.pages = pages

    def fetch(self, url: str, candidate_id: str) -> tuple[bytes, FetchAttempt]:
        data = self.pages[url]
        attempt = FetchAttempt(
            fetch_attempt_id="a1",
            candidate_id=candidate_id,
            source_url=url,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            http_status=200,
            content_length=len(data),
            downloaded_bytes=len(data),
            attempt_number=1,
        )
        return data, attempt


def test_load_companies_reads_csv(tmp_path) -> None:
    p = tmp_path / "companies.csv"
    p.write_text(
        "company_name,scrip_code,isin\nReliance Industries,500325,IN0000010264\n",
        encoding="utf-8",
    )
    companies = load_companies(p)
    assert len(companies) == 1
    assert companies[0].scrip_code == "500325"
    assert companies[0].isin == "IN0000010264"


def test_load_discovery_config_reads_json(tmp_path) -> None:
    p = tmp_path / "discovery.json"
    p.write_text(
        json.dumps({"fiscal_years": ["FY2024"], "doc_types": ["annual"], "listing_urls": {}}),
        encoding="utf-8",
    )
    cfg = load_discovery_config(p)
    assert cfg.fiscal_years == ["FY2024"]
    assert cfg.doc_types == ["annual"]
    assert cfg.listing_urls == {}


def test_build_candidates_dedupes_by_url() -> None:
    cfg = DiscoveryConfig(listing_urls={"500325": "http://fake/list"})
    fetcher = FakeFetcher(
        {"http://fake/list": b"<a href='http://x/a.pdf'>a</a><a href='http://x/a.pdf'>a</a>"}
    )
    companies = [Company(company_id="RELIANCE", company_name="Reliance", scrip_code="500325")]
    candidates, duplicates, _unmatched = build_candidates(companies, cfg, fetcher)
    assert len(candidates) == 1
    assert len(duplicates) == 1
    assert candidates[0]["url"] == "http://x/a.pdf"


def test_build_candidates_unmatched_when_no_listing() -> None:
    cfg = DiscoveryConfig(listing_urls={})
    fetcher = FakeFetcher({})
    companies = [Company(company_id="RELIANCE", company_name="Reliance", scrip_code="500325")]
    candidates, _duplicates, unmatched = build_candidates(companies, cfg, fetcher)
    assert candidates == []
    assert len(unmatched) == 1


def test_write_candidates_writes_csv(tmp_path) -> None:
    p = tmp_path / "candidates.csv"
    write_candidates(
        [
            {
                "company_id": "C1",
                "scrip_code": "1",
                "document_type": "annual",
                "fiscal_period": "FY2024",
                "filing_date": "2024-01-01",
                "url": "http://x/1.pdf",
                "source_page": "http://list",
                "discovered_at": "now",
            }
        ],
        p,
    )
    text = p.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(CANDIDATES_COLUMNS)
    assert "http://x/1.pdf" in text


VALID_HEADER = (
    "company_name,scrip_code,isin,company_id,verification_status,"
    "source_url,checked_on,notes"
)
VALID_ROW = "Reliance,500325,IN0000010264,RELIANCE,VERIFIED,https://x/1,2026-11-02,ok"
VALID_ROW2 = "Reliance2,500325,IN0000010265,RELIANCE,VERIFIED,https://x/2,2026-11-02,ok2"
INVALID_ROW = "invalid co,500325,in12345678abc,CO,NEEDS_VERIFICATION,https://x/1,2026-11-02,ok"
EMPTY_VERIFIED_ROW = "Reliance,,,RELIANCE,VERIFIED,https://x/1,2026-11-02,ok"


def _write_companies(path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_validate_companies_ok(tmp_path) -> None:
    p = tmp_path / "companies.csv"
    p.write_text(VALID_HEADER + "\n" + VALID_ROW + "\n", encoding="utf-8")
    assert validate_companies(p) == []


def test_validate_companies_reports_missing_columns(tmp_path) -> None:
    p = tmp_path / "companies.csv"
    text = "company_name,scrip_code,isin\nReliance,500325,IN0000010264\n"
    p.write_text(text, encoding="utf-8")
    problems = validate_companies(p)
    assert len(problems) == 1
    assert "missing required columns" in problems[0]


def test_validate_companies_reports_duplicates(tmp_path) -> None:
    p = tmp_path / "companies.csv"
    p.write_text(
        VALID_HEADER + "\n" + VALID_ROW + "\n" + VALID_ROW2 + "\n",
        encoding="utf-8",
    )
    problems = validate_companies(p)
    assert any("duplicate scrip_code" in problem for problem in problems)


def test_validate_companies_reports_invalid_isin(tmp_path) -> None:
    p = tmp_path / "companies.csv"
    p.write_text(VALID_HEADER + "\n" + INVALID_ROW + "\n", encoding="utf-8")
    problems = validate_companies(p)
    assert any("invalid ISIN format" in problem for problem in problems)


def test_validate_companies_reports_empty_verified_row(tmp_path) -> None:
    p = tmp_path / "companies.csv"
    p.write_text(VALID_HEADER + "\n" + EMPTY_VERIFIED_ROW + "\n", encoding="utf-8")
    problems = validate_companies(p)
    assert any("VERIFIED row has empty" in problem for problem in problems)


def test_main_validate_returns_zero_when_valid(tmp_path) -> None:
    text = VALID_HEADER + "\n" + VALID_ROW + "\n"
    p = _write_companies(tmp_path / "companies.csv", text)
    cfg = tmp_path / "discovery.json"
    cfg.write_text(
        json.dumps({"fiscal_years": ["FY2024"], "doc_types": ["annual"], "listing_urls": {}}),
        encoding="utf-8",
    )
    assert main(["--companies", p, "--config", str(cfg), "--validate"]) == 0


def test_main_validate_returns_one_when_problems(tmp_path) -> None:
    p = _write_companies(
        tmp_path / "companies.csv",
        "company_name,scrip_code,isin\nReliance,500325,IN0000010264\n",
    )
    cfg = tmp_path / "discovery.json"
    cfg.write_text(
        json.dumps({"fiscal_years": ["FY2024"], "doc_types": ["annual"], "listing_urls": {}}),
        encoding="utf-8",
    )
    assert main(["--companies", p, "--config", str(cfg), "--validate"]) == 1


def test_main_discover_refuses_needs_verification(tmp_path) -> None:
    p = _write_companies(
        tmp_path / "companies.csv",
        VALID_HEADER + "\nReliance,,,RELIANCE,NEEDS_VERIFICATION,https://x/1,2026-11-02,draft ok\n",
    )
    cfg = tmp_path / "discovery.json"
    cfg.write_text(
        json.dumps({"fiscal_years": ["FY2024"], "doc_types": ["annual"], "listing_urls": {}}),
        encoding="utf-8",
    )
    assert main(["--companies", p, "--config", str(cfg)]) == 2
