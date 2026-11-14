"""Bulk acquisition discovery: companies + config -> filing candidates.

Phase 1 of the bulk-acquisition plan (task LLD-35): read the owner-supplied
companies file and discovery config, list filing candidates per company from
the configured BSE-hosted listing pages, and emit candidates.csv.

HTML listing mode (2026-11-04): company IR/archive pages are ordinary HTML,
which the strict PDF Fetcher rejects, so listing-page link harvest and
bounded PDF URL probing live here and are usable independently of
`Fetcher.fetch`.
"""

from __future__ import annotations

import csv
import datetime
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .fetcher import Fetcher

# Honest client identity for listing/probe requests (no browser spoofing).
DEFAULT_USER_AGENT = "IFR-QA-Agent/0.1 (academic QA research; contact: repo owner)"
PROBE_MAX_BYTES = 512

CANDIDATES_COLUMNS = [
    "company_id",
    "scrip_code",
    "document_type",
    "fiscal_period",
    "filing_date",
    "url",
    "source_page",
    "discovered_at",
]

_DUPLICATES_COLUMNS = CANDIDATES_COLUMNS + ["superseded_by"]

_PDF_LINK_RE = re.compile(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', re.IGNORECASE)

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[A-Z0-9]$")


@dataclass
class DiscoveryConfig:
    fiscal_years: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    listing_urls: dict[str, str] = field(default_factory=dict)


@dataclass
class Company:
    company_id: str
    company_name: str
    scrip_code: str
    isin: str = ""


@dataclass
class Filing:
    url: str
    filing_date: str | None = None
    doc_type: str | None = None
    fiscal_period: str | None = None
    source_page: str | None = None


@dataclass
class ProbeResult:
    url: str
    status: int | None = None
    content_type: str = ""
    pdf_magic: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (200, 206) and self.pdf_magic


def extract_pdf_links(html_text: str) -> list[str]:
    """Harvest ``href=...pdf`` links from a listing page, order-preserving,
    de-duplicated. Mirrors the historical discovery semantics."""
    seen: set[str] = set()
    links: list[str] = []
    for match in _PDF_LINK_RE.finditer(html_text):
        url = match.group(1).strip()
        if url and url not in seen:
            seen.add(url)
            links.append(url)
    return links


def probe_pdf_url(
    url: str,
    *,
    max_bytes: int = PROBE_MAX_BYTES,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> ProbeResult:
    """Probe a candidate URL with a bounded ranged GET (<= max_bytes) and
    report status/content-type/%PDF magic. Never downloads whole documents."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/pdf,*/*",
            "Range": f"bytes=0-{max_bytes - 1}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(max_bytes)
            return ProbeResult(
                url=url,
                status=getattr(resp, "status", None),
                content_type=resp.headers.get("Content-Type", ""),
                pdf_magic=body.startswith(b"%PDF"),
            )
    except Exception as exc:  # probe must never raise; callers report it
        return ProbeResult(url=url, error=str(exc))


def candidates_from_listing_html(
    html_text: str,
    *,
    company_id: str,
    scrip_code: str,
    document_type: str,
    fiscal_period: str,
    source_page: str,
) -> list[dict]:
    """Build candidate rows from an already-fetched listing page (HTML
    listing mode). Fields left blank by design are filled by the caller's
    annotation step, matching build_candidates behaviour."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    return [
        {
            "company_id": company_id,
            "scrip_code": scrip_code,
            "document_type": document_type,
            "fiscal_period": fiscal_period,
            "filing_date": "",
            "url": url,
            "source_page": source_page,
            "discovered_at": now,
        }
        for url in extract_pdf_links(html_text)
    ]


def load_company_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def load_companies(path: str | Path) -> list[Company]:
    out: list[Company] = []
    for row in load_company_rows(path):
        out.append(
            Company(
                company_id=(row.get("company_id") or row.get("scrip_code") or "").strip(),
                company_name=(row.get("company_name") or "").strip(),
                scrip_code=(row.get("scrip_code") or "").strip(),
                isin=(row.get("isin") or "").strip(),
            )
        )
    return out


def load_discovery_config(path: str | Path) -> DiscoveryConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DiscoveryConfig(
        fiscal_years=[str(v) for v in payload.get("fiscal_years") or []],
        doc_types=[str(v) for v in payload.get("doc_types") or []],
        listing_urls={str(k): str(v) for k, v in (payload.get("listing_urls") or {}).items()},
    )


def discover_filing_urls(
    company: Company, fetcher: Fetcher, cfg: DiscoveryConfig
) -> list[Filing]:
    listing_url = cfg.listing_urls.get(company.scrip_code)
    if not listing_url:
        listing_url = cfg.listing_urls.get(company.company_id)
    if not listing_url:
        return []
    data, _attempt = fetcher.fetch(listing_url, company.scrip_code)
    text = data.decode("utf-8", errors="replace")
    filings: list[Filing] = []
    for match in _PDF_LINK_RE.finditer(text):
        url = match.group(1)
        filings.append(Filing(url=url, source_page=listing_url))
    return filings


def build_candidates(
    companies: list[Company], cfg: DiscoveryConfig, fetcher: Fetcher
) -> tuple[list[dict], list[dict], list[str]]:
    by_url: dict[str, dict] = {}
    by_cdp: dict[tuple[str, str, str], dict] = {}
    candidates: list[dict] = []
    duplicates: list[dict] = []
    unmatched: list[str] = []

    def offer(row: dict) -> None:
        url = row["url"]
        cdp = (row["company_id"], row["document_type"], row["fiscal_period"])
        if url in by_url:
            existing = by_url[url]
            if str(row.get("filing_date") or "") < str(existing.get("filing_date") or ""):
                duplicates.append(existing)
                by_url[url] = row
            else:
                duplicates.append(row)
            return
        if cdp in by_cdp:
            existing = by_cdp[cdp]
            if str(row.get("filing_date") or "") < str(existing.get("filing_date") or ""):
                duplicates.append(existing)
                by_cdp[cdp] = row
            else:
                duplicates.append(row)
            return
        by_url[url] = row
        by_cdp[cdp] = row
        candidates.append(row)

    for company in companies:
        listing_url = cfg.listing_urls.get(company.scrip_code)
        if not listing_url:
            listing_url = cfg.listing_urls.get(company.company_id)
        if not listing_url:
            unmatched.append(f"no listing url for company {company.company_id}")
            continue
        filings = discover_filing_urls(company, fetcher, cfg)
        for filing in filings:
            doc_type = filing.doc_type or (cfg.doc_types[0] if cfg.doc_types else "annual")
            offer(
                {
                    "company_id": company.company_id,
                    "scrip_code": company.scrip_code,
                    "document_type": doc_type,
                    "fiscal_period": filing.fiscal_period or "",
                    "filing_date": filing.filing_date or "",
                    "url": filing.url,
                    "source_page": filing.source_page or "",
                    "discovered_at": datetime.datetime.now(datetime.UTC).isoformat(),
                }
            )

    return candidates, duplicates, unmatched


def write_candidates(candidates: list[dict], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATES_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            writer.writerow({c: row.get(c, "") for c in CANDIDATES_COLUMNS})


def write_duplicates(duplicates: list[dict], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_DUPLICATES_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in duplicates:
            writer.writerow({c: row.get(c, "") for c in _DUPLICATES_COLUMNS})


def validate_companies(path: str | Path) -> list[str]:
    problems: list[str] = []
    required = [
        "company_name",
        "scrip_code",
        "isin",
        "company_id",
        "verification_status",
        "source_url",
        "checked_on",
        "notes",
    ]
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        missing = [c for c in required if c not in fieldnames]
        if missing:
            return [f"{path}: missing required columns: {', '.join(missing)}"]
        seen_company: set[str] = set()
        seen_scrip: set[str] = set()
        seen_isin: set[str] = set()
        for i, row in enumerate(reader, start=2):
            cid = (row.get("company_id") or "").strip()
            scrip = (row.get("scrip_code") or "").strip()
            isin = (row.get("isin") or "").strip()
            vstatus = (row.get("verification_status") or "").strip()
            name = (row.get("company_name") or "").strip()
            if cid and cid in seen_company:
                problems.append(f"{path}: line {i}: duplicate company_id {cid}")
            if cid:
                seen_company.add(cid)
            if scrip and scrip in seen_scrip:
                problems.append(f"{path}: line {i}: duplicate scrip_code {scrip}")
            if scrip:
                seen_scrip.add(scrip)
            if isin and isin in seen_isin:
                problems.append(f"{path}: line {i}: duplicate ISIN {isin}")
            if isin:
                seen_isin.add(isin)
            if vstatus == "VERIFIED":
                for col, val in (
                    ("company_name", name),
                    ("scrip_code", scrip),
                    ("isin", isin),
                    ("company_id", cid),
                ):
                    if not val:
                        problems.append(f"{path}: line {i}: VERIFIED row has empty {col}")
            if isin and not _ISIN_RE.fullmatch(isin):
                problems.append(f"{path}: line {i}: invalid ISIN format {isin}")
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="src.ingestion.discovery")
    parser.add_argument("--companies", required=True, help="CSV: company_name, scrip_code, isin")
    parser.add_argument("--config", required=True, help="JSON config (fiscal_years, doc_types)")
    parser.add_argument("--out", default=None, help="Output candidates.csv path")
    parser.add_argument("--duplicates", default=None, help="Output duplicates.csv path")
    parser.add_argument("--unmatched", default=None, help="Output unmatched report path")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate a companies file without network access",
    )
    parser.add_argument(
        "--allow-needs-verification",
        action="store_true",
        help="Allow discovery on companies files containing NEEDS_VERIFICATION rows",
    )
    args = parser.parse_args(argv)
    if args.validate:
        problems = validate_companies(args.companies)
        for problem in problems:
            print(problem)
        return 0 if not problems else 1
    rows = load_company_rows(args.companies)
    if not args.allow_needs_verification and any(
        (r.get("verification_status") or "").strip() == "NEEDS_VERIFICATION" for r in rows
    ):
        print(
            f"refusing to run discovery on {args.companies}: "
            "file contains NEEDS_VERIFICATION rows; pass --allow-needs-verification"
        )
        return 2
    companies = load_companies(args.companies)
    cfg = load_discovery_config(args.config)
    fetcher = Fetcher()
    candidates, duplicates, unmatched = build_candidates(companies, cfg, fetcher)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data") / "bulk_discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"candidates_{stamp}.csv"
    write_candidates(candidates, out_path)
    if args.duplicates:
        write_duplicates(duplicates, Path(args.duplicates))
    for reason in unmatched:
        print(reason)
    print(f"wrote {len(candidates)} candidates to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
