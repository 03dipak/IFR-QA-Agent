"""Bulk acquisition discovery: companies + config -> filing candidates.

Phase 1 of the bulk-acquisition plan (task LLD-35): read the owner-supplied
companies file and discovery config, list filing candidates per company from
the configured BSE-hosted listing pages, and emit candidates.csv.
"""

from __future__ import annotations

import csv
import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .fetcher import Fetcher

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
