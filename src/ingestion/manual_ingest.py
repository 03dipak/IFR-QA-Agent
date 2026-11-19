"""Ingest PDFs dropped into data/manual_inbox/<company_code>/ into the ledger.

Read-only discovery is not performed here: this command registers operator-placed
PDFs. For each PDF it computes sha256, dedupes against the ledger, applies the
same gate as the pipeline (analyze_pdf), copies the PDF into raw storage and
appends one ledger document row marked source=manual.

Doc type / fiscal period / variant come from a per-directory manifest
(manifest.json: list of {file, company, period, doc_type, variant}) or, when
absent, are parsed from the filename.

Hardened mode (default --dry-run) reports per-file decisions without writing
to raw or the ledger. Real ingest requires --apply.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from .extractor import extract_pages
from .gate_report import analyze_pdf, sha256_file
from .models import DocType
from .period_detection import detect_period_from_content, pdf_metadata_dates

VALID_TYPES = {d.value for d in DocType}

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

DEFAULT_COMPANY_FILES = [
    Path("data/seeds/companies_proposed.csv"),
    Path("data/acquisition/company_universe.csv"),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_company_registry(paths):
    """Return dict code -> {display_name, tickers:set}."""
    reg = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            for row in reader:
                if "code" in fields:
                    cid = (row.get("code") or "").strip()
                    name = (row.get("display_name") or cid).strip()
                    ticker = (row.get("bse_ticker") or "").strip().upper()
                    aliases = (row.get("aliases") or "").strip()
                else:
                    cid = (row.get("company_id") or "").strip()
                    name = (row.get("company_name") or row.get("company") or cid).strip()
                    ticker = ""
                    aliases = ""
                if not cid:
                    continue
                tickers = set()
                if ticker:
                    tickers.add(ticker)
                for a in aliases.split("|"):
                    a = a.strip().upper()
                    if a:
                        tickers.add(a)
                if cid not in reg:
                    reg[cid] = {"display_name": name, "tickers": set()}
                reg[cid]["display_name"] = reg[cid]["display_name"] or name
                reg[cid]["tickers"] |= tickers
                if cid not in reg[cid]["tickers"]:
                    reg[cid]["tickers"].add(cid.upper())
    return reg


def _load_company_codes(registry_paths, fallback_paths):
    reg = _load_company_registry(registry_paths)
    codes = set(reg)
    names = {c: reg[c]["display_name"] for c in reg}
    tickers = {c: reg[c]["tickers"] for c in reg}
    # merge fallback csvs
    for path in fallback_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cid = (row.get("company_id") or "").strip()
                if not cid:
                    continue
                codes.add(cid)
                name = (row.get("company_name") or row.get("company") or cid).strip()
                names.setdefault(cid, name)
                tickers.setdefault(cid, {cid.upper()})
    return codes, names, tickers


def _norm_fy_year(value: str) -> int:
    value = value.strip()
    if len(value) == 2:
        value = "20" + value
    return int(value)




_STATEMENT_HEADINGS = {
    "bs": ("balance sheet",),
    "pl": ("statement of profit and loss", "profit and loss", "profit & loss"),
    "cf": ("cash flow statement", "statement of cash flows"),
}


def _heading_match(text: str):
    """Return (short, scope) for a real statement title line, else None.

    Only short lines that *are* the statement title (optionally with a
    "standalone"/"consolidated" prefix and an "as at/for the year" suffix)
    count. This deliberately ignores the same words inside prose such as the
    auditors' report opinion or the notes.
    """
    for raw in (text or "").splitlines():
        line = " ".join(raw.split()).strip().lower()
        if not line or len(line) > 100 or line.endswith("."):
            continue
        if (" when " in line or " under the head " in line
                or " prepared on " in line):
            continue
        for short, keys in _STATEMENT_HEADINGS.items():
            for key in keys:
                if (line == key or line == f"standalone {key}"
                        or line == f"consolidated {key}"
                        or line.startswith(f"{key} as at ")
                        or line.startswith(f"{key} for the year")
                        or line.startswith(f"{key} for the period")
                        or line.startswith(f"{key} (contd")
                        or line.startswith(f"standalone {key} ")
                        or line.startswith(f"consolidated {key} ")):
                    scope = ("consolidated" if "consolidated" in line
                             else "standalone" if "standalone" in line
                             else None)
                    return short, scope
    return None


def _new_statement_result():
    return {"standalone_bs": None, "standalone_pl": None, "standalone_cf": None,
            "consolidated_bs": None, "consolidated_pl": None,
            "consolidated_cf": None}


def _update_statement_pages(res, state, page_no, text):
    """Update ``res`` from one page. ``page_no`` is 1-indexed physical."""
    low = (text or "").lower()
    head = " ".join(
        ln for ln in (text or "").splitlines()[:8]
    ).lower()
    if "standalone financial statements" in low[:400]:
        state["scope"] = "standalone"
    if "consolidated financial statements" in low[:400]:
        state["scope"] = "consolidated"
    # The auditor's report and the notes quote every statement title inside
    # prose (e.g. "The Consolidated Balance Sheet, ..."). Skip those pages;
    # the section marker above still updates the running scope.
    if ("independent auditor" in low or "other auditors" in low
            or "we have audited" in low or "report on the audit" in low
            or "notes to the financial statements" in head
            or "notes to the consolidated financial statements" in head):
        return
    match = _heading_match(text)
    if not match:
        return
    short, heading_scope = match
    scope = heading_scope or state["scope"]
    if scope not in ("standalone", "consolidated"):
        return
    field = f"{scope}_{short}"
    if res[field] is None:
        res[field] = page_no


def detect_statement_pages(texts):
    """Return statement page numbers as **1-indexed physical PDF pages**.

    Page 1 is the first page of the PDF, matching ``PageExtract.pdf_page_index``.
    """
    res = _new_statement_result()
    state = {"scope": None}
    for page_no, text in enumerate(texts, start=1):
        _update_statement_pages(res, state, page_no, text)
    return res


def detect_statement_pages_in_pdf(path):
    """Bounded, one-page-at-a-time statement detection (no whole-file read)."""
    from pypdf import PdfReader

    res = _new_statement_result()
    state = {"scope": None}
    reader = PdfReader(str(path))
    try:
        for idx in range(len(reader.pages)):
            try:
                text = reader.pages[idx].extract_text() or ""
            except Exception:
                text = ""
            _update_statement_pages(res, state, idx + 1, text)
    finally:
        try:
            reader.close()
        except Exception:
            pass
    return res

def _ticker_from_filename(name: str):
    m = re.search(r"AR_\d+_([A-Z0-9]+)_\d{4}", name)
    if m:
        return m.group(1).upper()
    m = re.search(r"_([A-Z]{3,})_", name)
    if m:
        return m.group(1).upper()
    return None


def _name_score(company_name: str, text: str) -> float:
    toks = [t for t in re.split(r"\W+", (company_name or "").lower()) if len(t) >= 3]
    toks = [t for t in toks if t not in _GENERIC]
    if not toks:
        toks = [company_name.lower()]
    low = (text or "").lower()
    hits = sum(1 for t in toks if t in low)
    return hits / len(toks) if toks else 0.0



def parse_period(text: str):
    """Return (period, doc_type_hint, reason). period is None on failure."""
    text = (text or "").strip()
    if not text:
        return None, None, "empty period"
    # Q1FY2025 / Q1_FY25 / Q1 FY25
    m = re.search(r"Q([1-4])\s*[_\-\s]*FY\s*'?(\d{2,4})", text, re.I)
    if m:
        return f"Q{m.group(1)}FY{_norm_fy_year(m.group(2))}", "quarterly", None
    # "June 2024 quarter"
    m = re.search(
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+(\d{4})\s+quarter", text, re.I)
    if m:
        month = MONTHS[m.group(1).lower()[:3]]
        year = int(m.group(2))
        if month in (4, 5, 6):
            q, fy = 1, year + 1
        elif month in (7, 8, 9):
            q, fy = 2, year + 1
        elif month in (10, 11, 12):
            q, fy = 3, year + 1
        else:
            q, fy = 4, year
        return f"Q{q}FY{fy}", "quarterly", None
    # FY2024 / FY24
    m = re.search(r"FY\s*'?(\d{2,4})", text, re.I)
    if m and "Q" not in text.upper():
        return f"FY{_norm_fy_year(m.group(1))}", "annual", None
    # AR_2017_2018 / 2023-24
    m = re.search(r"(\d{4})\s*[-_/]\s*(\d{2,4})", text)
    if m:
        y1 = int(m.group(1))
        y2 = _norm_fy_year(m.group(2))
        if y2 >= y1 and y2 - y1 <= 1:
            return f"FY{y2}", "annual", None
    m = re.search(r"(?<!\d)(20\d{2})(?!\d)", text)
    if m:
        return f"FY{m.group(1)}", "annual", None
    return None, None, "unparsed period"



def parse_period_from_text(text: str):
    """Backward-compatible 3-tuple wrapper around the content detector."""
    det = detect_period_from_content(text)
    hint = det["doc_type"] if det["doc_type"] in ("quarterly", "annual") else None
    return det["period"], hint, det["reason"]


def parse_period_from_filename_date(name: str):
    m = re.search(r"_(\d{2})(\d{2})(\d{4})", name)
    if not m:
        return None, None, "no date in filename"
    day, mon, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if mon < 1 or mon > 12:
        return None, None, "bad month"
    if mon in (4, 5, 6):
        q, fy = 1, year + 1
    elif mon in (7, 8, 9):
        q, fy = 2, year + 1
    elif mon in (10, 11, 12):
        q, fy = 3, year + 1
    else:
        q, fy = 4, year
    # only use as quarterly hint; annual unknown
    return f"Q{q}FY{fy}", "quarterly", None

def _fy_number(period: str) -> int:
    m = re.search(r"FY(\d{4})", period or "")
    return int(m.group(1)) if m else 0


def _quarter_number(period: str) -> int:
    m = re.search(r"Q([1-4])", period or "")
    return int(m.group(1)) if m else 0


def is_core_slice(period: str, doc_type: str) -> bool:
    if doc_type == "annual":
        return _fy_number(period) >= 2024
    if doc_type == "quarterly":
        fy = _fy_number(period)
        q = _quarter_number(period)
        if fy > 2025:
            return True
        if fy == 2025 and q >= 1:
            return True
        return False
    return True


def _parse_filename(name: str):
    stem = Path(name).stem
    parts = stem.split("_")
    company = parts[0] if parts else ""
    doc_type = None
    period = ""
    variant = "single"
    for p in parts:
        if p.lower() in ("standalone", "consolidated"):
            variant = p.lower()
    for p in parts[2:]:
        if p.upper().startswith("FY") and len(p) >= 4:
            period = p.upper()
        elif re.match(r"Q[1-4]", p, re.I):
            period = p.upper().replace("_", "")
    if "annual" in name.lower() or name.lower().startswith("ar"):
        doc_type = "annual"
    elif "quarter" in name.lower() or re.search(r"q[1-4]", name, re.I):
        doc_type = "quarterly"
    return company, doc_type, period, variant


def _first_n_pages_text(path: Path, n: int = 5):
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        texts = []
        for i in range(min(n, page_count)):
            try:
                texts.append(reader.pages[i].extract_text() or "")
            except Exception:
                texts.append("")
        return page_count, texts, False
    except Exception:
        try:
            pages = extract_pages(path.read_bytes(), "report", "report")
        except Exception:
            return 0, [], True
        texts = [p.page_text for p in pages[:n]]
        return len(pages), texts, False


def infer_doc_type(path: Path, manifest_doc_type, period_hint, page_count, first_text):
    if manifest_doc_type in VALID_TYPES:
        return manifest_doc_type, "high"
    if period_hint in ("annual", "quarterly"):
        return period_hint, "medium"
    lower = path.name.lower()
    if "quarter" in lower or re.search(r"(^|[_\-])q[1-4]", lower):
        return "quarterly", "medium"
    if "annual" in lower or lower.startswith("ar_") or lower.startswith("ar-"):
        return "annual", "medium"
    low_text = (first_text or "").lower()
    if "quarter" in low_text or "financial results" in low_text:
        return "quarterly", "low"
    if "annual report" in low_text:
        return "annual", "low"
    if page_count >= 40:
        return "annual", "low"
    if 0 < page_count <= 30:
        return "quarterly", "low"
    return "annual", "low"


_GENERIC = {"limited", "ltd", "private", "pvt", "india", "industries",
            "technologies", "technology", "corporation", "corp", "company",
            "co", "inc", "llc", "plc", "group", "enterprises"}


def company_matches(folder_code: str, company_name: str, first_text: str) -> bool:
    text = (first_text or "").lower()
    if folder_code and len(folder_code) >= 3 and folder_code.lower() in text:
        return True
    tokens = []
    for t in re.split(r"\W+", (company_name or "").lower()):
        if len(t) >= 3 and t not in _GENERIC:
            tokens.append(t)
    if folder_code and len(folder_code) >= 3:
        tokens.append(folder_code.lower())
    tokens = list(dict.fromkeys(tokens))
    return any(tok in text for tok in tokens)


def _load_manifest(inbox_dir: Path):
    m = inbox_dir / "manifest.json"
    if m.exists():
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {e.get("file"): e for e in data if e.get("file")}
            if isinstance(data, dict) and "entries" in data:
                return {e.get("file"): e for e in data.get("entries", []) if e.get("file")}
            return {}
        except Exception:
            return {}
    return {}


def _file_signature(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _load_hash_cache(path: Path) -> dict:
    """Load the decision cache; stale `unparsed` verdicts are dropped so the
    content-based detector reprocesses those files on the next dry run."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    fresh = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        row = entry.get("row")
        if isinstance(row, dict) and row.get("decision") == "unparsed":
            continue  # re-run the content detector on unparsed files
        fresh[key] = entry
    return fresh


def _save_hash_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _load_ledger_state(ledger_path: Path):
    """Return (checksums, document rows, withdrawn document ids)."""
    existing = set()
    docs = []
    withdrawn = set()
    if not ledger_path.exists():
        return existing, docs, withdrawn
    with ledger_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            kind = rec.get("kind")
            data = rec.get("data") or {}
            if kind == "document":
                if data.get("checksum"):
                    existing.add(data["checksum"])
                docs.append(data)
            elif kind == "withdrawn":
                if data.get("document_id"):
                    withdrawn.add(data["document_id"])
    return existing, docs, withdrawn


def _doc_key(data: dict):
    return (data.get("company_id"), data.get("fiscal_year"),
            data.get("doc_type"), data.get("variant"))


def _copy_verified(src: Path, dest: Path, expected_sha256: str) -> bool:
    """Copy a PDF into raw/ and verify the copy's sha256 matches.

    A truncated/partial copy (seen with the tcs AR: 3.9MB vs 17.4MB intact)
    claims a ledger checksum it never held; every copy is re-hashed after
    writing and the corrupt file is removed and retried once. Fails loudly.
    Returns True when a copy was performed, False when an existing identical
    file was already intact.
    """
    expected = expected_sha256.strip().lower()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fresh_copy = False
    for attempt in (1, 2):
        if dest.exists() and not fresh_copy:
            if sha256_file(dest) == expected:
                return False
            dest.unlink()
        shutil.copy2(src, dest)
        fresh_copy = True
        if sha256_file(dest) == expected:
            return True
        dest.unlink()
    raise ValueError(
        f"sha256 mismatch after copy into raw/: {dest} does not hash to "
        f"{expected} (source {src})")


def _append_ledger_record(ledger_path: Path, kind: str, data: dict) -> None:
    """Append+fsync one row so a killed run keeps every finished document."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": 1, "kind": kind, "data": data}
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False,
                            separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})


def update_coverage_matrix(matrix_path: Path, company: str, doc_type: str,
                          period: str, variant: str, core_slice: bool):
    if not matrix_path.exists():
        return
    with matrix_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []
    if not fieldnames:
        return
    match = None
    for row in rows:
        if (row.get("company_id") == company and row.get("document_type") == doc_type
                and row.get("fiscal_period") == period
                and (row.get("variant") in (None, "", variant))):
            match = row
            break
    if match is not None:
        match["status"] = "FOUND"
        match["variant"] = variant
        match["core_slice"] = "true" if core_slice else "false"
    else:
        new_row = {fn: "" for fn in fieldnames}
        new_row.update({
            "company_id": company,
            "document_type": doc_type,
            "fiscal_period": period,
            "url": "",
            "status": "FOUND",
            "source_page": "manual_inbox",
            "collected_by": "manual_ingest",
            "variant": variant,
            "core_slice": "true" if core_slice else "false",
        })
        rows.append(new_row)
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="src.ingestion.manual_ingest",
                                     description="Ingest manual_inbox PDFs into ledger")
    parser.add_argument("--inbox", default="data/manual_inbox")
    parser.add_argument("--ledger", default="data/corpus/ledger.jsonl")
    parser.add_argument("--raw", default="data/corpus/raw")
    parser.add_argument("--coverage-matrix", default="data/seeds/coverage_matrix.csv")
    parser.add_argument("--companies-proposed", default="data/seeds/companies_proposed.csv")
    parser.add_argument("--company-universe", default="data/acquisition/company_universe.csv")
    parser.add_argument("--company-registry", default="data/acquisition/company_registry.csv")
    parser.add_argument("--hash-cache", default=None,
                        help="path+size+mtime hash/decision cache "
                             "(default: <ledger dir>/hash_cache.json)")
    parser.add_argument("--apply", action="store_true",
                        help="actually write to raw/ledger; default is dry-run")
    parser.add_argument("--company", default=None,
                        help="only process this company code "
                             "(comma-separated codes allowed)")
    parser.add_argument("--batch", type=int, default=None,
                        help="1-indexed batch of companies to process; "
                             "combine with --batch-size")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="companies per --batch (default 5)")
    parser.add_argument("--all-slice", action="store_true",
                        help="inbox-wide mode: mark every ingested doc as core, "
                             "bypassing is_core_slice (default behavior unchanged)")
    args = parser.parse_args(argv)
    if args.batch is not None and args.batch < 1:
        parser.error("--batch must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    inbox = Path(args.inbox)
    ledger_path = Path(args.ledger)
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(args.coverage_matrix)
    registry_paths = [Path(args.company_registry)]
    fallback_paths = [Path(args.companies_proposed), Path(args.company_universe)]

    company_codes, company_names, company_tickers = _load_company_codes(
        registry_paths, fallback_paths)
    existing, ledger_docs, withdrawn_ids = _load_ledger_state(ledger_path)
    key_index = {}
    for doc in ledger_docs:
        key_index.setdefault(_doc_key(doc), []).append(doc)
    doc_by_checksum = {d.get("checksum"): d for d in ledger_docs
                       if d.get("checksum")}

    hash_cache_path = (Path(args.hash_cache) if args.hash_cache
                       else ledger_path.parent / "hash_cache.json")
    hash_cache = _load_hash_cache(hash_cache_path)
    new_cache = {}

    dry_rows = []
    unparsed_rows = []
    quarantine_rows = []
    withdrawn_rows = []
    decisions = Counter()
    added = kept = excluded = unreadable = dupes = skipped = reused = 0
    current_checksums = set()

    merge_cache = bool(args.company or args.batch is not None)

    def _persist_cache():
        if merge_cache:
            merged = dict(hash_cache)
            merged.update(new_cache)
            _save_hash_cache(hash_cache_path, merged)
        else:
            _save_hash_cache(hash_cache_path, new_cache)

    company_dirs = sorted(p for p in inbox.iterdir() if p.is_dir())
    if args.company:
        wanted = {c.strip() for c in args.company.split(",") if c.strip()}
        company_dirs = [p for p in company_dirs if p.name in wanted]
    elif args.batch is not None:
        start = (args.batch - 1) * args.batch_size
        company_dirs = company_dirs[start:start + args.batch_size]
    if (args.company or args.batch is not None) and not company_dirs:
        print("no companies selected by --company/--batch", file=sys.stderr)
        return 2

    for company_dir in company_dirs:
        code = company_dir.name
        if code not in company_codes:
            for pdf in sorted(company_dir.glob("*.pdf")):
                print(f"SKIP unknown company: {code}/{pdf.name}")
                dry_rows.append({
                    "company": code, "file": pdf.name, "sha256": "",
                    "period": "", "doc_type": "", "doc_type_confidence": "",
                    "variant": "", "duplicate": "", "core_slice": "",
                    "decision": "skip_unknown_company",
                    "problem": "unknown company code",
                })
                decisions["skip_unknown_company"] += 1
                skipped += 1
            continue

        manifest = _load_manifest(company_dir)
        seen_this_run = set()
        withdraw_names = {
            name: entry for name, entry in manifest.items()
            if entry.get("withdrawn")
            or entry.get("action") in ("withdraw", "withdrawn")
        }
        for name in sorted(withdraw_names):
            entry = withdraw_names[name]
            did = entry.get("checksum") or entry.get("document_id")
            if not did:
                candidate = company_dir / name
                did = sha256_file(candidate) if candidate.exists() else ""
            if not did:
                continue
            reason = entry.get("reason") or "withdrawn by manifest"
            withdrawn_rows.append({"filename": name, "folder": code,
                                   "document_id": did, "reason": reason})
            if args.apply and did not in withdrawn_ids:
                _append_ledger_record(ledger_path, "withdrawn", {
                    "document_id": did, "company_id": code, "checksum": did,
                    "reason": reason, "raw_kept": True, "recorded_at": _now(),
                })
                withdrawn_ids.add(did)

        for pdf in sorted(company_dir.glob("*.pdf")):
            if pdf.name in withdraw_names:
                continue
            cache_key = str(pdf.resolve())
            sig = _file_signature(pdf)
            cached = hash_cache.get(cache_key)
            cache_ok = (isinstance(cached, dict)
                        and cached.get("size") == sig["size"]
                        and cached.get("mtime_ns") == sig["mtime_ns"]
                        and isinstance(cached.get("row"), dict))
            checksum = cached["sha256"] if cache_ok else sha256_file(pdf)
            # A "new" row cached by an earlier dry run still has to be applied;
            # only take the reporting-only shortcut when there is nothing to do.
            reuse = cache_ok and not (
                args.apply and cached["row"].get("decision") == "new")
            current_checksums.add(checksum)
            dup = checksum in existing or checksum in seen_this_run
            seen_this_run.add(checksum)

            if reuse:
                row = dict(cached["row"])
                row["duplicate"] = "yes" if dup else "no"
                if dup:
                    row["decision"] = "duplicate"
                    row["problem"] = "duplicate hash"
                dry_rows.append(row)
                decisions[row["decision"]] += 1
                if dup:
                    dupes += 1
                if row["decision"] == "unparsed":
                    q = cached.get("unparsed") or {}
                    unparsed_rows.append({
                        "filename": pdf.name, "folder": code,
                        "reason": q.get("reason", row.get("problem", "")),
                    })
                    skipped += 1
                elif row["decision"] == "quarantine":
                    q = cached.get("quarantine") or {}
                    quarantine_rows.append({
                        "filename": pdf.name, "folder": code,
                        "reason": q.get("reason", row.get("problem", "")),
                        "score": q.get("score", ""),
                        "snippet": q.get("snippet", ""),
                    })
                    skipped += 1
                elif row["decision"] == "unreadable":
                    skipped += 1
                reused += 1
                new_cache[cache_key] = cached
                continue

            if dup:
                # Already in the ledger: do not open the PDF again.
                _company, fn_doc, fn_period, fn_variant = _parse_filename(pdf.name)
                quick_period = fn_period
                if not quick_period:
                    quick_period, _hint, _reason = parse_period(pdf.name)
                row = {
                    "company": code, "file": pdf.name, "sha256": checksum,
                    "period": quick_period or "", "doc_type": fn_doc or "",
                    "doc_type_confidence": "medium" if fn_doc else "",
                    "variant": fn_variant or "single", "version": "single",
                    "other_report": "false", "duplicate": "yes",
                    "core_slice": "false", "decision": "duplicate",
                    "problem": "duplicate hash", "supersedes": "",
                }
                dry_rows.append(row)
                decisions["duplicate"] += 1
                dupes += 1
                new_cache[cache_key] = {
                    "size": sig["size"], "mtime_ns": sig["mtime_ns"],
                    "sha256": checksum, "row": row,
                }
                if args.apply:
                    prior_doc = doc_by_checksum.get(checksum)
                    if prior_doc and prior_doc.get("source") == "manual":
                        # Finish anything a killed run left before the matrix
                        # write (idempotent: existing rows are updated).
                        update_coverage_matrix(
                            matrix_path, code,
                            prior_doc.get("doc_type") or "",
                            prior_doc.get("fiscal_year") or "",
                            prior_doc.get("variant") or "single",
                            bool(prior_doc.get("core_slice")))
                print(f"{code}: {pdf.name} -> duplicate (ledger)")
                continue

            meta = manifest.get(pdf.name, {})
            # period
            period_text = meta.get("period") if meta.get("period") is not None else pdf.name
            if meta.get("period") == "":
                period_text = pdf.name
            period, hint, reason = parse_period(period_text)
            period_src = "name" if period is not None else None
            det = None
            if period is None:
                # try PDF text (content-based scorer; first five pages)
                pc, pts, un = _first_n_pages_text(pdf, 20)
                if not un and pts:
                    det = detect_period_from_content(
                        "\f".join(pts), meta_dates=pdf_metadata_dates(pdf))
                    if det["period"] and det["confidence"] in ("high", "medium"):
                        period = det["period"]
                        period_src = "content"
                        hint = det["doc_type"] if det["doc_type"] in (
                            "quarterly", "annual") else None
                        reason = None
                    elif det["period"] is not None:
                        # conflicting / low-confidence content: never guess
                        reason = det["reason"]
            if period is None and det is None:
                # no readable text evidence: legacy filename-date fallback
                period, hint, reason = parse_period_from_filename_date(pdf.name)
                if period is not None:
                    period_src = "filename_date"
            period_confidence = (
                det["confidence"] if period_src == "content" and det
                else {"name": "high", "filename_date": "medium"}.get(period_src, "low")
            )
            if period is None:
                reason = reason or "unparsed period"
                unparsed_reason = f"{reason}: {det['reason']}" if det and det.get("reason") else reason
                print(f"UNPARSED: {code}/{pdf.name} ({unparsed_reason})")
                unparsed_rows.append({
                    "filename": pdf.name, "folder": code,
                    "reason": unparsed_reason,
                    "snippet": (det or {}).get("snippet", ""),
                    "period_snippet": (det or {}).get("snippet", ""),
                    "matched_snippet": (det or {}).get("snippet", ""),
                    "dates_found": "; ".join((det or {}).get("dates_found", [])),
                    "dates": "; ".join((det or {}).get("dates_found", [])),
                    "suggested_period": (det or {}).get("suggested_period") or "",
                    "candidates": (det or {}).get("candidates", ""),
                })
                row = {
                    "company": code, "file": pdf.name, "sha256": checksum,
                    "period": "", "doc_type": "", "doc_type_confidence": "",
                    "period_confidence": "low", "confidence": "low",
                    "variant": meta.get("variant") or "single",
                    "version": "single", "other_report": "false",
                    "duplicate": "yes" if dup else "no", "core_slice": "",
                    "decision": "unparsed", "problem": unparsed_reason,
                    "supersedes": "",
                    "snippet": (det or {}).get("snippet", ""),
                    "period_snippet": (det or {}).get("snippet", ""),
                    "matched_snippet": (det or {}).get("snippet", ""),
                    "audit_status": (det or {}).get("audit_status", ""),
                    "dates_found": "; ".join((det or {}).get("dates_found", [])),
                }
                dry_rows.append(row)
                decisions["unparsed"] += 1
                skipped += 1
                new_cache[cache_key] = {
                    "size": sig["size"], "mtime_ns": sig["mtime_ns"],
                    "sha256": checksum, "row": row,
                    "unparsed": {"reason": unparsed_reason},
                }
                continue

            page_count, page_texts, unreadable_pdf = _first_n_pages_text(pdf, 5)
            # retry once if unreadable
            if unreadable_pdf:
                page_count, page_texts, unreadable_pdf = _first_n_pages_text(pdf, 5)
            if unreadable_pdf:
                print(f"UNREADABLE: {code}/{pdf.name}")
                row = {
                    "company": code, "file": pdf.name, "sha256": checksum,
                    "period": period, "doc_type": "", "doc_type_confidence": "",
                    "period_confidence": period_confidence,
                    "confidence": period_confidence,
                    "snippet": "", "period_snippet": "", "matched_snippet": "",
                    "audit_status": "", "dates_found": "",
                    "variant": meta.get("variant") or "single",
                    "version": "single", "other_report": "false",
                    "duplicate": "yes" if dup else "no", "core_slice": "",
                    "decision": "unreadable", "problem": "unreadable pdf",
                    "supersedes": "",
                }
                dry_rows.append(row)
                decisions["unreadable"] += 1
                skipped += 1
                new_cache[cache_key] = {
                    "size": sig["size"], "mtime_ns": sig["mtime_ns"],
                    "sha256": checksum, "row": row,
                }
                continue

            first_text = page_texts[0] if page_texts else ""
            if code == "asian_paints" and "AR_14491_ASIANPAINT_2018_2019" in pdf.name:
                try:
                    sp = detect_statement_pages_in_pdf(pdf)
                except Exception:
                    sp = detect_statement_pages(page_texts)
                print(f"ASIAN_PAINTS_STATEMENT_PAGES {pdf.name} (1-indexed): "
                      f"standalone BS={sp['standalone_bs']} P&L={sp['standalone_pl']} "
                      f"CF={sp['standalone_cf']}; consolidated BS={sp['consolidated_bs']} "
                      f"P&L={sp['consolidated_pl']} CF={sp['consolidated_cf']}")
            ticker = _ticker_from_filename(pdf.name)
            registry_tickers = company_tickers.get(code, set())
            quarantine_reason = None
            score = 1.0
            snippet = (first_text or "")[:160].replace("\n", " ")
            best_text = first_text
            if ticker:
                if ticker not in registry_tickers:
                    quarantine_reason = (f"ticker mismatch: file {ticker} "
                                         f"vs folder {code}")
                    score = 0.0
            elif not any((t or "").strip() for t in page_texts):
                # image-only scan: no text layer to match against; the
                # company folder + manifest entry carry the identity and
                # the gate will flag it as image-only
                score = 1.0
                snippet = "image-only scan: no text layer"
                best_snippet = snippet
                best_text = first_text
            else:
                # fuzzy match across first 5 pages
                best_score = 0.0
                best_snippet = snippet
                for t in page_texts:
                    sc = _name_score(company_names.get(code, code), t)
                    if sc > best_score:
                        best_score = sc
                        best_snippet = (t or "")[:160].replace("\n", " ")
                score = best_score
                if best_score < 0.4:
                    quarantine_reason = "name mismatch across first 5 pages"
                    snippet = best_snippet
            if quarantine_reason:
                print(f"QUARANTINE wrong company: {code}/{pdf.name} "
                      f"(score={score:.2f})")
                quarantine_rows.append({
                    "filename": pdf.name, "folder": code,
                    "reason": quarantine_reason,
                    "score": f"{score:.2f}",
                    "snippet": snippet,
                })
                row = {
                    "company": code, "file": pdf.name, "sha256": checksum,
                    "period": period, "doc_type": "", "doc_type_confidence": "",
                    "period_confidence": period_confidence,
                    "confidence": period_confidence,
                    "snippet": snippet, "period_snippet": snippet,
                    "matched_snippet": snippet, "audit_status": "",
                    "dates_found": "",
                    "variant": meta.get("variant") or "single",
                    "version": "single", "other_report": "false",
                    "duplicate": "yes" if dup else "no", "core_slice": "",
                    "decision": "quarantine",
                    "problem": f"{quarantine_reason} score={score:.2f}",
                    "supersedes": "",
                }
                dry_rows.append(row)
                decisions["quarantine"] += 1
                skipped += 1
                new_cache[cache_key] = {
                    "size": sig["size"], "mtime_ns": sig["mtime_ns"],
                    "sha256": checksum, "row": row,
                    "quarantine": {
                        "reason": quarantine_reason,
                        "score": f"{score:.2f}",
                        "snippet": snippet,
                    },
                }
                continue

            # doc_type from filename first
            lower_name = pdf.name.lower()
            if lower_name.startswith("ar_") or "_ar_" in lower_name:
                forced_type = "annual"
            elif "brr_sr" in lower_name or lower_name.startswith("brr_"):
                forced_type = "sustainability_report"
            elif "sec_audit" in lower_name or "secretarial" in lower_name:
                forced_type = "secretarial_audit"
            elif ("outcome" in lower_name or "results" in lower_name
                  or "letter" in lower_name):
                forced_type = "disclosure"
            else:
                forced_type = None
            if forced_type and not meta.get("doc_type"):
                meta = dict(meta)
                meta["doc_type"] = forced_type
                confidence_override = "high"
            else:
                confidence_override = None

            doc_type, confidence = infer_doc_type(
                pdf, meta.get("doc_type"), hint, page_count, first_text)
            if confidence_override:
                confidence = confidence_override
            if meta.get("doc_type") in VALID_TYPES:
                confidence = "high"
            # if filename provided doc type via _parse_filename, keep low-ish
            if not meta.get("doc_type"):
                fn_company, fn_doc, fn_period, fn_variant = _parse_filename(pdf.name)
                if fn_doc in VALID_TYPES and not hint:
                    doc_type = fn_doc
                    confidence = "medium"
            # content-based doc type wins over filename heuristics (Part A)
            if period_src == "content" and det and det.get("doc_type"):
                doc_type = det["doc_type"]
                if det["confidence"] in ("high", "medium"):
                    confidence = det["confidence"]
            variant = meta.get("variant") or "single"
            if not meta.get("variant"):
                _, _, _, fn_variant = _parse_filename(pdf.name)
                if fn_variant:
                    variant = fn_variant
            # standalone/consolidated from title/text
            joined_text = " ".join(page_texts or []).lower()
            if "consolidated" in joined_text or "consolidated" in pdf.name.lower():
                if not meta.get("variant"):
                    variant = "consolidated"
            elif "standalone" in joined_text or "standalone" in pdf.name.lower():
                if not meta.get("variant"):
                    variant = "standalone"
            # content detector (approved: title/heading lines only) wins
            if period_src == "content" and det and det.get("variant"):
                variant = det["variant"]
            # version tag
            version = "single"
            if "_A_" in pdf.name or pdf.name.lower().endswith("_a.pdf"):
                version = "A"
            elif "_U_" in pdf.name or pdf.name.lower().endswith("_u.pdf"):
                version = "U"
            other_report = doc_type in ("sustainability_report", "secretarial_audit")
            core = True if args.all_slice else is_core_slice(period, doc_type)
            if other_report:
                core = False

            key = (code, period, doc_type, variant)
            prior = key_index.get(key, [])
            supersedes = ""
            older = [p for p in prior if p.get("checksum") != checksum]
            if older:
                supersedes = older[-1].get("document_id") or ""
                nums = [int(m.group(1)) for p in prior
                        for m in [re.match(r"v?(\d+)$", str(p.get("version") or ""))]
                        if m]
                version = f"v{(max(nums) if nums else 1) + 1}"

            row = {
                "company": code, "file": pdf.name, "sha256": checksum,
                "period": period, "doc_type": doc_type,
                "doc_type_confidence": confidence,
                "period_confidence": period_confidence,
                "confidence": period_confidence,
                "snippet": (det or {}).get("snippet", "") if period_src == "content" else "",
                "period_snippet": (det or {}).get("snippet", "") if period_src == "content" else "",
                "matched_snippet": (det or {}).get("snippet", "") if period_src == "content" else "",
                "audit_status": (det or {}).get("audit_status", "unclear") if period_src == "content" else "",
                "dates_found": ("; ".join((det or {}).get("dates_found", []))
                                if period_src == "content" else ""),
                "covers_full_year": ("true" if period_src == "content"
                                     and (det or {}).get("covers_full_year")
                                     else "false"),
                "variant": variant,
                "version": version, "other_report": "true" if other_report else "false",
                "duplicate": "no", "core_slice": "true" if core else "false",
                "decision": "new", "problem": "", "supersedes": supersedes,
            }
            dry_rows.append(row)
            decisions["new"] += 1
            new_cache[cache_key] = {
                "size": sig["size"], "mtime_ns": sig["mtime_ns"],
                "sha256": checksum, "row": row,
            }
            print(f"{code}: {pdf.name} -> new period={period} "
                  f"doc_type={doc_type}({confidence}) variant={variant} "
                  f"core_slice={core}"
                  f"{' supersedes=' + supersedes if supersedes else ''}")

            if args.apply:
                dest = raw / f"{checksum}.pdf"
                copied = _copy_verified(pdf, dest, checksum)
                record = analyze_pdf(dest, doc_type, meta if meta else None)
                # ExtractionStatus only allows pending/extracted/qa_failed;
                # the gate outcome is carried in gate_decision.
                extraction_status = "pending"
                data = {
                    "document_id": checksum, "filing_id": checksum,
                    "doc_type": doc_type, "company_id": code,
                    "fiscal_year": period, "reporting_period": doc_type,
                    "period_type": doc_type, "period_dates": None,
                    "revision_date": None, "publication_date": _now(),
                    "audit_status": ((det or {}).get("audit_status") or "unclear")
                    if period_src == "content" else "unclear",
                    "currency": record.get("currency") or "unknown",
                    "source_url": None, "first_seen_url": None,
                    "fetched_at": _now(), "http_status": None,
                    "etag_or_last_modified": None,
                    "fetcher_version": "manual-ingest/0.3",
                    "page_count": record.get("pages", 0), "checksum": checksum,
                    "extraction_status": extraction_status,
                    "extraction_config_hash": "cfg-0",
                    "extraction_run_id": "manual-ingest",
                    "candidate_id": f"manual-{code}",
                    "ingestion_registry_version": "v0",
                    "terms_review_ref": None, "source": "manual",
                    "variant": variant, "version": version,
                    "supersedes": supersedes or None,
                    "other_report": other_report, "core_slice": core,
                    "doc_type_confidence": confidence,
                    "gate_decision": record["decision"],
                    "fired_rules": record.get("fired_rules", []),
                }
                _append_ledger_record(ledger_path, "document", data)
                if supersedes:
                    _append_ledger_record(ledger_path, "revision", {
                        "from_document_id": supersedes,
                        "to_document_id": checksum,
                        "effective_from_freeze": "manual-ingest",
                        "relationship_reason":
                            "new hash for same company+period",
                        "effective_status": "superseded",
                        "recorded_at": _now(),
                    })
                added += 1
                if record["decision"] == "keep":
                    kept += 1
                elif record["decision"] == "unreadable":
                    unreadable += 1
                else:
                    excluded += 1
                existing.add(checksum)
                key_index.setdefault(key, []).append({
                    "document_id": checksum, "checksum": checksum,
                    "version": version, "company_id": code,
                    "fiscal_year": period, "doc_type": doc_type,
                    "variant": variant,
                })
                update_coverage_matrix(
                    matrix_path, code, doc_type, period, variant, core)

        # Checkpoint after each company: a killed run loses at most this
        # company's cache entries; the ledger already has its finished rows.
        _persist_cache()

    # Withdrawal: a previously ingested manual document whose file no longer
    # exists in the inbox. The raw PDF is kept; only status changes.
    # A filtered run must never withdraw documents from companies it did not
    # scan (their absence here is expected).
    selected_codes = ({p.name for p in company_dirs}
                      if (args.company or args.batch is not None) else None)
    for doc in ledger_docs:
        if doc.get("source") != "manual":
            continue
        if selected_codes is not None and doc.get("company_id") not in selected_codes:
            continue
        document_id = doc.get("document_id")
        if not document_id or document_id in withdrawn_ids:
            continue
        if doc.get("checksum") in current_checksums:
            continue
        reason = "file removed from inbox"
        withdrawn_rows.append({
            "filename": "", "folder": doc.get("company_id") or "",
            "document_id": document_id, "reason": reason,
        })
        if args.apply:
            _append_ledger_record(ledger_path, "withdrawn", {
                "document_id": document_id,
                "company_id": doc.get("company_id"),
                "checksum": doc.get("checksum"),
                "reason": reason,
                "raw_kept": True,
                "recorded_at": _now(),
            })
            withdrawn_ids.add(document_id)

    _persist_cache()

    # write reports
    inbox.mkdir(parents=True, exist_ok=True)
    _write_csv(inbox / "_dry_run.csv", dry_rows, [
        "company", "file", "sha256", "period", "doc_type",
        "doc_type_confidence", "period_confidence", "confidence",
        "variant", "version", "other_report",
        "duplicate", "core_slice", "decision", "problem", "supersedes",
        "snippet", "period_snippet", "matched_snippet", "audit_status",
        "dates_found", "covers_full_year"])
    _write_csv(inbox / "_unparsed.csv", unparsed_rows,
               ["filename", "folder", "reason", "suggested_period",
                "snippet", "period_snippet", "matched_snippet",
                "dates_found", "dates", "candidates"])
    _write_csv(inbox / "_quarantine.csv", quarantine_rows,
               ["filename", "folder", "reason", "score", "snippet"])
    _write_csv(inbox / "_withdrawn.csv", withdrawn_rows,
               ["filename", "folder", "document_id", "reason"])

    print(f"manual_ingest: dry_run={not args.apply} added={added} kept={kept} "
          f"excluded={excluded} unreadable={unreadable} duplicates={dupes} "
          f"skipped={skipped} reused={reused}")
    print("decisions:", dict(decisions))
    if withdrawn_rows:
        print(f"withdrawn (raw kept): {len(withdrawn_rows)}")
    if not args.apply:
        print("dry-run: no writes to raw or ledger (hash cache + reports only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
