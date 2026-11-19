"""Read-only corpus gate report (LLD-45 regression harness).

Usage:
    python -m src.ingestion.gate_report --storage-dir data/corpus/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .extractor import (
    MIN_STATEMENT_LIKE_PAGES,
    is_statement_like,
    scanned_document_rules,
)
from .ledger import DEFAULT_LEDGER_PATH, Ledger, LedgerError
from .models import PageExtract

# Anomaly thresholds are diagnostic only. They never change the gate decision.
ANOMALY_HIGH_IMAGE_SHARE_LIMIT = 0.05  # TBD: native reports should have
# almost no image-only/dominant pages; 5% is a conservative review limit.
ANOMALY_HIGH_UFFFD_PAGE_SHARE = 0.05  # TBD: a page with more than 5% U+FFFD
# is likely a broken font encoding.
ANOMALY_MIN_UFFFD_CHARS = 20

def compute_config_hash() -> str:
    """Derive the gate config hash from threshold constants and the gating
    source code so any logic change invalidates cached results."""
    import inspect

    h = hashlib.sha256()
    h.update(repr(ANOMALY_HIGH_IMAGE_SHARE_LIMIT).encode())
    h.update(repr(ANOMALY_HIGH_UFFFD_PAGE_SHARE).encode())
    h.update(repr(ANOMALY_MIN_UFFFD_CHARS).encode())
    h.update(repr(MIN_STATEMENT_LIKE_PAGES).encode())
    h.update(inspect.getsource(scanned_document_rules).encode())
    h.update(inspect.getsource(is_statement_like).encode())
    return h.hexdigest()[:16]


GATE_CONFIG_HASH = compute_config_hash()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _doc_type_map(ledger_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not ledger_path.exists():
        return result
    try:
        ledger = Ledger(ledger_path)
    except LedgerError:
        return result
    for row in ledger.rows.values():
        result[row.checksum] = row.doc_type.value
    return result


def _ledger_gate_seed(ledger_path: Path) -> dict[str, dict]:
    """Conclusive ingest-time gate decisions (sha256 -> decision/reason) so the
    corpus gate can skip re-analysis for files already decided. The regression
    harness typically passes a ledger with no decisions and is unaffected."""
    result: dict[str, dict] = {}
    if not ledger_path.exists():
        return result
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "document":
                continue
            d = row.get("data") or {}
            checksum = d.get("checksum") or d.get("document_id")
            decision = d.get("gate_decision")
            if checksum and decision in ("keep", "exclude", "unreadable"):
                result[checksum] = {
                    "decision": decision,
                    "reason": d.get("exclusion_reason") or d.get("gate_reason") or "",
                }
    except Exception:
        return result
    return result


_USD_PRESENTATION = re.compile(
    r"\b(?:in USD|USD millions?|US dollars?\s*(?:in\s*)?millions?|"
    r"United States dollars?\s*(?:in\s*)?millions?)\b", re.I)
_IFRS_STATEMENT = re.compile(
    r"\b(?:under IFRS|IFRS\s*(?:compliant|financial statements))\b"
    r"|International Financial Reporting Standards", re.I)
_IND_AS = re.compile(r"\bInd(?:ian)?[ -]?AS\b", re.I)


def _currency_and_basis(pages: list[PageExtract]) -> tuple[str, str]:
    """Best-effort marking from the early report pages (no OCR)."""
    currency = "unknown"
    early = "\n".join(page.page_text for page in pages[:40])
    if "\u20b9" in early or re.search(r"\bINR\b|\bRs\.?", early):
        currency = "INR"
    elif _USD_PRESENTATION.search(early):
        currency = "USD"
    basis = "unknown"
    joined = "\n".join(p.page_text for p in pages[:80])
    if _IFRS_STATEMENT.search(joined):
        basis = "IFRS"
    elif _IND_AS.search(joined):
        basis = "Ind AS"
    elif re.search(r"Indian GAAP|Accounting Standards", joined, re.I):
        basis = "Indian GAAP"
    return currency, basis


def _page_class_counts(pages: list[PageExtract]) -> dict[str, int]:
    counts = Counter(p.page_class.value for p in pages)
    for cls in ("blank", "low_text", "text", "image_only",
                "image_dominant", "garbled"):
        counts.setdefault(cls, 0)
    return {key: counts[key] for key in sorted(counts)}


def _anomaly_flags(
    path: Path,
    pages: list[PageExtract],
    decision: str,
    fired_rules: list[str],
    statement_like_count: int,
    doc_type: str,
    manifest_status: str,
) -> list[str]:
    flags: list[str] = []
    if len(pages) == 0:
        flags.append("zero_pages")
    if decision == "exclude" and statement_like_count >= MIN_STATEMENT_LIKE_PAGES:
        flags.append("excluded_with_statements")
    minimum = MIN_STATEMENT_LIKE_PAGES
    if doc_type == "quarterly":
        minimum = 1
    if doc_type == "disclosure":
        minimum = 0
    if decision == "keep" and minimum > 0 and statement_like_count < minimum:
        flags.append("kept_below_minimum")
    image_pages = [p for p in pages
                   if p.page_class.value in ("image_only", "image_dominant")]
    if pages and (len(image_pages) / len(pages)) > ANOMALY_HIGH_IMAGE_SHARE_LIMIT:
        flags.append("high_image_share")
    for page in pages:
        if page.character_count >= ANOMALY_MIN_UFFFD_CHARS:
            share = page.ufffd_count / page.character_count
            if share > ANOMALY_HIGH_UFFFD_PAGE_SHARE:
                flags.append("high_ufffd_share")
                break
    if manifest_status in ("FAIL", "CHECKSUM_MISMATCH"):
        flags.append("manifest_mismatch")
    return flags


def analyze_pdf(path: Path, doc_type: str, manifest_entry) -> dict:
    record: dict = {
        "filename": path.name,
        "checksum": "",
        "short_checksum": "",
        "doc_type": doc_type,
        "pages": 0,
        "decision": "unreadable",
        "fired_rules": [],
        "page_class_counts": {},
        "statement_like_count": 0,
        "ufffd_share": 0.0,
        "anomalies": [],
        "manifest_status": "UNTRACKED",
        "currency": "unknown",
        "accounting_basis": "unknown",
        "usd_or_ifrs": False,
        "ledger_status": "untracked",
    }
    try:
        checksum = sha256_file(path)
    except OSError:
        record["anomalies"] = ["unreadable_pdf"]
        return record
    record["checksum"] = checksum
    record["short_checksum"] = checksum[:12]
    try:
        from .extractor import extract_pages
        data = path.read_bytes()
        pages = extract_pages(data, "report", "report")
    except Exception:  # noqa: BLE001
        record["anomalies"] = ["unreadable_pdf"]
        record["manifest_status"] = _manifest_status(manifest_entry, record, None)
        return record
    record["pages"] = len(pages)
    currency, basis = _currency_and_basis(pages)
    record["currency"] = currency
    record["accounting_basis"] = basis
    record["usd_or_ifrs"] = currency == "USD" or basis == "IFRS"
    fired = scanned_document_rules(pages, doc_type or None)
    record["fired_rules"] = fired
    record["decision"] = "exclude" if fired else "keep"
    record["page_class_counts"] = _page_class_counts(pages)
    record["statement_like_count"] = sum(
        1 for p in pages if is_statement_like(p.page_text))
    uffd_pages = sum(1 for p in pages if p.ufffd_count > 0)
    record["ufffd_share"] = (uffd_pages / len(pages)) if pages else 0.0
    record["manifest_status"] = _manifest_status(manifest_entry, record, pages)
    record["anomalies"] = _anomaly_flags(
        path, pages, record["decision"], fired, record["statement_like_count"],
        doc_type, record["manifest_status"],
    )
    return record


def _manifest_status(manifest_entry, record: dict, pages) -> str:
    if manifest_entry is None:
        return "UNTRACKED"
    if record.get("checksum") and manifest_entry.get("_checksum_mismatch"):
        return "CHECKSUM_MISMATCH"
    expected_decision = manifest_entry.get("expected_decision")
    expected_rule = manifest_entry.get("expected_rule")
    expected_counts = manifest_entry.get("expected_page_class_counts")
    actual_rule = "none" if not record["fired_rules"] else ",".join(record["fired_rules"])
    if expected_decision != record["decision"]:
        return "FAIL"
    if expected_rule not in (None, actual_rule):
        return "FAIL"
    if expected_counts is not None and expected_counts != record.get("page_class_counts"):
        return "FAIL"
    return "PASS"


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"entries": {}}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "entries" in data:
        return data
    if isinstance(data, dict):
        return {"entries": data}
    return {"entries": {}}


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
        fh.write("\n")




def _analyze_worker(args):
    path_s, doc_type, manifest_entry = args
    return analyze_pdf(Path(path_s), doc_type, manifest_entry)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.ingestion.gate_report",
        description="Read-only corpus gate report.",
    )
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--manifest", default="tests/regression/gate_expectations.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--cache", default="data/corpus/gate_cache.json",
        help="path to the incremental gate cache (default: data/corpus/gate_cache.json)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="re-analyse every PDF instead of reusing cached results",
    )
    args = parser.parse_args(argv)

    import time as _time
    t0 = _time.monotonic()

    storage_dir = Path(args.storage_dir)
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    entries = manifest.get("entries", {})
    doc_types = _doc_type_map(Path(args.ledger))
    cache_path = Path(args.cache)
    cache = {} if args.full else _load_cache(cache_path)
    ledger_seed = {} if args.full else _ledger_gate_seed(Path(args.ledger))
    workers = max(1, (os.cpu_count() or 2) - 1)

    # Detect checksum mismatches by filename before analysis.
    by_filename: dict[str, tuple[str, dict]] = {}
    for checksum, entry in entries.items():
        filename = entry.get("filename")
        if filename:
            by_filename[filename] = (checksum, entry)

    local_files = sorted(
        p for p in storage_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    ) if storage_dir.exists() else []
    local_names = {p.name for p in local_files}

    # Phase 1: classify files as cache hits vs misses.
    # Normal run: checksum = filename (stored as <sha256>.pdf), no content
    # hashing. --full run: compute sha256 and verify it matches filename.
    file_info: list[tuple[Path, object, str, bool, str | None]] = []
    for path in local_files:
        try:
            st = path.stat()
        except OSError:
            continue
        stem = path.name[: -len(path.suffix)]
        checksum = stem
        entry = entries.get(checksum)
        status = None
        if args.full:
            actual = sha256_file(path)
            if actual != checksum:
                checksum = actual
                entry = entries.get(checksum)
                status = "CHECKSUM_MISMATCH"
        else:
            if entry is None and path.name in by_filename:
                # entry matches by filename: only a real content-hash
                # difference is a mismatch (a file named e.g. sample.pdf
                # whose bytes hash to the manifest key is fine)
                manifest_checksum, manifest_entry = by_filename[path.name]
                if sha256_file(path) == manifest_checksum:
                    entry, checksum = manifest_entry, manifest_checksum
                else:
                    entry = dict(manifest_entry)
                    entry["_checksum_mismatch"] = True
                    status = "CHECKSUM_MISMATCH"
            if entry is not None and entry.get("_checksum_mismatch"):
                status = "CHECKSUM_MISMATCH"
        cached = cache.get(checksum)
        seeded = (not args.full and checksum in ledger_seed
                  and cached is None)
        use_cache = (
            not args.full
            and cached is not None
            and cached.get("config_hash") == GATE_CONFIG_HASH
            and cached.get("mtime_ns") == st.st_mtime_ns
            and cached.get("size") == st.st_size
        )
        file_info.append((path, st, checksum, use_cache, seeded, status))

    # Phase 2: cache hits, then ledger-seeded, then true misses (pooled).
    records: list[dict] = []
    statuses: list[str] = []
    reused = 0
    reanalysed = 0

    def _emit(path: Path, st, checksum: str, record: dict, status: str | None) -> None:
        records.append(record)
        statuses.append(record["manifest_status"])
        rules = ",".join(record["fired_rules"]) if record["fired_rules"] else "none"
        anomaly_text = (",".join(record["anomalies"])
                        if record["anomalies"] else "none")
        counts = ",".join(f"{k}:{v}" for k, v in record["page_class_counts"].items())
        print(
            f"{record['short_checksum']} {record['doc_type']} pages={record['pages']} "
            f"decision={record['decision']} rules={rules} classes={{{counts}}} "
            f"statement_like={record['statement_like_count']} "
            f"ufffd_share={record['ufffd_share']:.4f} anomalies={anomaly_text} "
            f"manifest={record['manifest_status']} "
            f"ledger={record['ledger_status']} "
            f"currency={record['currency']} basis={record['accounting_basis']} "
            f"usd_or_ifrs={record['usd_or_ifrs']}"
        )

    # (a) cache hits
    for path, st, checksum, use_cache, seeded, status in file_info:
        if not use_cache:
            continue
        cached = cache[checksum]
        record = dict(cached["result"])
        record["filename"] = path.name
        record["ledger_status"] = (
            "matched" if checksum in doc_types else "untracked")
        if status is not None:
            record["manifest_status"] = status
        reused += 1
        _emit(path, st, checksum, record, status)

    # (b) ledger-seeded: ingest-time decisions, no re-analysis
    seeded_n = 0
    for path, st, checksum, use_cache, seeded, status in file_info:
        if use_cache or not seeded:
            continue
        seed = ledger_seed[checksum]
        entry = entries.get(checksum)
        record = {
            "filename": path.name, "checksum": checksum,
            "short_checksum": checksum[:12],
            "doc_type": doc_types.get(checksum, "unknown"),
            "pages": 0, "decision": seed["decision"],
            "fired_rules": [seed["reason"]] if seed["reason"] and seed["decision"] != "keep" else [],
            "page_class_counts": {}, "statement_like_count": 0,
            "ufffd_share": 0.0, "anomalies": [],
            "manifest_status": _manifest_status(entry, {"decision": seed["decision"], "fired_rules": [], "checksum": checksum, "page_class_counts": {}}, None) if entry else "UNTRACKED",
            "currency": "unknown", "accounting_basis": "unknown",
            "usd_or_ifrs": False,
            "ledger_status": "matched" if checksum in doc_types else "untracked",
        }
        if status is not None:
            record["manifest_status"] = status
        cache[checksum] = {
            "config_hash": GATE_CONFIG_HASH,
            "mtime_ns": st.st_mtime_ns, "size": st.st_size,
            "result": record,
        }
        seeded_n += 1
        reused += 1
        _emit(path, st, checksum, record, status)

    # (c) true misses: analyzed, pooled, cache flushed every 25
    misses = [(path, st, checksum, status) for path, st, checksum, use_cache, seeded, status in file_info
              if not use_cache and not seeded]
    total_miss = len(misses)
    since_flush = 0
    if total_miss:
        print(f"analyzing {total_miss} files (seeded={seeded_n} cache_hit={reused - seeded_n} workers={workers})")
    work = [(str(path), doc_types.get(checksum, "unknown"), entries.get(checksum)) for path, st, checksum, status in misses]
    miss_meta = {str(path): (st, checksum, status) for path, st, checksum, status in misses}
    if workers <= 1 or total_miss <= 1:
        for args_i, (path, st, checksum, status) in zip(work, misses):
            record = analyze_pdf(path, args_i[1], args_i[2])
            record["ledger_status"] = "matched" if checksum in doc_types else "untracked"
            if status is not None:
                record["manifest_status"] = status
            cache[checksum] = {"config_hash": GATE_CONFIG_HASH, "mtime_ns": st.st_mtime_ns, "size": st.st_size, "result": record}
            reanalysed += 1; since_flush += 1
            _emit(path, st, checksum, record, status)
            print(f"progress {reanalysed}/{total_miss}", flush=True)
            if since_flush >= 25:
                _save_cache(cache_path, cache); since_flush = 0
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_analyze_worker, a): a[0] for a in work}
            for fut in as_completed(futs):
                path_s = futs[fut]
                st, checksum, status = miss_meta[path_s]
                try:
                    record = fut.result()
                except Exception:
                    record = {"filename": Path(path_s).name, "checksum": checksum, "short_checksum": checksum[:12],
                              "doc_type": doc_types.get(checksum, "unknown"), "pages": 0, "decision": "unreadable",
                              "fired_rules": [], "page_class_counts": {}, "statement_like_count": 0,
                              "ufffd_share": 0.0, "anomalies": ["unreadable_pdf"], "manifest_status": "UNTRACKED",
                              "currency": "unknown", "accounting_basis": "unknown", "usd_or_ifrs": False, "ledger_status": "untracked"}
                record["ledger_status"] = "matched" if checksum in doc_types else "untracked"
                if status is not None:
                    record["manifest_status"] = status
                cache[checksum] = {"config_hash": GATE_CONFIG_HASH, "mtime_ns": st.st_mtime_ns, "size": st.st_size, "result": record}
                reanalysed += 1; since_flush += 1
                _emit(Path(path_s), st, checksum, record, status)
                print(f"progress {reanalysed}/{total_miss}", flush=True)
                if since_flush >= 25:
                    _save_cache(cache_path, cache); since_flush = 0

    # Missing manifest entries.
    skipped = 0
    for checksum, entry in entries.items():
        filename = entry.get("filename")
        if filename and filename not in local_names:
            skipped += 1
            print(f"SKIPPED {filename} (missing locally)")

    kept = sum(1 for r in records if r["decision"] == "keep")
    excluded = sum(1 for r in records if r["decision"] == "exclude")
    unreadable = sum(1 for r in records if r["decision"] == "unreadable")
    anomaly_count = sum(1 for r in records if r["anomalies"])
    pass_count = sum(1 for s in statuses if s == "PASS")
    fail_count = sum(1 for s in statuses if s == "FAIL")
    checksum_mismatch_count = sum(1 for s in statuses if s == "CHECKSUM_MISMATCH")
    untracked = sum(1 for s in statuses if s == "UNTRACKED")
    usd_ifrs = sum(1 for r in records if r["usd_or_ifrs"])
    ledger_untracked = sum(1 for r in records if r["ledger_status"] == "untracked")
    t1 = _time.monotonic()
    print(
        f"summary documents={len(records)} kept={kept} excluded={excluded} "
        f"unreadable={unreadable} anomalies={anomaly_count} skipped={skipped} "
        f"pass={pass_count} fail={fail_count} "
        f"checksum_mismatch={checksum_mismatch_count} untracked={untracked} "
        f"usd_or_ifrs={usd_ifrs} ledger_untracked={ledger_untracked}"
    )
    print(
        f"timing total={t1-t0:.2f}s reused={reused} reanalysed={reanalysed} seeded={seeded_n} "
        f"files={len(records)} workers={workers}"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": records,
            "summary": {
                "documents": len(records),
                "kept": kept,
                "excluded": excluded,
                "unreadable": unreadable,
                "anomalies": anomaly_count,
                "skipped": skipped,
                "pass": pass_count,
                "fail": fail_count,
                "checksum_mismatch": checksum_mismatch_count,
                "untracked": untracked,
                "usd_or_ifrs": usd_ifrs,
                "ledger_untracked": ledger_untracked,
            },
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    _save_cache(cache_path, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
