"""Reconcile git-ignored raw PDFs against tracked ledger/extract artefacts.

Recovery section of the db-LLD (§7 failure handling): raw PDFs under
data/corpus/raw are git-ignored while the ledger and extract JSONLs are
tracked, so a fresh clone (or a deleted extract, a copied PDF, a lost
ledger) can leave the three out of step. This CLI detects the drift and
offers explicit, never-silent repairs:

    python -m src.ingestion.reconcile --storage-dir data/corpus \
        [--ledger data/corpus/ledger.jsonl] [--re-extract] [--register-orphans]

Default (no repair flags) is a read-only dry run: one state line per
document, one line per orphan, plus summary counts. Exit 2 if any state is
not OK, 0 otherwise.

States:
    OK                 everything present and consistent
    MISSING_EXTRACT    pages/tables JSONL absent for a ledger document
    MISSING_RAW        no local raw bytes for a ledger document (reported
                       only; repair = re-download with bulk)
    ORPHAN_RAW         raw PDF with no ledger row (--register-orphans fixes,
                       but refuses bytes kept by an excluded candidate)
    CORRUPT_EXTRACT    pages/tables JSONL unreadable/truncated for a ledger
                       document (--re-extract rebuilds)
    CHECKSUM_MISMATCH  raw exists at the canonical name but its content
                       hashes to something else (--re-extract refuses to
                       build from it)
    EXTRACT_ERROR      --re-extract could not build extracts from the intact
                       raw PDF (e.g. pypdf parse limit); the row stays
                       pending/retryable

Writes are atomic (temp file, fsync, rename) and idempotent: running twice
changes nothing the second time. Never writes under raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..tables.extractor import extract_tables
from .dedupe import filing_id
from .extractor import extract_pages
from .ledger import DEFAULT_LEDGER_PATH, Ledger
from .models import CandidateState, DocType, DocumentLedgerRow, ExtractionStatus

STATE_OK = "OK"
STATE_MISSING_EXTRACT = "MISSING_EXTRACT"
STATE_MISSING_RAW = "MISSING_RAW"
STATE_ORPHAN_RAW = "ORPHAN_RAW"
STATE_CORRUPT_EXTRACT = "CORRUPT_EXTRACT"
STATE_CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
STATE_EXTRACT_ERROR = "EXTRACT_ERROR"
ALL_STATES = (STATE_OK, STATE_MISSING_EXTRACT, STATE_MISSING_RAW,
              STATE_ORPHAN_RAW, STATE_CORRUPT_EXTRACT, STATE_CHECKSUM_MISMATCH,
              STATE_EXTRACT_ERROR)


def _excluded_checksums(ledger: Ledger) -> dict[str, str]:
    """checksum -> candidate_id for EXCLUDED candidates (raw bytes kept)."""
    excluded: dict[str, str] = {}
    for candidate_id, candidate in ledger.candidates.items():
        if candidate.state != CandidateState.EXCLUDED:
            continue
        for attempt in candidate.fetch_attempts:
            if attempt.checksum:
                excluded[attempt.checksum] = candidate_id
    return excluded


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_bytes(items: list) -> bytes:
    lines = []
    for item in items:
        payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else (
            asdict(item)  # type: ignore[arg-type]
            if is_dataclass(item) else item)
        lines.append(json.dumps(payload, sort_keys=True, default=str))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _jsonl_valid(path: Path, document_id: str) -> bool:
    try:
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            return False
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if payload.get("document_id") != document_id:
                    return False
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return True


def _extract_states(row: DocumentLedgerRow, extract_dir: Path) -> set[str]:
    states: set[str] = set()
    pages_path = extract_dir / "page_extracts" / f"{row.document_id}.pages.jsonl"
    tables_path = extract_dir / "table_records" / f"{row.document_id}.tables.jsonl"
    for path in (pages_path, tables_path):
        if not path.is_file():
            states.add(STATE_MISSING_EXTRACT)
        elif not _jsonl_valid(path, row.document_id):
            states.add(STATE_CORRUPT_EXTRACT)
    return states


def scan(ledger: Ledger, storage_dir: Path) -> tuple[list[tuple], dict[str, list[str]], list[str]]:
    """Read-only scan. Returns (per-row, raw-files, orphans).

    per-row: (row, non-OK states or {STATE_OK}).
    raw-files: content-hash -> names under raw/. orphans: hashes with no row.
    """
    raw_dir = storage_dir / "raw"
    raw_files: dict[str, list[str]] = {}
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.pdf")):
            raw_files.setdefault(_sha256_file(path), []).append(path.name)

    results: list[tuple] = []
    for row in ledger.rows.values():
        states: set[str] = set()
        canonical = raw_dir / f"{row.checksum}.pdf"
        raw_bytes_present = bool(raw_files.get(row.checksum))
        if not raw_bytes_present:
            states.add(STATE_CHECKSUM_MISMATCH if canonical.is_file()
                       else STATE_MISSING_RAW)
        states |= _extract_states(row, storage_dir / "extract")
        results.append((row, states or {STATE_OK}))

    orphans = [h for h in raw_files if ledger.by_checksum(h) is None]
    return results, raw_files, orphans


def _write_atomic_if_changed(path: Path, data: bytes) -> bool:
    """Atomically write only when content differs. Returns True if written."""
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return True


def re_extract(row: DocumentLedgerRow, storage_dir: Path, ledger: Ledger) -> str:
    """Rebuild missing/corrupt extracts from the intact local raw PDF."""
    raw_path = storage_dir / "raw" / f"{row.checksum}.pdf"
    if not raw_path.is_file() or _sha256_file(raw_path) != row.checksum:
        return STATE_MISSING_RAW
    data = raw_path.read_bytes()
    run_id = str(uuid.uuid4())
    pages = extract_pages(data, document_id=row.document_id, extraction_run_id=run_id)
    tables = extract_tables(pages, document_id=row.document_id,
                            extraction_run_id=run_id,
                            extraction_config_hash=row.extraction_config_hash)
    extract_dir = storage_dir / "extract"
    _write_atomic_if_changed(
        extract_dir / "page_extracts" / f"{row.document_id}.pages.jsonl",
        _model_bytes(pages))
    _write_atomic_if_changed(
        extract_dir / "table_records" / f"{row.document_id}.tables.jsonl",
        _model_bytes(tables))
    if (row.page_count != len(pages)
            or row.extraction_status != ExtractionStatus.EXTRACTED):
        ledger.update_extraction_status(row.document_id, ExtractionStatus.EXTRACTED,
                                        page_count=len(pages))
    return STATE_OK


def register_orphan(storage_dir: Path, ledger: Ledger, content_hash: str,
                    names: list[str]) -> bool:
    """Add a ledger row for an orphan raw PDF (source=local, no invented URL)."""
    raw_dir = storage_dir / "raw"
    canonical = raw_dir / f"{content_hash}.pdf"
    path = canonical if canonical.is_file() else raw_dir / names[0]
    from pypdf import PdfReader
    page_count = len(PdfReader(str(path)).pages)
    row = DocumentLedgerRow(
        document_id=content_hash,
        filing_id=filing_id("local", "unknown", "unknown",
                            DocType.DISCLOSURE.value, None),
        doc_type=DocType.DISCLOSURE,
        company_id="local",
        fiscal_year="unknown",
        reporting_period="unknown",
        publication_date=datetime.now(UTC),
        page_count=page_count,
        checksum=content_hash,
        extraction_status=ExtractionStatus.PENDING,
        extraction_config_hash="local-unextracted",
        source="local",
    )
    created, _ = ledger.register_document(row)
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile ignored raw PDFs against tracked ledger/extracts")
    parser.add_argument("--storage-dir", default="data/corpus",
                        help="corpus dir containing raw/ and extract/ (default data/corpus)")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--re-extract", action="store_true",
                        help="rebuild missing/corrupt extracts from intact local raw PDFs")
    parser.add_argument("--register-orphans", action="store_true",
                        help="add ledger rows for orphan raw PDFs (source=local)")
    parser.add_argument("--allow-excluded-orphans", action="store_true",
                        help="register orphans whose bytes belong to an "
                             "excluded candidate (default: refuse)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage_dir = Path(args.storage_dir)
    ledger = Ledger(args.ledger)
    results, raw_files, orphans = scan(ledger, storage_dir)

    counts: dict[str, int] = {state: 0 for state in ALL_STATES}
    for row, states in results:
        for state in states:
            counts[state] += 1
            print(f"{state:18} {row.company_id:12} {row.checksum[:12]}")
    for content_hash in orphans:
        counts[STATE_ORPHAN_RAW] += 1
        print(f"{STATE_ORPHAN_RAW:18} {'(no row)':12} {content_hash[:12]} "
              f"files={','.join(raw_files[content_hash])}")

    if args.re_extract:
        for row, states in results:
            repairable = states & {STATE_MISSING_EXTRACT, STATE_CORRUPT_EXTRACT}
            blocked = states & {STATE_MISSING_RAW, STATE_CHECKSUM_MISMATCH}
            if repairable and not blocked:
                try:
                    outcome = re_extract(row, storage_dir, ledger)
                except Exception:  # noqa: BLE001
                    print(f"{STATE_EXTRACT_ERROR:18} {row.company_id:12} "
                          f"{row.checksum[:12]}")
                    outcome = STATE_EXTRACT_ERROR
                for state in repairable:
                    counts[state] -= 1
                counts[outcome] += 1
    if args.register_orphans:
        excluded = _excluded_checksums(ledger)
        for content_hash in orphans:
            candidate_id = excluded.get(content_hash)
            if candidate_id is not None and not args.allow_excluded_orphans:
                print(f"CONFLICT           {'(excluded)':12} {content_hash[:12]} "
                      f"excluded as candidate {candidate_id}; refusing to "
                      f"register as kept (use --allow-excluded-orphans to "
                      f"override)")
                continue
            register_orphan(storage_dir, ledger, content_hash,
                            raw_files[content_hash])
            counts[STATE_ORPHAN_RAW] -= 1
            counts[STATE_OK] += 1

    copies_other = sum(1 for h, names in raw_files.items()
                       if ledger.by_checksum(h) is not None
                       and names != [f"{h}.pdf"])
    summary = ", ".join(f"{s}={n}" for s, n in counts.items() if n)
    print(f"summary: {summary} | copies_at_other_names={copies_other}")
    bad = any(counts[s] for s in ALL_STATES if s != STATE_OK)
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
