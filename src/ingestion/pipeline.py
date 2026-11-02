"""Ingestion pipeline orchestrator.

Implements lld_ingestion_pdf.txt: candidate discovery -> fetch -> checksum ->
dedupe -> extract -> QA -> freeze. Ties the components together; the only
writer of ledger rows is the Ledger (idempotency contract).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .checksummer import sha256_bytes
from .dedupe import classify_mirror
from .extractor import extract_pages, has_scanned_statement_page
from .fetcher import Fetcher, FetchError
from .ledger import DEFAULT_LEDGER_PATH, Ledger, LedgerError
from .models import (
    AuditStatus,
    Candidate,
    CandidateState,
    DocType,
    DocumentLedgerRow,
    ExclusionReason,
    ExclusionRecord,
    ExtractionStatus,
)
from .storage import LocalRawStorage, RawStorage


def _fetch_failure_reason(code: str) -> ExclusionReason:
    """Map a distinct fetcher failure code to an exclusion reason.

    HTML error pages are a document exclusion; corrupt/truncated bodies are
    broken downloads; everything else remains a fetch failure.
    """
    if code == "html_only":
        return ExclusionReason.HTML_ONLY
    if code in {
        "corrupt_pdf",
        "truncated",
        "invalid_pdf_header",
        "content_length_mismatch",
        "file_too_large",
    }:
        return ExclusionReason.CORRUPT_PDF
    return ExclusionReason.FETCH_FAILED


def _storage_key(checksum: str, prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{checksum}.pdf" if prefix else f"{checksum}.pdf"


def process_candidate(candidate: Candidate, fetcher: Fetcher,
                      ledger: Ledger, storage: RawStorage,
                      storage_key_prefix: str = "",
                      extractor_version: str = "pymupdf-0.1",
                      extraction_config_hash: str = "cfg-0") -> str:
    """Process one candidate end-to-end. Returns the resulting state name.

    Mutates candidate.state. Writes to ledger on acceptance.
    """
    ledger.register_candidate(candidate)
    candidate.state = CandidateState.FETCH_PENDING
    ledger.set_candidate_state(candidate.candidate_id, CandidateState.FETCH_PENDING)

    # Fetch
    try:
        if candidate.source_url is None:
            raise ValueError("candidate has no source_url")
        data, attempt = fetcher.fetch(candidate.source_url, candidate.candidate_id)
    except FetchError as exc:
        candidate.fetch_attempts.extend(exc.attempts)
        candidate.state = CandidateState.FETCH_FAILED
        ledger.set_candidate_state(candidate.candidate_id, CandidateState.FETCH_FAILED)
        ledger.add_exclusion(ExclusionRecord(
            candidate_id=candidate.candidate_id,
            reason=_fetch_failure_reason(exc.code),
            measurement=str(exc),
        ))
        return candidate.state.value

    candidate.fetch_attempts.append(attempt)

    # Checksum
    checksum = sha256_bytes(data)
    # Mirror classification against existing ledger rows
    existing_rows = list(ledger.rows.values())
    match = classify_mirror(data, None, existing_rows)
    if match.kind == "exact_byte":
        candidate.state = CandidateState.FETCHED
        ledger.set_candidate_state(candidate.candidate_id, CandidateState.FETCHED)
        # mirror: do not create a new document; record and stop
        return "mirror"

    # Size / header validation already done in fetcher; store raw
    storage.put(_storage_key(checksum, storage_key_prefix), data)

    # Native-digital detection happens after extraction; first extract
    document_id = DocumentLedgerRow.make_document_id(data)
    filing_id = DocumentLedgerRow.make_filing_id(
        candidate.company_id, candidate.fiscal_year, candidate.reporting_period,
        candidate.doc_type.value, None)

    # Extract
    import uuid as _u
    run_id = str(_u.uuid4())
    try:
        pages = extract_pages(data, document_id, run_id)
    except ValueError as exc:
        reason = (ExclusionReason.ENCRYPTED_PDF if "encrypted" in str(exc)
                  else ExclusionReason.CORRUPT_PDF)
        candidate.state = CandidateState.EXTRACTION_FAILED
        ledger.set_candidate_state(candidate.candidate_id, CandidateState.EXTRACTION_FAILED)
        ledger.add_exclusion(ExclusionRecord(
            candidate_id=candidate.candidate_id, reason=reason, measurement=str(exc)))
        return candidate.state.value

    # Scanned statement page -> exclude whole document
    if has_scanned_statement_page(pages):
        candidate.state = CandidateState.EXCLUDED
        ledger.set_candidate_state(candidate.candidate_id, CandidateState.EXCLUDED,
                                  exclusion_reason=ExclusionReason.SCANNED_PDF)
        ledger.add_exclusion(ExclusionRecord(
            candidate_id=candidate.candidate_id, reason=ExclusionReason.SCANNED_PDF))
        return candidate.state.value

    # Register document (idempotent)
    row = DocumentLedgerRow(
        document_id=document_id,
        filing_id=filing_id,
        doc_type=candidate.doc_type,
        company_id=candidate.company_id,
        fiscal_year=candidate.fiscal_year,
        reporting_period=candidate.reporting_period,
        publication_date=datetime.now(UTC),
        page_count=len(pages),
        checksum=checksum,
        extraction_status=ExtractionStatus.EXTRACTED,
        extraction_config_hash=extraction_config_hash,
        extraction_run_id=run_id,
        candidate_id=candidate.candidate_id,
        audit_status=AuditStatus.UNCLEAR,
    )
    created, did = ledger.register_document(row)
    if not created:
        # checksum already present (race) — treat as mirror
        candidate.state = CandidateState.FETCHED
        return "mirror"

    candidate.state = CandidateState.EXTRACTED
    ledger.set_candidate_state(candidate.candidate_id, CandidateState.EXTRACTED,
                              document_id=did)
    # store page extracts externally (caller collects); here we just return state
    candidate.document_id = did
    return candidate.state.value


def _candidate_from_url(url: str) -> Candidate:
    """Build a lightweight candidate from a bare source URL (LLD-34).

    URL-driven mode: the caller supplies fetch targets directly; candidate
    identity is derived deterministically so fetch/dedupe/extract bookkeeping
    still runs through the standard candidate lifecycle.
    """
    return Candidate(
        candidate_id=f"url-{sha256_bytes(url.encode('utf-8'))[:16]}",
        company_id="",
        fiscal_year="",
        reporting_period="",
        doc_type=DocType.ANNUAL,
        source_url=url,
    )


REQUIRED_CANDIDATE_COLUMNS = ("url",)


@dataclass
class CandidateLoadResult:
    """Outcome of loading a discovery CSV (skip-and-report policy)."""

    candidates: list[Candidate]
    problems: list[str]
    rows_seen: int = 0

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index]


def candidate_from_discovery_row(row: dict) -> Candidate:
    """Map one discovery CSV row onto the ingestion Candidate (LLD-35 Phase 2)."""
    url = (row.get("url") or "").strip()
    if not url:
        raise ValueError("discovery row without url")
    doc_type_raw = (row.get("document_type") or DocType.ANNUAL.value).strip()
    valid_types = {d.value for d in DocType}
    if doc_type_raw not in valid_types:
        raise ValueError(f"invalid document_type: {doc_type_raw!r}")
    doc_type = DocType(doc_type_raw)
    fiscal_period = (row.get("fiscal_period") or "").strip()
    fiscal_year = fiscal_period or (row.get("filing_date") or "").strip()[:4]
    return Candidate(
        candidate_id=f"disc-{sha256_bytes(url.encode('utf-8'))[:16]}",
        company_id=(row.get("company_id") or "").strip(),
        fiscal_year=fiscal_year,
        reporting_period=doc_type.value,
        doc_type=doc_type,
        source_url=url,
    )


def load_discovery_candidates(path: str) -> CandidateLoadResult:
    """Load discovery CSV output using skip-and-report validation.

    Policy: the whole file is not aborted for one bad row. Rows without a
    non-empty URL, rows whose field count does not match the header, rows with
    an unknown document_type, and malformed CSV records are reported with line
    numbers and skipped; valid rows are still returned. An empty file,
    header-only file, duplicate/missing required header is reported and yields
    no candidates. Callers that treat problems as fatal (the bulk CLI does)
    must inspect ``problems`` and choose a non-zero exit code.
    """
    problems: list[str] = []
    candidates: list[Candidate] = []

    try:
        fh = open(path, newline="", encoding="utf-8")
    except OSError as exc:
        return CandidateLoadResult([], [f"{path}: cannot read: {exc}"])

    with fh:
        reader = csv.reader(fh, strict=True)
        try:
            fieldnames = next(reader)
        except StopIteration:
            return CandidateLoadResult([], [f"{path}: empty file: missing header"])
        except csv.Error as exc:
            return CandidateLoadResult([], [f"{path}: malformed header: {exc}"])

        fieldnames = [name.strip() for name in fieldnames]
        duplicates = sorted({
            name for name in fieldnames
            if name and fieldnames.count(name) > 1
        })
        if duplicates:
            problems.append(
                f"{path}: duplicate header column(s): {', '.join(duplicates)}"
            )
            return CandidateLoadResult([], problems)
        missing = [name for name in REQUIRED_CANDIDATE_COLUMNS if name not in fieldnames]
        if missing:
            problems.append(
                f"{path}: missing required column(s): {', '.join(missing)}"
            )
            return CandidateLoadResult([], problems)

        rows_seen = 0
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                problems.append(
                    f"{path}: line {reader.line_num}: malformed CSV: {exc}"
                )
                break
            if not row or all(not value.strip() for value in row):
                continue
            rows_seen += 1
            line = reader.line_num
            if len(row) != len(fieldnames):
                problems.append(
                    f"{path}: line {line}: expected {len(fieldnames)} columns, "
                    f"got {len(row)}"
                )
                continue
            record = {
                fieldnames[position]: row[position]
                for position in range(len(fieldnames))
                if fieldnames[position]
            }
            url = (record.get("url") or "").strip()
            if not url:
                problems.append(f"{path}: line {line}: missing url")
                continue
            try:
                candidate = candidate_from_discovery_row(record)
            except ValueError as exc:
                problems.append(f"{path}: line {line}: {exc}")
                continue
            candidates.append(candidate)

        if rows_seen == 0:
            problems.append(f"{path}: header-only file: no candidate rows")

    return CandidateLoadResult(candidates, problems, rows_seen)


def _logical_candidate_key(
    candidate: Candidate,
) -> tuple[str, str, str, str, str] | None:
    """Identity tuple plus announcement id, when the id is available.

    Without an announcement id the same tuple may be a revised/amended filing,
    so it is NOT treated as a pre-fetch duplicate; only the URL is deduped.
    """
    announcement_id = (candidate.announcement_id or "").strip()
    if not announcement_id:
        return None
    parts = (
        candidate.company_id.strip(),
        candidate.fiscal_year.strip(),
        candidate.reporting_period.strip(),
        candidate.doc_type.value,
        announcement_id,
    )
    return parts if all(parts) else None


def dedupe_candidates(
    candidates: list[Candidate],
) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Drop repeated targets before any fetch (LLD-43).

    Policy: keep the first occurrence, drop later rows with the same normalized
    URL (``same_url``) or the same available logical identity
    ``(company_id, fiscal_year, reporting_period, doc_type)``
    (``same_logical_filing``). Candidate ids and input order are unchanged for
    kept rows. When identity fields are blank, only URL dedupe applies.
    """
    unique: list[Candidate] = []
    duplicates: list[tuple[Candidate, str]] = []
    by_url: dict[str, Candidate] = {}
    by_identity: dict[tuple[str, str, str, str, str], Candidate] = {}
    for candidate in candidates:
        url = (candidate.source_url or "").strip()
        if url and url in by_url:
            duplicates.append((candidate, "same_url"))
            continue
        identity = _logical_candidate_key(candidate)
        if identity is not None and identity in by_identity:
            duplicates.append((candidate, "same_logical_filing"))
            continue
        unique.append(candidate)
        if url:
            by_url[url] = candidate
        if identity is not None:
            by_identity[identity] = candidate
    return unique, duplicates


EXIT_INTERRUPTED = 130


class BulkInterruptedError(Exception):
    """First SIGINT during bulk ingestion; state remains resumable."""

    def __init__(self, counts: dict[str, int], quarantined: list[str],
                 completed_ids: list[str], remaining_ids: list[str]):
        super().__init__("bulk ingestion interrupted")
        self.counts = dict(counts)
        self.quarantined = list(quarantined)
        self.completed_ids = completed_ids
        self.remaining_ids = remaining_ids


_COMPLETED_STATES = frozenset({
    CandidateState.VALIDATED,
    CandidateState.EXTRACTION_PENDING,
    CandidateState.EXTRACTION_FAILED,
    CandidateState.EXTRACTED,
    CandidateState.QA_FAILED,
    CandidateState.QA_PASSED,
    CandidateState.FROZEN,
    CandidateState.EXCLUDED,
    CandidateState.FETCHED,
})


def _candidate_is_completed(ledger: Ledger, candidate_id: str) -> bool:
    candidate = ledger.candidates.get(candidate_id)
    if candidate is None:
        return False
    if candidate.state in _COMPLETED_STATES:
        return True
    if candidate.state == CandidateState.FETCH_FAILED:
        return any(exclusion.candidate_id == candidate_id
                   for exclusion in ledger.exclusions)
    return False


def run_bulk(
    candidates: list[Candidate],
    fetcher: Fetcher,
    ledger: Ledger,
    storage: RawStorage,
    storage_key_prefix: str = "",
) -> tuple[dict[str, int], list[str]]:
    """Run the full per-candidate lifecycle over a bulk candidate set.

    A ledger containing a FETCH_PENDING candidate marks a previous run as
    interrupted; that run resumes by skipping already-terminal candidates so
    completed work is not re-counted. A normal re-run without a pending
    candidate keeps the mirror-on-rerun contract.
    """
    counts: dict[str, int] = {}
    quarantined: list[str] = []
    unique, duplicates = dedupe_candidates(candidates)
    counts["duplicate"] = len(duplicates)
    resume = any(candidate.state == CandidateState.FETCH_PENDING
                 for candidate in ledger.candidates.values())
    all_ids = [candidate.candidate_id for candidate in unique]
    if resume:
        counts["resumed"] = 0
    try:
        for candidate in unique:
            if resume and _candidate_is_completed(ledger, candidate.candidate_id):
                counts["resumed"] += 1
                continue
            state = process_candidate(
                candidate, fetcher=fetcher, ledger=ledger, storage=storage,
                storage_key_prefix=storage_key_prefix)
            key = state if isinstance(state, str) else state.value
            counts[key] = counts.get(key, 0) + 1
            if key not in ("fetched", "mirror", "extracted"):
                quarantined.append(candidate.candidate_id)
    except KeyboardInterrupt:
        completed = [cid for cid in all_ids
                     if _candidate_is_completed(ledger, cid)]
        remaining = [cid for cid in all_ids if cid not in completed]
        raise BulkInterruptedError(counts, quarantined, completed, remaining) from None
    return counts, quarantined


class _SigintGuard:
    """First SIGINT raises KeyboardInterrupt; second force-exits."""

    def __init__(self) -> None:
        self._count = 0
        self._previous = None

    def __enter__(self):
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._count == 0 and self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
        return False

    def _handle(self, _signum, _frame):
        self._count += 1
        if self._count >= 2:
            os._exit(EXIT_INTERRUPTED)
        raise KeyboardInterrupt


def reconcile_quarantine(ledger: Ledger, quarantined: list[str]) -> dict[str, int]:
    """Cross-check bulk-quarantined candidates against ledger exclusion records."""
    by_cid: dict[str, list[ExclusionRecord]] = {}
    for exclusion in ledger.exclusions:
        by_cid.setdefault(exclusion.candidate_id, []).append(exclusion)
    with_exclusion = sum(1 for cid in quarantined if cid in by_cid)
    return {
        "quarantined_total": len(quarantined),
        "with_exclusion_record": with_exclusion,
        "unmatched_quarantined": len(quarantined) - with_exclusion,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.ingestion.pipeline",
        description="PDF ingestion pipeline: fetch -> checksum -> dedupe -> "
                    "extract -> QA -> freeze.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  run   (default)  process a JSON candidate list (--candidates) "
            "or one/more URLs (--url).\n"
            "  bulk             process a discovery CSV (--candidates); rerun "
            "the same command to resume after Ctrl-C.\n"
            "\n"
            "storage:\n"
            "  --storage-dir ROOT writes every accepted PDF to "
            "ROOT/<sha256>.pdf.\n"
            "  Use --storage-dir data/corpus/raw for the canonical layout "
            "(default: data/raw).\n"
            "\n"
            "exit codes:\n"
            "  0    all requested work completed and accounting reconciled.\n"
            "  2    invalid input, ledger error, or unresolved/accounting gap.\n"
            "  130  interrupted by Ctrl-C; a second Ctrl-C force-exits."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "bulk"),
        help="run (default) processes JSON/URLs; bulk processes a discovery CSV.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Candidates path. run: JSON list of candidate objects. bulk: "
             "discovery CSV with a required 'url' column.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="Source URL to fetch (repeatable; run command only; mutually "
             "exclusive with --candidates).",
    )
    parser.add_argument(
        "--storage-dir",
        default="data/raw",
        help="Raw storage root (default: data/raw). Accepted PDFs are written "
             "to <storage-dir>/<sha256>.pdf; use data/corpus/raw for the "
             "canonical layout.",
    )
    parser.add_argument(
        "--ledger",
        default=DEFAULT_LEDGER_PATH,
        help=f"Append-only ledger JSONL path (default: {DEFAULT_LEDGER_PATH}).",
    )
    return parser


def _load_candidates(path: str) -> list[Candidate]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidates file must contain a JSON list of candidates")
    return [Candidate.model_validate(item) for item in payload]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "bulk":
        if args.url:
            parser.error("bulk does not accept --url; use --candidates CSV")
        if not args.candidates:
            parser.error("bulk requires --candidates PATH")
        loaded = load_discovery_candidates(args.candidates)
        for problem in loaded.problems:
            print(f"candidate CSV: {problem}")
        if not loaded.candidates:
            return 2
        try:
            ledger = Ledger(args.ledger)
        except LedgerError as exc:
            print(f"ledger error: {exc}")
            return 2
        storage = LocalRawStorage(args.storage_dir)
        fetcher = Fetcher()
        with _SigintGuard():
            try:
                counts, quarantined = run_bulk(
                    loaded.candidates, fetcher=fetcher, ledger=ledger,
                    storage=storage)
            except BulkInterruptedError as exc:
                print("interrupted: SIGINT")
                for key, value in sorted(exc.counts.items()):
                    print(f"{key}: {value}")
                print(f"completed: {len(exc.completed_ids)}")
                print(f"remaining: {len(exc.remaining_ids)}")
                print("resume: rerun the same command")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(EXIT_INTERRUPTED)
        for key, value in sorted(counts.items()):
            print(f"{key}: {value}")
        reconciled = reconcile_quarantine(ledger, quarantined)
        for key, value in sorted(reconciled.items()):
            print(f"reconcile {key}: {value}")
        if quarantined:
            print(f"quarantined: {len(quarantined)} {sorted(quarantined)}")

        unique, _duplicates = dedupe_candidates(loaded.candidates)
        accounting = ledger.accounting([c.candidate_id for c in unique])
        print(f"accounting discovered: {accounting['discovered']}")
        print(f"accounting accepted: {accounting['accepted']}")
        print(f"accounting permanently_excluded: {accounting['permanently_excluded']}")
        print(f"accounting unresolved: {accounting['unresolved']}")
        print(f"accounting mirrors: {accounting['mirrors']}")
        print(f"accounting duplicate_exclusion_records: "
              f"{accounting['duplicate_exclusion_records']}")
        print(f"accounting by_state: {accounting['by_state']}")
        print(f"accounting fetch_attempts: {accounting['fetch_attempts']} "
              f"(successful={accounting['successful_fetches']}, "
              f"failed={accounting['failed_attempts']})")
        if accounting["ok"]:
            print("accounting invariant: PASS")
        else:
            print("accounting invariant: FAIL")
            gaps: list[str] = []
            if accounting["gap"]:
                gaps.append(f"unaccounted={accounting['gap']}")
            if accounting["unresolved_candidate_ids"]:
                gaps.append(
                    f"unresolved={accounting['unresolved_candidate_ids']}")
            if accounting["missing_candidate_ids"]:
                gaps.append(f"missing={accounting['missing_candidate_ids']}")
            if accounting["orphan_exclusion_candidate_ids"]:
                gaps.append(
                    f"orphan_exclusions={accounting['orphan_exclusion_candidate_ids']}")
            if accounting["overlap_candidate_ids"]:
                gaps.append(f"overlap={accounting['overlap_candidate_ids']}")
            if accounting["duplicate_exclusion_records"]:
                gaps.append(
                    f"duplicate_exclusions="
                    f"{accounting['duplicate_exclusion_records']}")
            print("accounting gap: " + ("; ".join(gaps) or "unknown"))
        return 2 if loaded.problems or not accounting["ok"] else 0
    if args.command != "run":
        parser.error(f"unknown command: {args.command}")
    urls = args.url or []
    if args.candidates and urls:
        parser.error("--candidates and --url are mutually exclusive")
    if not args.candidates and not urls:
        parser.error("one of --candidates or --url is required")
    if urls:
        candidates = [_candidate_from_url(u) for u in urls]
    else:
        candidates = _load_candidates(args.candidates)
    try:
        ledger = Ledger(args.ledger)
    except LedgerError as exc:
        print(f"ledger error: {exc}")
        return 2
    storage = LocalRawStorage(args.storage_dir)
    fetcher = Fetcher()
    for candidate in candidates:
        state = process_candidate(
            candidate, fetcher=fetcher, ledger=ledger, storage=storage)
        print(f"{candidate.candidate_id}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
