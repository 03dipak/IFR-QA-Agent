"""Ingestion pipeline orchestrator.

Implements lld_ingestion_pdf.txt: candidate discovery -> fetch -> checksum ->
dedupe -> extract -> QA -> freeze. Ties the components together; the only
writer of ledger rows is the Ledger (idempotency contract).
"""

from __future__ import annotations

from datetime import UTC, datetime

from .checksummer import sha256_bytes
from .dedupe import classify_mirror
from .extractor import extract_pages, has_scanned_statement_page
from .fetcher import Fetcher, FetchError
from .ledger import Ledger
from .models import (
    AuditStatus,
    Candidate,
    CandidateState,
    DocumentLedgerRow,
    ExclusionReason,
    ExclusionRecord,
    ExtractionStatus,
)
from .storage import RawStorage


def process_candidate(candidate: Candidate, fetcher: Fetcher,
                      ledger: Ledger, storage: RawStorage,
                      storage_key_prefix: str = "raw",
                      extractor_version: str = "pymupdf-0.1",
                      extraction_config_hash: str = "cfg-0") -> str:
    """Process one candidate end-to-end. Returns the resulting state name.

    Mutates candidate.state. Writes to ledger on acceptance.
    """
    ledger.register_candidate(candidate)
    candidate.state = CandidateState.FETCH_PENDING

    # Fetch
    try:
        if candidate.source_url is None:
            raise ValueError("candidate has no source_url")
        data, attempt = fetcher.fetch(candidate.source_url, candidate.candidate_id)
    except FetchError as exc:
        candidate.state = CandidateState.FETCH_FAILED
        ledger.set_candidate_state(candidate.candidate_id, CandidateState.FETCH_FAILED)
        ledger.add_exclusion(ExclusionRecord(
            candidate_id=candidate.candidate_id,
            reason=ExclusionReason.FETCH_FAILED,
            measurement=str(exc),
        ))
        return candidate.state.value

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
    storage.put(f"{storage_key_prefix}/{checksum}.pdf", data)

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
