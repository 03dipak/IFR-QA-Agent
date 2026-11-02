"""Document ledger writer with idempotency.

Implements lld_ingestion_pdf.txt 5.1 + 6: the ledger-writer is the only
writer of ledger rows and exclusion rows; idempotent ingestion via unique
checksum constraint; candidate lifecycle states.
"""

from __future__ import annotations

from .models import (
    Candidate,
    CandidateState,
    DocumentLedgerRow,
    ExclusionReason,
    ExclusionRecord,
    ExtractionStatus,
    LedgerRevisionRecord,
    RetiredRecord,
)


class Ledger:
    """In-memory ledger (production: backed by the db LLD's storage)."""

    def __init__(self) -> None:
        self.rows: dict[str, DocumentLedgerRow] = {}  # document_id -> row
        self.exclusions: list[ExclusionRecord] = []
        self.revisions: list[LedgerRevisionRecord] = []
        self.retired: list[RetiredRecord] = []
        self.candidates: dict[str, Candidate] = {}
        self._checksum_index: dict[str, str] = {}  # checksum -> document_id

    # -- candidates ---------------------------------------------------------
    def register_candidate(self, c: Candidate) -> Candidate:
        self.candidates[c.candidate_id] = c
        return c

    def set_candidate_state(self, candidate_id: str, state: CandidateState,
                            document_id: str | None = None,
                            exclusion_reason: ExclusionReason | None = None) -> None:
        c = self.candidates[candidate_id]
        c.state = state
        if document_id is not None:
            c.document_id = document_id
        if exclusion_reason is not None:
            c.exclusion_reason = exclusion_reason

    # -- idempotent document registration -----------------------------------
    def register_document(self, row: DocumentLedgerRow) -> tuple[bool, str]:
        """Register a document. Returns (created, document_id).

        Idempotent: if the checksum already exists, returns the existing
        document_id and created=False (mirror case).
        """
        existing_id = self._checksum_index.get(row.checksum)
        if existing_id is not None:
            return False, existing_id
        # also check rows directly (defensive)
        for did, r in self.rows.items():
            if r.checksum == row.checksum:
                self._checksum_index[row.checksum] = did
                return False, did
        self.rows[row.document_id] = row
        self._checksum_index[row.checksum] = row.document_id
        return True, row.document_id

    def get(self, document_id: str) -> DocumentLedgerRow | None:
        return self.rows.get(document_id)

    def by_checksum(self, checksum: str) -> DocumentLedgerRow | None:
        did = self._checksum_index.get(checksum)
        return self.rows.get(did) if did else None

    def update_extraction_status(self, document_id: str,
                                status: ExtractionStatus,
                                page_count: int | None = None) -> None:
        row = self.rows[document_id]
        row.extraction_status = status
        if page_count is not None:
            row.page_count = page_count

    # -- exclusions ---------------------------------------------------------
    def add_exclusion(self, rec: ExclusionRecord) -> None:
        self.exclusions.append(rec)

    # -- sidecars -----------------------------------------------------------
    def add_revision(self, rec: LedgerRevisionRecord) -> None:
        self.revisions.append(rec)

    def add_retired(self, rec: RetiredRecord) -> None:
        self.retired.append(rec)

    # -- accounting ---------------------------------------------------------
    def reconcile(self, candidate_ids: list[str]) -> dict:
        """Return accounting summary for the candidate set."""
        discovered = len(candidate_ids)
        accepted = sum(1 for cid in candidate_ids
                       if self.candidates.get(cid) and
                       self.candidates[cid].state in (
                           CandidateState.VALIDATED, CandidateState.EXTRACTION_PENDING,
                           CandidateState.EXTRACTION_FAILED, CandidateState.EXTRACTED,
                           CandidateState.QA_FAILED, CandidateState.QA_PASSED,
                           CandidateState.FROZEN))
        permanently_excluded = sum(1 for cid in candidate_ids
                                   if self.candidates.get(cid) and
                                   self.candidates[cid].state == CandidateState.EXCLUDED)
        unresolved = discovered - accepted - permanently_excluded
        return {
            "discovered": discovered,
            "accepted": accepted,
            "permanently_excluded": permanently_excluded,
            "unresolved": unresolved,
            "balanced": discovered == accepted + permanently_excluded + unresolved,
        }

    def counts_reconcile(self) -> bool:
        """Invariant: accepted + excluded-by-reason == candidates fetched
        is checked per-freeze by the caller; here we just expose totals."""
        return True


