"""Document ledger writer with idempotency and optional JSONL persistence.

Implements lld_ingestion_pdf.txt 5.1 + 6: the ledger-writer is the only
writer of ledger rows and exclusion rows; idempotent ingestion via unique
checksum constraint; candidate lifecycle states.

Persistence (LLD-42) is an append-only JSONL file using only the standard
library. Each line is a versioned record::

    {"schema": 1, "kind": "document", "data": {...}}

Default project-local location is ``data/corpus/ledger.jsonl`` (the CLI wires
this path; tests use explicit tmp_path files). A missing file is an empty
ledger; a corrupt or unsupported record raises ``LedgerError`` rather than
being silently ignored. Replaying the file rebuilds the same document ids and
checksum index, so a second process run converges instead of duplicating rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    CandidateState,
    DocumentLedgerRow,
    ExclusionReason,
    ExclusionRecord,
    ExtractionStatus,
    LedgerRevisionRecord,
    RetiredRecord,
    SupersededRecord,
    WithdrawnRecord,
)

LEDGER_SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = "data/corpus/ledger.jsonl"


class LedgerError(RuntimeError):
    """Raised when a persisted ledger is corrupt or unsupported."""


class Ledger:
    """Ledger with optional append-only JSONL persistence.

    ``Ledger()`` is an in-memory ledger (used by unit tests). ``Ledger(path)``
    loads an existing file and appends every mutation so a fresh process can
    replay it. The ledger-writer remains the only writer of rows.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.rows: dict[str, DocumentLedgerRow] = {}  # document_id -> row
        self.exclusions: list[ExclusionRecord] = []
        self.revisions: list[LedgerRevisionRecord] = []
        self.retired: list[RetiredRecord] = []
        self.withdrawn: list[WithdrawnRecord] = []
        self.superseded: list[SupersededRecord] = []
        self.candidates: dict[str, Candidate] = {}
        self._checksum_index: dict[str, str] = {}  # checksum -> document_id
        if self.path is not None and self.path.exists():
            self._load(self.path)

    # -- persistence --------------------------------------------------------
    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerError(
                        f"{path}: line {line_no}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise LedgerError(f"{path}: line {line_no}: record is not an object")
                if record.get("schema") != LEDGER_SCHEMA_VERSION:
                    raise LedgerError(
                        f"{path}: line {line_no}: unsupported schema version "
                        f"{record.get('schema')!r}"
                    )
                kind = record.get("kind")
                data = record.get("data")
                if not isinstance(data, dict):
                    raise LedgerError(f"{path}: line {line_no}: missing record data")
                self._replay(kind, data, path, line_no)

    def _replay(self, kind: Any, data: dict[str, Any], path: Path,
                line_no: int) -> None:
        try:
            if kind == "candidate":
                candidate = Candidate.model_validate(data)
                self.candidates[candidate.candidate_id] = candidate
                return
            if kind == "document":
                row = DocumentLedgerRow.model_validate(data)
                existing_id = self._checksum_index.get(row.checksum)
                if existing_id is not None and existing_id != row.document_id:
                    raise LedgerError(
                        f"{path}: line {line_no}: checksum {row.checksum} maps to "
                        f"both {existing_id} and {row.document_id}"
                    )
                self.rows[row.document_id] = row
                self._checksum_index[row.checksum] = row.document_id
                return
            if kind == "exclusion":
                self.exclusions.append(ExclusionRecord.model_validate(data))
                return
            if kind == "revision":
                self.revisions.append(LedgerRevisionRecord.model_validate(data))
                return
            if kind == "retired":
                self.retired.append(RetiredRecord.model_validate(data))
                return
            if kind == "withdrawn":
                self.withdrawn.append(WithdrawnRecord.model_validate(data))
                return
            if kind == "superseded":
                self.superseded.append(SupersededRecord.model_validate(data))
                return
        except LedgerError:
            raise
        except (ValueError, TypeError) as exc:
            raise LedgerError(
                f"{path}: line {line_no}: invalid {kind!r} record: {exc}"
            ) from exc
        raise LedgerError(f"{path}: line {line_no}: unknown record kind {kind!r}")

    def _append(self, kind: str, data: dict[str, Any]) -> None:
        if self.path is None:
            return
        record = {"schema": LEDGER_SCHEMA_VERSION, "kind": kind, "data": data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

    # -- candidates ---------------------------------------------------------
    def register_candidate(self, c: Candidate) -> Candidate:
        self.candidates[c.candidate_id] = c
        self._append("candidate", c.model_dump(mode="json"))
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
        self._append("candidate", c.model_dump(mode="json"))

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
        self._append("document", row.model_dump(mode="json"))
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
        self._append("document", row.model_dump(mode="json"))

    # -- exclusions ---------------------------------------------------------
    def add_exclusion(self, rec: ExclusionRecord) -> None:
        self.exclusions.append(rec)
        self._append("exclusion", rec.model_dump(mode="json"))

    # -- sidecars -----------------------------------------------------------
    def add_revision(self, rec: LedgerRevisionRecord) -> None:
        self.revisions.append(rec)
        self._append("revision", rec.model_dump(mode="json"))

    def add_retired(self, rec: RetiredRecord) -> None:
        self.retired.append(rec)
        self._append("retired", rec.model_dump(mode="json"))

    def add_withdrawn(self, rec: WithdrawnRecord) -> None:
        self.withdrawn.append(rec)
        self._append("withdrawn", rec.model_dump(mode="json"))

    # -- accounting ---------------------------------------------------------
    ACCEPTED_STATES = frozenset({
        CandidateState.VALIDATED,
        CandidateState.EXTRACTION_PENDING,
        CandidateState.EXTRACTION_FAILED,
        CandidateState.EXTRACTED,
        CandidateState.QA_FAILED,
        CandidateState.QA_PASSED,
        CandidateState.FROZEN,
    })

    def accounting(self, candidate_ids: list[str] | None = None) -> dict:
        """Compute the LLD accounting invariant over candidates.

        ``discovered = accepted + permanently_excluded + unresolved`` where a
        candidate is counted once. A fetched candidate with no document_id is
        a byte-mirror (duplicate of an existing document), reported separately
        as ``mirrors`` and counted as permanently excluded for the invariant.
        A fetch failure with an exclusion record is permanently excluded; an
        exclusion record is the permanent reason code. Retryable/unclassified
        fetch failures without an exclusion remain unresolved.
        """
        if candidate_ids is None:
            ids = list(self.candidates)
        else:
            ids = list(dict.fromkeys(candidate_ids))
        discovered = len(ids)

        by_state: dict[str, int] = {}
        missing_ids: list[str] = []
        accepted_ids: set[str] = set()
        excluded_ids: set[str] = set()
        mirror_ids: set[str] = set()
        for cid in ids:
            candidate = self.candidates.get(cid)
            if candidate is None:
                missing_ids.append(cid)
                continue
            by_state[candidate.state.value] = by_state.get(candidate.state.value, 0) + 1
            if candidate.state in self.ACCEPTED_STATES:
                accepted_ids.add(cid)
            if candidate.state == CandidateState.EXCLUDED:
                excluded_ids.add(cid)
            if (candidate.state == CandidateState.FETCHED
                    and candidate.document_id is None):
                mirror_ids.add(cid)

        known_ids = set(ids)
        exclusions_by_candidate: dict[str, list[ExclusionRecord]] = {}
        for exclusion in self.exclusions:
            exclusions_by_candidate.setdefault(exclusion.candidate_id, []).append(exclusion)
        for cid in exclusions_by_candidate:
            # An exclusion record is historical; a candidate whose final
            # state is accepted (e.g. append-only re-acceptance) supersedes it.
            if cid in known_ids and cid not in accepted_ids:
                excluded_ids.add(cid)

        orphan_ids = sorted(set(exclusions_by_candidate) - set(self.candidates))
        duplicate_exclusion_records = sum(
            len(records) - 1 for cid, records in exclusions_by_candidate.items()
            if cid in known_ids
        )
        excluded_ids |= mirror_ids
        overlap_ids = sorted(accepted_ids & excluded_ids)

        accepted = len(accepted_ids)
        permanently_excluded_ids = excluded_ids - accepted_ids
        permanently_excluded = len(permanently_excluded_ids)
        unresolved_ids = sorted(
            set(ids) - accepted_ids - excluded_ids - set(missing_ids)
        ) + missing_ids
        unresolved = len(unresolved_ids)
        gap = discovered - accepted - permanently_excluded - unresolved

        attempt_rows = [
            attempt
            for candidate in self.candidates.values()
            if candidate.candidate_id in known_ids
            for attempt in candidate.fetch_attempts
        ]
        fetch_attempts = len(attempt_rows)
        successful_fetches = sum(1 for attempt in attempt_rows if attempt.error_code is None)
        failed_attempts = fetch_attempts - successful_fetches
        balanced = (
            gap == 0
            and not overlap_ids
            and fetch_attempts == successful_fetches + failed_attempts
        )
        return {
            "discovered": discovered,
            "accepted": accepted,
            "permanently_excluded": permanently_excluded,
            "unresolved": unresolved,
            "unresolved_candidate_ids": unresolved_ids,
            "mirrors": len(mirror_ids),
            "duplicate_exclusion_records": duplicate_exclusion_records,
            "orphan_exclusion_candidate_ids": orphan_ids,
            "missing_candidate_ids": missing_ids,
            "overlap_candidate_ids": overlap_ids,
            "by_state": dict(sorted(by_state.items())),
            "fetch_attempts": fetch_attempts,
            "successful_fetches": successful_fetches,
            "failed_attempts": failed_attempts,
            "gap": gap,
            "balanced": balanced,
            "ok": (
                balanced
                and unresolved == 0
                and not orphan_ids
                and not duplicate_exclusion_records
            ),
        }

    def reconcile(self, candidate_ids: list[str]) -> dict:
        """Return the legacy accounting summary for the candidate set."""
        accounting = self.accounting(candidate_ids)
        return {
            key: accounting[key]
            for key in (
                "discovered", "accepted", "permanently_excluded",
                "unresolved", "balanced",
            )
        }

    def counts_reconcile(self, candidate_ids: list[str] | None = None) -> bool:
        """True only when the invariant holds and no candidate is unresolved.

        The freeze gate requires zero unresolved candidates; an unbalanced,
        orphaned, missing or double-counted candidate is also a failure.
        """
        return bool(self.accounting(candidate_ids)["ok"])
