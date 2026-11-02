"""PDF ingestion and corpus freeze — data models.

Implements lld_ingestion_pdf.txt (sections 5, 6): DocumentLedgerRow,
ExclusionRecord, PageExtract, FreezeManifest, FetchAttempt, candidate
lifecycle, coverage matrix. Every model here mirrors the LLD's contract
so the ingestion pipeline, QA gate and freeze manager share one schema.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Closed vocabularies (LLD 5.3)
# ---------------------------------------------------------------------------

class DocType(StrEnum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    DISCLOSURE = "disclosure"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    QA_FAILED = "qa_failed"


class ExclusionReason(StrEnum):
    FETCH_FAILED = "fetch_failed"
    EXTRACT_TIMEOUT = "extract_timeout"
    ENCRYPTED_PDF = "encrypted_pdf"
    HTML_ONLY = "html_only"
    SCANNED_PDF = "scanned_pdf"
    CORRUPT_PDF = "corrupt_pdf"
    NOT_A_FINANCIAL_REPORT = "not_a_financial_report"
    DUPLICATE_OF_DOCUMENT = "duplicate_of_document"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    NON_INR_CURRENCY = "non_inr_currency"
    OUT_OF_WINDOW = "out_of_window"
    OUT_OF_COMPANY_SCOPE = "out_of_company_scope"
    OUT_OF_PERIOD_SCOPE = "out_of_period_scope"
    UNSUPPORTED_DOCUMENT_TYPE = "unsupported_document_type"
    TAKEDOWN = "takedown"


class AuditStatus(StrEnum):
    AUDITED = "audited"
    UNAUDITED = "unaudited"
    REVIEWED = "reviewed"
    UNCLEAR = "unclear"


class CandidateState(StrEnum):
    DISCOVERED = "discovered"
    FETCH_PENDING = "fetch_pending"
    FETCH_FAILED = "fetch_failed"
    FETCHED = "fetched"
    VALIDATED = "validated"
    EXCLUDED = "excluded"
    EXTRACTION_PENDING = "extraction_pending"
    EXTRACTION_FAILED = "extraction_failed"
    EXTRACTED = "extracted"
    QA_FAILED = "qa_failed"
    QA_PASSED = "qa_passed"
    FROZEN = "frozen"


class CoverageStatus(StrEnum):
    PRESENT_USABLE = "present_usable"
    PRESENT_NOT_USABLE = "present_not_usable"
    PRESENT_NOT_EXTRACTED = "present_not_extracted"
    NOT_PRESENT = "not_present"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"


class PageType(StrEnum):
    STATEMENT = "statement"
    PROSE = "prose"
    TABLE_HEAVY = "table-heavy"
    SCAN = "scan"


class EffectiveStatus(StrEnum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    UNDER_REVIEW = "under_review"


# ---------------------------------------------------------------------------
# Candidate / fetch attempt (LLD 5.6 FetchAttempt, candidate lifecycle 6)
# ---------------------------------------------------------------------------

class FetchAttempt(BaseModel):
    """One HTTP attempt, separate from documents (LLD 5.6)."""
    fetch_attempt_id: str
    candidate_id: str
    source_url: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    http_status: int | None = None
    content_length: int | None = None
    downloaded_bytes: int = 0
    attempt_number: int = 1
    error_code: str | None = None
    retryable: bool = False


class Candidate(BaseModel):
    """A discovered filing candidate (not yet a document)."""
    candidate_id: str
    company_id: str
    fiscal_year: str
    reporting_period: str
    doc_type: DocType
    source_url: str | None = None
    first_seen_url: str | None = None
    announcement_id: str | None = None
    state: CandidateState = CandidateState.DISCOVERED
    fetch_attempts: list[FetchAttempt] = Field(default_factory=list)
    document_id: str | None = None
    exclusion_reason: ExclusionReason | None = None


# ---------------------------------------------------------------------------
# Document ledger (LLD 5.1)
# ---------------------------------------------------------------------------

class DocumentLedgerRow(BaseModel):
    """One row per accepted document (LLD 5.1)."""
    document_id: str  # sha256 of file bytes
    filing_id: str  # logical identity (company, fiscal year, period, doc_type + announcement id)
    doc_type: DocType
    company_id: str
    fiscal_year: str
    reporting_period: str
    period_type: str = "annual"  # annual | quarterly
    period_dates: dict[str, str] | None = None  # start/end as printed
    revision_date: datetime | None = None
    publication_date: datetime
    audit_status: AuditStatus = AuditStatus.UNCLEAR
    currency: str = "INR"
    source_url: str | None = None
    first_seen_url: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    http_status: int | None = None
    etag_or_last_modified: str | None = None
    fetcher_version: str = "0.1.0"
    page_count: int
    checksum: str  # sha256 of file bytes
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    extraction_config_hash: str
    extraction_run_id: str | None = None
    candidate_id: str | None = None
    ingestion_registry_version: str = "v0"
    terms_review_ref: str | None = None

    @staticmethod
    def make_document_id(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def make_filing_id(company_id: str, fiscal_year: str, reporting_period: str,
                       doc_type: str, announcement_id: str | None = None) -> str:
        payload = (
            f"{company_id}|{fiscal_year}|{reporting_period}|"
            f"{doc_type}|{announcement_id or ''}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExclusionRecord(BaseModel):
    """One row per excluded candidate (LLD 5.3)."""
    candidate_id: str
    reason: ExclusionReason
    measurement: str | None = None
    excluded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LedgerRevisionRecord(BaseModel):
    """Append-only sidecar linking superseded -> effective-latest (LLD 5.6)."""
    from_document_id: str
    to_document_id: str
    effective_from_freeze: str
    relationship_reason: str  # amendment notice | restatement transmission | ...
    effective_status: EffectiveStatus = EffectiveStatus.SUPERSEDED
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetiredRecord(BaseModel):
    """Takedown tombstone (LLD 5.6)."""
    document_id: str
    retired_from_freeze: str
    reason: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Page extraction (LLD 5.4)
# ---------------------------------------------------------------------------

class PageExtract(BaseModel):
    """The frozen unit the gold set is authored against (LLD 5.4)."""
    document_id: str
    extraction_run_id: str
    pdf_page_index: int  # 1-based physical
    printed_page_label: str | None = None
    page_text: str
    extraction_hash: str  # sha256 of page_text (normalised)
    page_type: PageType = PageType.PROSE
    has_table: bool = False
    character_count: int = 0
    extraction_status: ExtractionStatus = ExtractionStatus.EXTRACTED


# ---------------------------------------------------------------------------
# Freeze manifest (LLD 5.5)
# ---------------------------------------------------------------------------

class FreezeManifest(BaseModel):
    """Published freeze manifest (LLD 5.5)."""
    freeze_id: str
    freeze_tag: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parent_freeze_id: str | None = None
    ledger_snapshot_checksum: str
    page_extract_set_hash: str
    table_record_set_hash: str
    extraction_config_hash: str
    checksum_registry: dict[str, str] = Field(default_factory=dict)  # doc_id -> checksum
    qa_report_checksum: str | None = None
    qa_passed: bool = False
    freeze_gate_passed: bool = False
    coverage_report_checksum: str | None = None
    document_count: int = 0
    page_count: int = 0
    run_id: str
    pipeline_git_commit: str | None = None
    config_hash: str
    manifest_checksum: str | None = None  # sha256 of canonical serialization
    status: str = "pending"  # pending | published | published_previous

    def canonical_bytes(self) -> bytes:
        """Canonical serialization: sort by stable ids, sha256.

        Operational lifecycle fields (status) and the checksum itself are
        excluded so publishing/demotion does not invalidate the content hash.
        """
        payload = self.model_dump(mode="json")
        payload.pop("manifest_checksum", None)
        payload.pop("status", None)
        # Sort keys and use compact separators for determinism
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return canonical.encode("utf-8")

    def compute_manifest_checksum(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Coverage matrix (LLD 4 / entity registry G4)
# ---------------------------------------------------------------------------

class CompanyPeriodCoverage(BaseModel):
    """Materialized coverage matrix row (entity registry LLD section 4)."""
    freeze_id: str
    company_id: str
    fiscal_period_id: str | None = None
    document_type: str
    eligible_document_count: int = 0
    accepted_document_count: int = 0
    usable_document_count: int = 0
    coverage_status: CoverageStatus = CoverageStatus.UNKNOWN
    evidence_document_ids: list[str] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    registry_version: str = "v0"
    ingestion_registry_version: str = "v0"


# ---------------------------------------------------------------------------
# Terms review (LLD 6)
# ---------------------------------------------------------------------------

class TermsReview(BaseModel):
    """Per-source-domain terms/robots review record (LLD 6)."""
    domain: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    permitted: bool
    notes: str = ""
