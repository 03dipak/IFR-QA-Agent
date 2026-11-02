"""Citations + EvidenceObject — data models.

Implements lld_citation_labels.txt sections 3-5: EvidenceObject (canonical
verification record), page-label status, claim linkage, derived evidence,
handles. Canonical owner of these types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PageLabelStatus(StrEnum):
    DETECTED = "detected"
    ABSENT = "absent"
    UNREADABLE = "unreadable"
    AMBIGUOUS = "ambiguous"


class EvidenceType(StrEnum):
    PROSE = "prose"
    TABLE_CELL = "table_cell"
    DERIVED = "derived"


class VerificationOutcome(StrEnum):
    VERIFIED = "verified"
    SOURCE_NOT_FOUND = "source_not_found"
    VALUE_MISMATCH = "value_mismatch"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CALCULATION_MISMATCH = "calculation_mismatch"
    UNVERIFIABLE = "unverifiable"
    REQUIRES_HUMAN_REVIEW = "requires_human_review"


class ExecutionStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SupportRole(StrEnum):
    PRIMARY_VALUE = "primary_value"
    COMPARISON_VALUE = "comparison_value"
    NUMERATOR = "numerator"
    DENOMINATOR = "denominator"
    CONTEXT = "context"
    QUALIFICATION = "qualification"


@dataclass
class PageRef:
    pdf_page_index: int
    printed_page_label: str | None = None
    page_label_status: PageLabelStatus = PageLabelStatus.DETECTED
    page_label_source: str = "auto"
    page_label_confidence: float | None = None


@dataclass
class SourceLocator:
    page_extraction_id: str | None = None
    table_extraction_id: str | None = None
    table_id: str | None = None
    row_id: str | None = None
    column_id: str | None = None
    cell_id: str | None = None
    prose_block_id: str | None = None
    text_span_start: int | None = None
    text_span_end: int | None = None
    quoted_text_hash: str | None = None


@dataclass
class Figure:
    metric_id: str
    metric_definition_version: str = "v1"
    source_metric_label: str = ""
    raw: str = ""
    normalised: Any = None  # NormalisedValue


@dataclass
class Context:
    reporting_basis: str = ""
    fiscal_period_id: str | None = None
    period_span: str = ""


@dataclass
class Provenance:
    extraction_run_id: str | None = None
    extraction_hash: str | None = None
    normalization_rules_version: str = "v0"
    metric_definition_version: str = "v1"
    calculator_version: str | None = None
    formula_version: str | None = None
    verification_policy_version: str = "v1"


@dataclass
class EvidenceObject:
    evidence_id: str
    evidence_type: EvidenceType
    corpus_freeze_id: str
    document_id: str
    page: PageRef
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    figure: Figure = field(default_factory=lambda: Figure(metric_id=""))
    context: Context = field(default_factory=Context)
    provenance: Provenance = field(default_factory=Provenance)
    reported_instance_id: str | None = None
    restatement_status: str | None = None
    source_publication_date: str | None = None
    first_seen_freeze_id: str | None = None
    # turn-scoped handle (rendering only)
    turn_id: str | None = None
    handle: int | None = None
    citation_order: int | None = None

    @staticmethod
    def make_evidence_id(document_id: str, pdf_page_index: int,
                         raw: str, locator_key: str = "") -> str:
        payload = f"{document_id}|{pdf_page_index}|{raw}|{locator_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render_citation(self, company: str = "", report: str = "",
                        filed_date: str = "") -> str:
        """Full citation string with document context + dual page labels."""
        parts = []
        if company:
            parts.append(company)
        if report:
            parts.append(report)
        if filed_date:
            parts.append(f"(filed {filed_date})")
        page_part = f"PDF p. {self.page.pdf_page_index}"
        if (self.page.printed_page_label is not None
                and self.page.printed_page_label != str(self.page.pdf_page_index)
                and self.page.page_label_status == PageLabelStatus.DETECTED):
            page_part += f" (printed p. {self.page.printed_page_label})"
        parts.append(page_part)
        # human-readable locator
        loc = self.source_locator
        if loc.table_id and loc.row_id:
            parts.append(f"[{loc.table_id}:{loc.row_id}]")
        return ", ".join(parts)


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    claim_type: str = "factual"
    materiality: str = "material"


@dataclass
class ClaimEvidenceLink:
    claim_id: str
    evidence_id: str
    support_role: SupportRole = SupportRole.PRIMARY_VALUE
    required_for_verification: bool = True


@dataclass
class ClaimVerification:
    claim_id: str
    verification_status: VerificationOutcome
    verification_reason: str = ""
    verifier_version: str = "v1"
    verification_policy_version: str = "v1"
    verified_at: str | None = None


@dataclass
class DerivedEvidence:
    derived_evidence_id: str
    formula_id: str
    formula_version: str
    operand_evidence_ids: list[str] = field(default_factory=list)
    calculation_inputs: list[Any] = field(default_factory=list)
    calculation_result: Any = None
    calculator_version: str = ""
    tolerance_policy_version: str = "v1"


@dataclass
class CitationHandle:
    turn_id: str
    handle: int
    evidence_id: str
    citation_order: int


@dataclass
class HandleTable:
    turn_id: str
    handles: dict[int, str] = field(default_factory=dict)  # handle -> evidence_id

    def mint(self, evidence_id: str) -> int:
        h = len(self.handles) + 1
        self.handles[h] = evidence_id
        return h

    def resolve(self, handle: int) -> str | None:
        return self.handles.get(handle)

    def validate(self, handle: int, turn_id: str) -> bool:
        if turn_id != self.turn_id:
            return False
        return handle in self.handles


@dataclass
class CitationRenderResult:
    text: str
    handles_used: list[int] = field(default_factory=list)
