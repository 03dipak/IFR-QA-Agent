"""Dual citation labels + EvidenceObject.

Implements lld_citation_labels.txt — canonical owner of EvidenceObject and
the citation-label rendering rules. One figure = one EvidenceObject.
"""

from .eval_hooks import compute_citation_coverage, compute_citation_precision
from .models import (
    CitationHandle,
    CitationRenderResult,
    Claim,
    ClaimEvidenceLink,
    ClaimVerification,
    Context,
    DerivedEvidence,
    EvidenceObject,
    EvidenceType,
    ExecutionStatus,
    Figure,
    HandleTable,
    PageLabelStatus,
    PageRef,
    Provenance,
    SourceLocator,
    SupportRole,
    VerificationOutcome,
)
from .verification import derived_verify, direct_verify, failure_reason_to_outcome

__all__ = [
    "CitationHandle", "CitationRenderResult", "Claim", "ClaimEvidenceLink",
    "ClaimVerification", "Context", "DerivedEvidence", "EvidenceObject",
    "EvidenceType", "ExecutionStatus", "Figure", "HandleTable", "PageLabelStatus",
    "PageRef", "Provenance", "SourceLocator", "SupportRole", "VerificationOutcome",
    "direct_verify", "derived_verify", "failure_reason_to_outcome",
    "compute_citation_precision", "compute_citation_coverage",
]
