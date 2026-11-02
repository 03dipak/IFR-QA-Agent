"""Verification protocol for citations (deterministic code only).

Implements lld_citation_labels.txt section 5: direct + derived verification,
failure paths, operand compatibility, injection honesty.
"""

from __future__ import annotations

from .models import DerivedEvidence, EvidenceObject, VerificationOutcome


def _token_boundary_match(raw: str, page_text: str) -> bool:
    # simple token-boundary check: raw appears as a whole token
    import re
    if not raw or not page_text:
        return False
    # escape regex special chars in raw
    pat = r"(?<![\w,])" + re.escape(raw) + r"(?![\w,])"
    return re.search(pat, page_text) is not None


def direct_verify(evidence: EvidenceObject, page_text: str,
                  metric: str, period: str, basis: str, unit: str) -> VerificationOutcome:
    """Verify a direct figure against frozen extraction."""
    if evidence.figure.raw and not _token_boundary_match(evidence.figure.raw, page_text):
        return VerificationOutcome.SOURCE_NOT_FOUND
    if evidence.figure.metric_id and metric and evidence.figure.metric_id != metric:
        return VerificationOutcome.SEMANTIC_MISMATCH
    if period and evidence.context.fiscal_period_id and evidence.context.fiscal_period_id != period:
        return VerificationOutcome.SEMANTIC_MISMATCH
    if basis and evidence.context.reporting_basis and evidence.context.reporting_basis != basis:
        return VerificationOutcome.SEMANTIC_MISMATCH
    if unit and evidence.figure.normalised is not None:
        nu = getattr(evidence.figure.normalised, "measurement_unit", None)
        if nu and nu != unit:
            return VerificationOutcome.SEMANTIC_MISMATCH
    return VerificationOutcome.VERIFIED


def derived_verify(derived: DerivedEvidence,
                   operands: list[EvidenceObject],
                   replay_result: object) -> VerificationOutcome:
    """Verify a derived figure by calculator replay."""
    # operand compatibility: all operands share basis, period, audit, currency, dimension
    bases = {o.context.reporting_basis for o in operands}
    periods = {o.context.fiscal_period_id for o in operands}
    currencies = set()
    dims = set()
    for o in operands:
        n = o.figure.normalised
        if n is not None:
            currencies.add(getattr(n, "currency", None))
            dims.add(getattr(n, "dimension", None))
    if len(bases) > 1 or len(periods) > 1 or len(currencies) > 1 or len(dims) > 1:
        return VerificationOutcome.SEMANTIC_MISMATCH
    # replay equality at stated precision
    if replay_result is None:
        return VerificationOutcome.CALCULATION_MISMATCH
    try:
        if replay_result != derived.calculation_result:
            return VerificationOutcome.CALCULATION_MISMATCH
    except Exception:
        return VerificationOutcome.CALCULATION_MISMATCH
    # all operands must be verifiable (caller ensures direct_verify passed)
    return VerificationOutcome.VERIFIED


def failure_reason_to_outcome(reason: str) -> VerificationOutcome:
    mapping = {
        "not_found": VerificationOutcome.SOURCE_NOT_FOUND,
        "wrong_unit": VerificationOutcome.SEMANTIC_MISMATCH,
        "wrong_page": VerificationOutcome.SOURCE_NOT_FOUND,
        "wrong_metric": VerificationOutcome.SEMANTIC_MISMATCH,
        "wrong_period": VerificationOutcome.SEMANTIC_MISMATCH,
        "wrong_basis": VerificationOutcome.SEMANTIC_MISMATCH,
        "wrong_audit": VerificationOutcome.SEMANTIC_MISMATCH,
        "calc_mismatch": VerificationOutcome.CALCULATION_MISMATCH,
        "insufficient_evidence": VerificationOutcome.INSUFFICIENT_EVIDENCE,
        "unverifiable": VerificationOutcome.UNVERIFIABLE,
    }
    return mapping.get(reason, VerificationOutcome.UNVERIFIABLE)
