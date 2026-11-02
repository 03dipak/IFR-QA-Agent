"""Citation evaluation hooks.

Implements lld_citation_labels.txt section 5/9: precision, coverage,
faithfulness definitions. Harness stays in the eval LLD.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CitationMetrics:
    citation_precision: float = 0.0
    citation_coverage: float = 0.0
    claim_faithfulness: float = 0.0
    evidence_resolution_rate: float = 0.0
    verification_pass_rate: float = 0.0
    false_verification_rate: float = 0.0
    abstention_correctness: float = 0.0


def compute_citation_precision(links_supported: int, links_total: int) -> float:
    if links_total == 0:
        return 1.0
    return links_supported / links_total


def compute_citation_coverage(with_evidence: int, total: int) -> float:
    if total == 0:
        return 1.0
    return with_evidence / total
