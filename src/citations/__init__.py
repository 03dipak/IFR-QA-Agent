"""Dual citation labels + EvidenceObject.

Implements lld_citation_labels.txt — canonical owner of EvidenceObject and
the citation-label rendering rules. One figure = one EvidenceObject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvidenceObject:
    """Canonical verification record. Consumed by critic, comparator,
    eval scorer, and UI citations.
    """
    document_id: str
    pdf_page_index: int
    printed_page_label: Optional[str] = None
    table_id: Optional[str] = None
    row: Optional[int] = None
    column: Optional[int] = None
    metric_name: Optional[str] = None
    raw: Optional[str] = None
    unit: Optional[str] = None
    statement_type: Optional[str] = None
    period: Optional[str] = None
    extraction_hash: Optional[str] = None


@dataclass(frozen=True)
class Citation:
    """Rendered citation with dual page labels."""
    text: str
    evidence: EvidenceObject

    def render(self) -> str:
        """Render citation with dual page labels when they differ."""
        ev = self.evidence
        if ev.printed_page_label and ev.printed_page_label != str(ev.pdf_page_index):
            return f"PDF p. {ev.pdf_page_index} (printed p. {ev.printed_page_label})"
        return f"PDF p. {ev.pdf_page_index}"
