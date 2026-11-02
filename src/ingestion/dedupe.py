"""Document identity and deduplication.

Implements lld_ingestion_pdf.txt 5.2: two keys, two jobs.
- BYTE identity: content checksum (sha256) — exact same-file mirrors.
- LOGICAL identity (filing_id): (company, fiscal year, period, doc_type) +
  exchange announcement id. Revised/amended filings are different logical
  documents.
Mirrors are detected by byte checksum; inexact mirrors (same content,
different metadata) are flagged for human review, never auto-merged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import DocumentLedgerRow


@dataclass
class MirrorMatch:
    kind: str  # "exact_byte" | "inexact_page_text" | "none"
    existing_document_id: str | None = None


def byte_identity(file_bytes: bytes) -> str:
    import hashlib as _h
    return _h.sha256(file_bytes).hexdigest()


def filing_id(company_id: str, fiscal_year: str, reporting_period: str,
              doc_type: str, announcement_id: str | None = None) -> str:
    return DocumentLedgerRow.make_filing_id(company_id, fiscal_year,
                                            reporting_period, doc_type,
                                            announcement_id)


def classify_mirror(new_bytes: bytes, new_page_text_hash: str | None,
                    existing: list[DocumentLedgerRow],
                    existing_page_text_hashes: dict | None = None) -> MirrorMatch:
    """Classify a newly fetched file against existing ledger rows.

    Returns:
      exact_byte   — same sha256 as an existing row (mirror, do not re-count)
      inexact_page_text — same filing tuple but different bytes; page-text
                          hash matches an existing row's page text (suspected
                          mirror, route to human review)
      none         — genuinely new
    """
    new_checksum = byte_identity(new_bytes)
    for row in existing:
        if row.checksum == new_checksum:
            return MirrorMatch("exact_byte", row.document_id)
    # inexact: same filing tuple? We don't have filing_id here; caller checks.
    # Page-text hash comparison is a heuristic; flag only, never auto-merge.
    if new_page_text_hash and existing_page_text_hashes:
        for doc_id, pth in existing_page_text_hashes.items():
            if pth == new_page_text_hash:
                return MirrorMatch("inexact_page_text", doc_id)
    return MirrorMatch("none", None)
