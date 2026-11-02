"""Page-level PDF extraction (native-digital only, no OCR).

Implements lld_ingestion_pdf.txt 5.4 + 6: PageExtract, native-digital
definition (per-page text density + garble detection), extraction hash,
table detection flag, character count. Scanned pages are detected per page;
a scanned STATEMENT page excludes the whole document.
"""

from __future__ import annotations

import re
import unicodedata

from pypdf import PdfReader

from .models import ExtractionStatus, PageExtract, PageType

# Thresholds (calibrated at pilot; these are starting proposals)
TEXT_DENSITY_MIN_CHARS = 40  # per page
GARBLED_GLYPH_RATIO = 0.30   # fraction of replacement/unknown glyphs


def _normalize_text(text: str) -> str:
    """Light normalization for the frozen page text. Rules are pinned in
    extraction_config_hash (unicode normalization, whitespace, etc.)."""
    text = unicodedata.normalize("NFC", text)
    # collapse runs of whitespace but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_garbled(text: str) -> bool:
    """Detect garbled font encodings (replacement glyphs, empty font maps)."""
    if not text:
        return True
    bad = sum(1 for ch in text if ch in "\ufffd" or unicodedata.category(ch) == "Cn")
    return (bad / max(len(text), 1)) > GARBLED_GLYPH_RATIO


def _page_type(text: str, has_table: bool) -> PageType:
    if len(text.strip()) < TEXT_DENSITY_MIN_CHARS:
        return PageType.SCAN
    if has_table:
        return PageType.TABLE_HEAVY
    # crude statement-page heuristic: contains financial keywords
    low = text.lower()
    if any(k in low for k in ("balance sheet", "profit and loss", "cash flow",
                              "statement of", "revenue", "total assets")):
        return PageType.STATEMENT
    return PageType.PROSE


def _detect_tables(text: str) -> bool:
    """Crude table detection: 3+ lines with multiple whitespace-aligned
    columns. Real detection heuristics are pinned at the freeze gate."""
    lines = text.splitlines()
    tabular = 0
    for line in lines:
        if len(line.split()) >= 3 and "  " in line:
            tabular += 1
    return tabular >= 3


def extract_pages(pdf_bytes: bytes, document_id: str, extraction_run_id: str) -> list[PageExtract]:
    """Extract page-level text from a native-digital PDF.

    Returns one PageExtract per physical page. Raises ValueError if the
    PDF cannot be opened (corrupt/encrypted).
    """
    from io import BytesIO
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise ValueError("encrypted_pdf")
    out: list[PageExtract] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"extract_error: {exc}") from exc
        text = _normalize_text(raw)
        has_table = _detect_tables(text)
        pt = _page_type(text, has_table)
        # printed page label: try metadata; fall back to None
        printed = None
        try:
            printed = page.get("/P")
        except Exception:  # noqa: BLE001
            printed = None
        if printed is not None:
            printed = str(printed)
        extraction_hash = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
        out.append(PageExtract(
            document_id=document_id,
            extraction_run_id=extraction_run_id,
            pdf_page_index=i,
            printed_page_label=printed,
            page_text=text,
            extraction_hash=extraction_hash,
            page_type=pt,
            has_table=has_table,
            character_count=len(text),
            extraction_status=ExtractionStatus.EXTRACTED,
        ))
    return out


def has_scanned_statement_page(pages: list[PageExtract]) -> bool:
    """True if any page fails native-digital text validation.

    The LLD excludes a whole DOCUMENT when a statement page is scanned or its
    text is garbled. Without OCR, an image-only page cannot be proven to be a
    non-statement page, so the document-level gate is deliberately
    conservative: any scan/density/garbled page excludes the document.
    """
    return any(
        page.page_type == PageType.SCAN
        or page.character_count < TEXT_DENSITY_MIN_CHARS
        or _is_garbled(page.page_text)
        for page in pages
    )


def text_density_ok(text: str) -> bool:
    return len(text.strip()) >= TEXT_DENSITY_MIN_CHARS and not _is_garbled(text)
