"""Page-level PDF extraction (native-digital only, no OCR).

Implements lld_ingestion_pdf.txt 5.4 + 6: PageExtract, native-digital
definition (per-page text density + garble detection), extraction hash,
table detection flag, character count. Scanned pages are detected per page;
a scanned STATEMENT page excludes the whole document.
"""

from __future__ import annotations

import re
import unicodedata

from pypdf import PdfReader, apply_configuration

from .models import ExtractionStatus, PageClass, PageExtract, PageType

# pypdf defaults this parse budget to 2,000,000 tokens per content stream;
# several real annual reports (graphics-heavy pages) exceed it and raise
# LimitReachedError during extract_text(). 5x rescues them without changing
# any other limit.
STREAM_PARSING_WORK_MAXIMUM_COUNT = 10_000_000

# Thresholds (calibrated at pilot; these are starting proposals)
TEXT_DENSITY_MIN_CHARS = 40  # per page
GARBLED_GLYPH_RATIO = 0.30   # fraction of replacement/unknown glyphs

# Document-level gate thresholds (LLD-45). Values are hypotheses until
# calibrated; evidence is recorded in the Phase 0/1 measurements.
STATEMENT_LIKE_MIN_NUMERIC_TOKENS = 5  # >=5 splits statement pages from
# heading-only dividers in the 5 sample PDFs (L&T 273/299, HDFC 282/309,
# ICICI 29/36, Reliance 103/103, Tata 240/241); dividers L&T 382/512 have 0.
MIN_STATEMENT_LIKE_PAGES = 3  # TBD-CALIBRATE: sample minimum is 29, so
# this boundary is not exercised by the 5 real PDFs.
IMAGE_PAGE_SHARE_LIMIT = 0.5  # TBD-UNCALIBRATED: sample maximum is 0.75%%
MIN_STATEMENT_LIKE_PAGES_QUARTERLY = 1  # TBD-CALIBRATE: quarterly results
# usually expose one primary statement/result table; no quarterly fixture exists
# in the 5 sample PDFs, so this is a proposal only.
# (L&T 5/668); no scanned positive exists, so this is a proposal only.
IMAGE_DOMINANT_MAX_CHARS = TEXT_DENSITY_MIN_CHARS


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



_NUMERIC_TOKEN_RE = re.compile(r"\(?-?\d[\d,\.]*\)?")


def numeric_token_count(text: str) -> int:
    """Count numeric tokens, including Indian grouping and bracket negatives."""
    return len(_NUMERIC_TOKEN_RE.findall(text))


_RESULTS_HEADING_RES = tuple(re.compile(p) for p in (
    r"statement of (?:standalone|consolidated) (?:unaudited|audited) financial"
    r" results",
    r"financial results for the quarter ended",
))


def has_statement_heading(text: str) -> bool:
    low = text.lower()
    if any(k in low for k in ("balance sheet", "statement of profit and loss",
                              "cash flow", "independent auditor's report",
                              "financial statements")):
        return True
    return any(rx.search(low) for rx in _RESULTS_HEADING_RES)


def is_statement_like(text: str) -> bool:
    """A page that carries a statement heading with real numeric rows."""
    return (has_statement_heading(text)
            and numeric_token_count(text) >= STATEMENT_LIKE_MIN_NUMERIC_TOKENS)


def _is_large_image(image_count: int, largest_w: int, largest_h: int,
                    page_w: float, page_h: float) -> bool:
    if image_count <= 0 or largest_w <= 0 or largest_h <= 0:
        return False
    if page_w <= 0 or page_h <= 0:
        return False
    return (largest_w * largest_h) >= 0.25 * (page_w * page_h)


def _image_info(page) -> tuple[int, int, int, float, float]:
    count = 0
    largest_w = 0
    largest_h = 0
    largest_area = 0
    try:
        resources = page.get("/Resources")
        if resources is not None and hasattr(resources, "get_object"):
            resources = resources.get_object()
        if isinstance(resources, dict) and "/XObject" in resources:
            xobjects = resources["/XObject"]
            if hasattr(xobjects, "get_object"):
                xobjects = xobjects.get_object()
            if isinstance(xobjects, dict):
                for key in xobjects:
                    obj = xobjects[key]
                    if hasattr(obj, "get_object"):
                        obj = obj.get_object()
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("/Subtype") != "/Image":
                        continue
                    count += 1
                    try:
                        w = int(obj.get("/Width", 0))
                        h = int(obj.get("/Height", 0))
                    except (TypeError, ValueError):
                        w, h = 0, 0
                    if w * h > largest_area:
                        largest_area = w * h
                        largest_w, largest_h = w, h
    except Exception:  # noqa: BLE001
        pass
    try:
        pw = float(page.mediabox.width)
        ph = float(page.mediabox.height)
    except Exception:  # noqa: BLE001
        pw, ph = 0.0, 0.0
    return count, largest_w, largest_h, pw, ph


def classify_page(text: str, image_count: int, largest_w: int, largest_h: int,
                  page_w: float, page_h: float) -> PageClass:
    """Classify a page for the document-level native-PDF gate."""
    if not text.strip():
        if image_count > 0 and _is_large_image(image_count, largest_w, largest_h,
                                               page_w, page_h):
            return PageClass.IMAGE_ONLY
        return PageClass.BLANK
    if _is_garbled(text):
        return PageClass.GARBLED
    if len(text.strip()) < TEXT_DENSITY_MIN_CHARS:
        if image_count > 0 and _is_large_image(image_count, largest_w, largest_h,
                                               page_w, page_h):
            return PageClass.IMAGE_DOMINANT
        return PageClass.LOW_TEXT
    return PageClass.TEXT

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
    with apply_configuration(
            stream_parsing_work_maximum_count=STREAM_PARSING_WORK_MAXIMUM_COUNT):
        for i, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"extract_error: {exc}") from exc
            text = _normalize_text(raw)
            has_table = _detect_tables(text)
            img_count, img_w, img_h, pw, ph = _image_info(page)
            page_class = classify_page(text, img_count, img_w, img_h, pw, ph)
            if page_class == PageClass.TEXT:
                pt = _page_type(text, has_table)
            else:
                pt = PageType.SCAN
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
                page_class=page_class,
                usable_as_evidence=page_class == PageClass.TEXT,
                ufffd_count=text.count("\ufffd"),
                has_table=has_table,
                character_count=len(text),
                extraction_status=ExtractionStatus.EXTRACTED,
            ))
    return out


def _statement_minimum(doc_type) -> int:
    """Minimum statement-like pages by document type."""
    if doc_type is None:
        value = "annual"
    elif hasattr(doc_type, "value"):
        value = str(doc_type.value).lower()
    else:
        value = str(doc_type).lower()
    if value == "quarterly":
        return MIN_STATEMENT_LIKE_PAGES_QUARTERLY
    if value == "disclosure":
        return 0
    return MIN_STATEMENT_LIKE_PAGES


def scanned_document_rules(pages: list[PageExtract], doc_type=None) -> list[str]:
    """Return the names of the document-level rules that fire."""
    rules: list[str] = []
    image_pages = [p for p in pages
                   if p.page_class in (PageClass.IMAGE_ONLY, PageClass.IMAGE_DOMINANT)]
    if pages and (len(image_pages) / len(pages)) > IMAGE_PAGE_SHARE_LIMIT:
        rules.append("image_share")
    minimum = _statement_minimum(doc_type)
    if minimum > 0:
        stmt_like = [p for p in pages if is_statement_like(p.page_text)]
        if len(stmt_like) < minimum:
            rules.append("statements_not_found")
    return rules


def is_scanned_document(
        pages: list[PageExtract], doc_type=None,
) -> tuple[bool, list[tuple[int, str]]]:
    """Decide whether a document is a scanned/image-dominated PDF.

    The statement-evidence minimum is scoped by document type: annual reports
    use MIN_STATEMENT_LIKE_PAGES, quarterly results use the lower quarterly
    constant, and disclosures are exempt. The image-share rule applies to all
    types. An empty page is BLANK, never GARBLED, and a single bad page never
    excludes a document by itself.
    """
    triggers: list[tuple[int, str]] = []
    rules = scanned_document_rules(pages, doc_type)
    if "image_share" in rules:
        image_pages = [p for p in pages
                       if p.page_class in (PageClass.IMAGE_ONLY, PageClass.IMAGE_DOMINANT)]
        triggers.extend((p.pdf_page_index, p.page_class.value) for p in image_pages)
    if "statements_not_found" in rules:
        non_text = [p for p in pages if p.page_class != PageClass.TEXT]
        if non_text:
            triggers.extend((p.pdf_page_index, p.page_class.value) for p in non_text)
        else:
            triggers.extend((p.pdf_page_index, p.page_class.value) for p in pages)
    seen: set[tuple[int, str]] = set()
    unique: list[tuple[int, str]] = []
    for trigger in triggers:
        if trigger not in seen:
            seen.add(trigger)
            unique.append(trigger)
    return (len(rules) > 0, unique)


def has_scanned_statement_page(pages: list[PageExtract], doc_type=None) -> bool:
    """Backward-compatible boolean wrapper around is_scanned_document."""
    return is_scanned_document(pages, doc_type)[0]


def scanned_document_triggers(pages: list[PageExtract], doc_type=None) -> list[tuple[int, str]]:
    """Triggering (page index, class) pairs for an excluded document."""
    return is_scanned_document(pages, doc_type)[1]



def text_density_ok(text: str) -> bool:
    return len(text.strip()) >= TEXT_DENSITY_MIN_CHARS and not _is_garbled(text)


def _image_dimensions(page, target_name: str) -> tuple[int, int]:
    try:
        resources = page.get("/Resources")
        if resources is not None and hasattr(resources, "get_object"):
            resources = resources.get_object()
        if not (isinstance(resources, dict) and "/XObject" in resources):
            return 0, 0
        xobjects = resources["/XObject"]
        if hasattr(xobjects, "get_object"):
            xobjects = xobjects.get_object()
        if not isinstance(xobjects, dict):
            return 0, 0
        for key in xobjects:
            obj = xobjects[key]
            if hasattr(obj, "get_object"):
                obj = obj.get_object()
            if not isinstance(obj, dict) or obj.get("/Subtype") != "/Image":
                continue
            name = str(key).lstrip("/")
            if name != target_name and target_name not in str(key):
                continue
            try:
                return int(obj.get("/Width", 0)), int(obj.get("/Height", 0))
            except (TypeError, ValueError):
                return 0, 0
    except Exception:  # noqa: BLE001
        pass
    return 0, 0


def extract_images(pdf_bytes: bytes) -> list[dict]:
    """Extract embedded image bytes without OCR or image decoding."""
    from io import BytesIO

    from pypdf import PdfReader

    out: list[dict] = []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception:  # noqa: BLE001
        return out
    for i, page in enumerate(reader.pages, start=1):
        try:
            images = page.images
        except Exception:  # noqa: BLE001
            continue
        index = 0
        while True:
            try:
                image = images[index]
            except IndexError:
                break
            except Exception:  # noqa: BLE001
                # One unsupported image format (e.g. JBIG2 without the
                # optional decoder binary) must not abort the document.
                index += 1
                continue
            index += 1
            try:
                data = image.data
            except Exception:  # noqa: BLE001
                continue
            name = getattr(image, "name", "image")
            width, height = _image_dimensions(page, str(name))
            out.append({
                "page_index": i,
                "name": str(name),
                "data": data,
                "width": width,
                "height": height,
            })
    return out
