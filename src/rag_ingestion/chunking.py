"""Chunking for the dense-slice baseline index.

Two variants:
  * ``page`` — one chunk per PDF page, plus one chunk per table record
    keyed to its page.
  * ``fixed`` — fixed-size windows (default 800 chars, 100 overlap) over
    page text, plus the same table chunks.

Chunk metadata (both variants): document_id, page (1-indexed PDF page),
company_id, period (fiscal_year), variant, doc_type, plus statement_kind
and unit_header for table chunks, and chunk_type in {page, table}.

Reads ONLY the contract: data/corpus/freeze_v1.json + page/table extracts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

FREEZE = "data/corpus/freeze_v1.json"
PAGE_DIR = "data/corpus/extract/page_extracts"
TABLE_DIR = "data/corpus/extract/table_records"
DENSE_COMPANIES = ("bhel", "maruti_suzuki", "deepak_nitrite", "srf",
                   "berger_paints", "dabur", "bosch")
DENSE_YEARS = ("FY2024", "FY2025", "FY2026")
CHUNK_CONFIG = {"fixed_chars": 800, "fixed_overlap": 100,
                "variant_page": {"unit": "page"},
                "variant_fixed": {"unit": "fixed_chars"}}


def slice_documents() -> list[dict]:
    docs = json.load(open(FREEZE, encoding="utf-8"))["documents"]
    annuals = [d for d in docs
               if d["doc_type"] == "annual" and d["fiscal_year"] in DENSE_YEARS]
    quarterlies = [d for d in docs
                   if d["doc_type"] == "quarterly"
                   and d["company_id"] in DENSE_COMPANIES]
    return sorted(annuals + quarterlies, key=lambda d: d["document_id"])


def _meta(doc: dict, page: int, chunk_type: str, **extra) -> dict:
    m = {"document_id": doc["document_id"], "page": int(page),
         "company_id": doc["company_id"], "period": doc["fiscal_year"],
         "variant": doc.get("variant"), "doc_type": doc["doc_type"],
         "chunk_type": chunk_type}
    m.update(extra)
    return m


def iter_page_chunks(doc: dict) -> Iterator[dict]:
    path = Path(PAGE_DIR) / f"{doc['document_id']}.pages.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("page_text") or ""
            if not text.strip():
                continue
            page = rec.get("pdf_page_index")
            if not isinstance(page, int):
                continue
            yield {"id": f"{doc['document_id']}:p{page}", "text": text,
                   "meta": _meta(doc, page, "page")}


def _fixed_windows(text: str, size: int, overlap: int) -> Iterator[str]:
    step = size - overlap
    if step <= 0:
        step = size
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += step


def iter_fixed_chunks(doc: dict, size: int = 800, overlap: int = 100) -> Iterator[dict]:
    path = Path(PAGE_DIR) / f"{doc['document_id']}.pages.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("page_text") or ""
            page = rec.get("pdf_page_index")
            if not isinstance(page, int):
                continue
            for n, window in enumerate(_fixed_windows(text, size, overlap)):
                if window.strip():
                    yield {"id": f"{doc['document_id']}:p{page}:f{n}",
                           "text": window,
                           "meta": _meta(doc, page, "fixed",
                                         char_start=n * (size - overlap))}


def _row_text(row: dict) -> str:
    label = (row.get("row_label_raw") or "").strip()
    cells = row.get("cells") or []
    raw = [str(c.get("raw") or "").strip() for c in cells]
    raw = [c for c in raw if c]
    return f"{label}: {' | '.join(raw)}" if label else " | ".join(raw)


def iter_table_chunks(doc: dict) -> Iterator[dict]:
    path = Path(TABLE_DIR) / f"{doc['document_id']}.tables.jsonl"
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            page = rec.get("page_start")
            if not isinstance(page, int):
                continue
            parts = [f"statement_kind: {rec.get('statement_kind')}",
                     f"unit: {(rec.get('unit_header') or '').strip()}",
                     f"caption: {(rec.get('caption') or '').strip()}",
                     f"reporting_basis: {rec.get('reporting_basis')}"]
            rows = rec.get("rows") or []
            parts.extend(_row_text(r) for r in rows)
            text = "\n".join(p for p in parts if p.strip())
            if not text.strip():
                continue
            yield {"id": f"{doc['document_id']}:t{n}", "text": text,
                   "meta": _meta(doc, page, "table",
                                 statement_kind=rec.get("statement_kind"),
                                 reporting_basis=rec.get("reporting_basis"))}


def chunks_for(doc: dict, variant: str) -> list[dict]:
    if variant == "page":
        chunks = list(iter_page_chunks(doc))
    elif variant == "fixed":
        chunks = list(iter_fixed_chunks(doc,
                                        CHUNK_CONFIG["fixed_chars"],
                                        CHUNK_CONFIG["fixed_overlap"]))
    else:
        raise ValueError(f"unknown chunk variant: {variant}")
    return chunks + list(iter_table_chunks(doc))


def freeze_manifest_sha256() -> str:
    return json.load(open(FREEZE, encoding="utf-8"))["manifest_sha256"]


def index_version(variant: str, embed_model: str | None) -> str:
    """index_version = hash(freeze manifest sha256 + chunk config + embed model)."""
    payload = json.dumps(
        {"manifest_sha256": freeze_manifest_sha256(),
         "variant": variant,
         "chunk_config": CHUNK_CONFIG,
         "embedding_model": embed_model or "none"},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_chunks(variant: str) -> tuple[list[dict], float]:
    import time
    t0 = time.time()
    chunks: list[dict] = []
    for doc in slice_documents():
        chunks.extend(chunks_for(doc, variant))
    chunks.sort(key=lambda c: c["id"])
    return chunks, time.time() - t0
