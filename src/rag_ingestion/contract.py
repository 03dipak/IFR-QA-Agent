"""Contract reader for rag_ingestion (one-way data_pipeline -> rag_ingestion).

Reads ONLY contract artefacts with the standard library:
  - the document ledger (kept rows, merged per checksum),
  - PageExtract / table-record JSONL paths and contents,
  - the published freeze manifest.

It imports nothing from ``src.ingestion`` (the data_pipeline package):
the boundary is files on disk, defined in docs/ingestion_contract.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

CONTRACT_VERSION = "1"

KEEP_STATES = (None, "keep")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:  # malformed line: skip, never crash the reader
                continue
            if isinstance(rec, dict):
                yield rec


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

_MERGE_KEYS = ("company_id", "doc_type", "fiscal_year", "gate_decision",
               "variant", "extraction_status", "version")


def kept_documents(ledger_path: Path | str) -> list[dict]:
    """Merged document rows whose effective gate decision is keep (or unset).

    Rows are merged per checksum in append order: a later non-empty value
    wins. Only ``kind == "document"`` rows participate.
    """
    merged: dict[str, dict] = {}
    for row in _iter_jsonl(Path(ledger_path)):
        if row.get("kind") != "document":
            continue
        data = row.get("data") or {}
        checksum = data.get("checksum")
        if not checksum:
            continue
        cur = merged.setdefault(checksum, {})
        for key in _MERGE_KEYS:
            if cur.get(key) in (None, "", []) and data.get(key) is not None:
                cur[key] = data[key]
        for key, value in data.items():
            if key not in _MERGE_KEYS:
                cur.setdefault(key, value)
    return [d for d in merged.values() if d.get("gate_decision") in KEEP_STATES]


# ---------------------------------------------------------------------------
# Extract artefacts
# ---------------------------------------------------------------------------

def page_extracts_path(storage_dir: Path | str, document_id: str) -> Path:
    return (Path(storage_dir) / "extract" / "page_extracts"
            / f"{document_id}.pages.jsonl")


def table_records_path(storage_dir: Path | str, document_id: str) -> Path:
    return (Path(storage_dir) / "extract" / "table_records"
            / f"{document_id}.tables.jsonl")


def read_page_extracts(storage_dir: Path | str, document_id: str) -> list[dict]:
    path = page_extracts_path(storage_dir, document_id)
    return list(_iter_jsonl(path)) if path.is_file() else []


def read_table_records(storage_dir: Path | str, document_id: str) -> list[dict]:
    path = table_records_path(storage_dir, document_id)
    return list(_iter_jsonl(path)) if path.is_file() else []


def page_text(storage_dir: Path | str, document_id: str,
              pdf_page_index: int) -> str:
    for rec in read_page_extracts(storage_dir, document_id):
        if rec.get("pdf_page_index") == pdf_page_index:
            return rec.get("page_text") or ""
    return ""


# ---------------------------------------------------------------------------
# Freeze manifest
# ---------------------------------------------------------------------------

def read_freeze_manifest(path: Path | str) -> list[dict]:
    """Rows of the published freeze manifest (one JSON object per line)."""
    return list(_iter_jsonl(Path(path)))
