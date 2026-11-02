"""Checksum utilities for ingestion.

Implements lld_ingestion_pdf.txt section 3 (checksummer): sha256 per file,
the byte-identity key. Used for deduplication and provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Return hex sha256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Stream-hash a file (memory-safe for large PDFs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Hash normalized text (page extraction hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
