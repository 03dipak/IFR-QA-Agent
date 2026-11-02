"""Raw PDF storage interface (behind the db layer).

Implements lld_ingestion_pdf.txt 5.6: raw files live OUTSIDE git behind a
storage interface, checksummed on write, with per-freeze backup verification.
This module provides a local-filesystem implementation suitable for the
capstone (D6: local only). Cloud backends can implement the same interface.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class StorageConflictError(RuntimeError):
    """Raised when a put would overwrite a different document at the same key."""


class RawStorage(Protocol):
    def put(self, key: str, data: bytes) -> str: ...  # returns checksum
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


class LocalRawStorage:
    """Filesystem-backed raw storage. Keys are relative paths.

    Canonical corpus key convention (LLD-41): the caller passes a bare
    filename, normally ``<sha256>.pdf``, so a root of ``data/corpus/raw``
    stores ``data/corpus/raw/<sha256>.pdf`` with no repeated ``raw`` segment.

    Legacy compatibility: files written by the old pipeline under
    ``raw/<sha256>.pdf`` remain readable (and are found by ``exists``) through
    a read fallback. New writes always go to the canonical path. No automatic
    migration/rename is performed; any migration needs separate approval.

    ``put`` is idempotent for identical bytes and refuses to overwrite a
    different document at the same key.
    """

    LEGACY_PREFIX = "raw/"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Prevent path traversal
        root = self.root.resolve()
        p = (root / key).resolve()
        try:
            p.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"invalid storage key: {key}") from exc
        return p

    def _candidate_paths(self, key: str) -> list[Path]:
        paths = [self._path(key)]
        if not key.startswith(self.LEGACY_PREFIX):
            paths.append(self._path(self.LEGACY_PREFIX + key))
        return paths

    def put(self, key: str, data: bytes) -> str:
        canonical = self._path(key)
        for path in self._candidate_paths(key):
            if path.is_file() and path.read_bytes() != data:
                raise StorageConflictError(
                    f"refusing to overwrite different bytes at storage key: {key}"
                )
        if not canonical.is_file():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> bytes:
        for path in self._candidate_paths(key):
            if path.is_file():
                return path.read_bytes()
        raise FileNotFoundError(self._path(key))

    def exists(self, key: str) -> bool:
        return any(path.is_file() for path in self._candidate_paths(key))

    def delete(self, key: str) -> None:
        for path in self._candidate_paths(key):
            if path.is_file():
                path.unlink()
