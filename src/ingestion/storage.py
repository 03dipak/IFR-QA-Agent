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


class RawStorage(Protocol):
    def put(self, key: str, data: bytes) -> str: ...  # returns checksum
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


class LocalRawStorage:
    """Filesystem-backed raw storage. Keys are relative paths."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Prevent path traversal
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"invalid storage key: {key}")
        return p

    def put(self, key: str, data: bytes) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.is_file():
            p.unlink()
