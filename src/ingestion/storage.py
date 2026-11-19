"""Raw PDF storage interface (behind the db layer).

Implements lld_ingestion_pdf.txt 5.6: raw files live OUTSIDE git behind a
storage interface, checksummed on write, with per-freeze backup verification.
This module provides a local-filesystem implementation suitable for the
capstone (D6: local only). Cloud backends can implement the same interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Protocol


def _atomic_write_text(path: Path, write_lines) -> None:
    """Write via a temp file in the same directory, then rename.

    Only a complete file ever appears at ``path``; a crash mid-write leaves
    a ``*.tmp-*`` sibling that startup sweeps remove.
    """
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            write_lines(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def sweep_extract_dir(root: str | Path) -> list[Path]:
    """Startup sweep for extract dirs: remove extract files that cannot be
    trusted, so a re-run redacts them instead of skipping them.

    Removes: stray ``*.tmp-*`` files from interrupted atomic writes,
    zero-byte ``*.jsonl`` extract files, and non-empty ``*.jsonl`` files
    with at least one line that does not parse as JSON (or with no
    parseable lines at all). A zero-byte ``.tables.jsonl`` with a sibling
    ``.empty`` marker (written by LocalTableRecordStore for legitimate
    zero-record extractions) is kept, not redone; a marker whose data file
    is missing or non-zero is stale and is removed instead. Returns the
    removed paths.
    """
    removed: list[Path] = []
    root = Path(root)
    if not root.is_dir():
        return removed
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if ".tmp-" in path.name:
            path.unlink()
            removed.append(path)
            continue
        if path.name.endswith(".empty"):
            data = path.with_name(path.name[: -len(".empty")])
            if (not data.is_file()) or data.stat().st_size > 0:
                path.unlink()
                removed.append(path)
            continue
        if not path.name.endswith(".jsonl"):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0:
            if path.name.endswith(".tables.jsonl") and (
                    path.with_name(path.name + ".empty").is_file()):
                continue  # legitimate zero-record extraction (marked)
            path.unlink()
            removed.append(path)
            continue
        ok = False
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except ValueError:
                    ok = False
                    break
                ok = True
        if not ok:
            path.unlink()
            removed.append(path)
    return removed


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
            tmp = canonical.with_name(
                f"{canonical.name}.tmp-{uuid.uuid4().hex}")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, canonical)
            except BaseException:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
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


class PageExtractStore(Protocol):
    def put(self, document_id: str, pages: list) -> int: ...
    def get(self, document_id: str) -> list: ...


class LocalPageExtractStore:
    """Filesystem-backed page-extract storage (one JSON object per page)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, document_id: str) -> Path:
        return self.root / f"{document_id}.pages.jsonl"

    def put(self, document_id: str, pages: list) -> int:
        path = self._path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _write(fh):
            for page in pages:
                fh.write(json.dumps(page.model_dump(mode="json"), sort_keys=True))
                fh.write("\n")

        _atomic_write_text(path, _write)
        return len(pages)

    def get(self, document_id: str) -> list:
        from .models import PageExtract

        path = self._path(document_id)
        if not path.is_file():
            return []
        out = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(PageExtract.model_validate_json(line))
        return out


class TableRecordStore(Protocol):
    def put(self, document_id: str, records: list) -> int: ...
    def get(self, document_id: str) -> list: ...


class LocalTableRecordStore:
    """Filesystem-backed table-record storage (one JSON object per table)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, document_id: str) -> Path:
        return self.root / f"{document_id}.tables.jsonl"

    def put(self, document_id: str, records: list) -> int:
        import dataclasses

        path = self._path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _write(fh):
            for record in records:
                payload = dataclasses.asdict(record)
                fh.write(json.dumps(payload, sort_keys=True, default=str))
                fh.write("\n")

        _atomic_write_text(path, _write)
        marker = path.with_name(path.name + ".empty")
        if len(records) == 0:
            marker.write_text(json.dumps(
                {"document_id": document_id, "records": 0,
                 "note": "legitimate empty result from table extractor"},
                sort_keys=True) + "\n", encoding="utf-8")
        else:
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
        return len(records)

    def get(self, document_id: str) -> list:

        from ..tables.models import (
            AuditStatus,
            CellKind,
            ColumnKind,
            FootnoteAssertion,
            PeriodResolutionStatus,
            ReportingBasis,
            RowKind,
            ScopeType,
            StatementKind,
            TableCell,
            TableColumn,
            TableRecord,
            TableRow,
        )

        path = self._path(document_id)
        if not path.is_file():
            return []
        out = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                columns = [TableColumn(**c) for c in data.get("columns", [])]
                rows = []
                for row_data in data.get("rows", []):
                    cells = [TableCell(**c) for c in row_data.get("cells", [])]
                    row_payload = dict(row_data)
                    row_payload["cells"] = cells
                    rows.append(TableRow(**row_payload))
                footnotes = [
                    FootnoteAssertion(**f) for f in data.get("footnote_assertions", [])
                ]
                payload = dict(data)
                payload["columns"] = columns
                payload["rows"] = rows
                payload["footnote_assertions"] = footnotes
                if isinstance(payload.get("statement_kind"), str):
                    payload["statement_kind"] = StatementKind(payload["statement_kind"])
                if isinstance(payload.get("reporting_basis"), str):
                    payload["reporting_basis"] = ReportingBasis(payload["reporting_basis"])
                if isinstance(payload.get("audit_status"), str):
                    payload["audit_status"] = AuditStatus(payload["audit_status"])
                for col in columns:
                    if isinstance(col.column_kind, str):
                        col.column_kind = ColumnKind(col.column_kind)
                    if isinstance(
                            col.period_resolution_status, str):
                        col.period_resolution_status = (
                            PeriodResolutionStatus(col.period_resolution_status)
                        )
                for row in rows:
                    if isinstance(row.row_kind, str):
                        row.row_kind = RowKind(row.row_kind)
                    for cell in row.cells:
                        if isinstance(cell.cell_kind, str):
                            cell.cell_kind = CellKind(cell.cell_kind)
                for footnote in footnotes:
                    if isinstance(footnote.scope_type, str):
                        footnote.scope_type = ScopeType(footnote.scope_type)
                out.append(TableRecord(**payload))
        return out
