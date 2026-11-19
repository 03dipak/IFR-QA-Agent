"""Gold-file read/write with EDGAR last-wins replay and atomic persist."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .gold_schema import canonical, GOLD_SET_VERSION


class HoldoutAccessError(RuntimeError):
    """Raised when a code path tries to read a holdout row."""


def assert_no_holdout(rows, *, context: str = "") -> None:
    """Guard for every consumer path: holdout rows must never be read."""
    bad = [r.get("id") or r.get("row_id") or "?" for r in rows
           if r.get("split") == "holdout"]
    if bad:
        where = f" ({context})" if context else ""
        raise HoldoutAccessError(
            f"holdout rows must not be read{where}: " + ", ".join(bad))


def load_rows(path: str | Path, active_only: bool = False) -> list[dict]:
    """Load gold rows. active_only=True is for consumers (harness, scorer,
    gold_check, public export): superseded/rejected rows never reach them.
    Lifecycle tools (author, merge, freeze, migrate) must load everything."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: dict[str, dict] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if not rid:
                continue
            if active_only and rec.get("status") not in (None, "active"):
                continue
            if rid not in rows:
                order.append(rid)
            rows[rid] = rec
    return [rows[rid] for rid in order]


def dump_rows(rows: list[dict]) -> str:
    return "".join(canonical(r).decode("utf-8") for r in
                  sorted(rows, key=lambda r: r.get("id") or ""))


def persist(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(dump_rows(rows), encoding="utf-8")
    os.replace(tmp, path)


def append_correction(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(canonical(record).decode("utf-8"))


def next_id(rows: list[dict]) -> str:
    mx = 0
    for r in rows:
        rid = str(r.get("id") or "")
        if rid.startswith("G-"):
            try:
                mx = max(mx, int(rid[2:]))
            except ValueError:
                pass
    return f"G-{mx + 1:03d}"


def bump_set_version(rows: list[dict], current: str) -> str:
    base = current.split(".")[0]
    suffix = current[len(base):]
    if suffix and suffix.startswith("."):
        try:
            n = int(suffix[1:])
        except ValueError:
            n = 0
    else:
        n = 0
    new = f"{base}.{n + 1}"
    for r in rows:
        r["gold_set_version"] = new
    return new
