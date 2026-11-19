"""Review state machine: stamps, ordered history, row_revision, multi-reviewer.

Event clauses are written into notes in chronological order:
  [review history: reviewed <ts> by <id>, reviewed <ts> by <id>, reset <ts> by <id>, ...]
row_revision = 1 + number of events (0 while unsigned). A signature is only
valid while the row's signed-field hash matches the one stored at signing time.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from .gold_schema import signed_content_hash

_HISTORY_RE = re.compile(r"\[review history:[^\]]*\]")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def review_events(row: dict) -> list[str]:
    m = _HISTORY_RE.search(row.get("notes") or "")
    if not m:
        return []
    inner = m.group(0)[len("[review history: "):-1]
    return [e.strip() for e in inner.split(",") if e.strip()]


def _append_event(row: dict, clause: str) -> None:
    ev = review_events(row)
    ev.append(clause)
    m = _HISTORY_RE.search(row.get("notes") or "")
    if m:
        row["notes"] = (row["notes"][:m.start()] +
                        "[review history: " + ", ".join(ev) + "]" +
                        row["notes"][m.end():])
    else:
        row["notes"] = ((row.get("notes") or "").rstrip() +
                        " [review history: " + ", ".join(ev) + "]").strip()
    row["row_revision"] = 1 + len(ev)


def sign(row: dict, reviewer: str) -> dict:
    if not (reviewer or "").strip():
        raise ValueError("signing requires a reviewer id")
    ev = review_events(row)
    typer = "reviewed"
    row.setdefault("reviewers", [])
    if reviewer not in row["reviewers"]:
        row["reviewers"].append(reviewer)
    # setdefault would keep a pre-existing null from the draft; overwrite it
    row["annotated_at"] = row.get("annotated_at") or _now()
    row["review_status"] = "peer_reviewed" if len(row["reviewers"]) >= 2 else "self_reviewed"
    row["re_annotated_at"] = _now() if ev else None
    row["signed_sha256"] = signed_content_hash(row)
    _append_event(row, f"{typer} {_now()} by {reviewer}")
    return row


def reject(row: dict, reviewer: str, reason: str) -> dict:
    if not (reviewer or "").strip():
        raise ValueError("rejecting requires a reviewer id")
    row["status"] = "rejected"
    row["reviewer"] = reviewer
    row["signed_sha256"] = signed_content_hash(row)
    _append_event(row, f"rejected {_now()} by {reviewer}: {reason}")
    return row


def skip(row: dict) -> dict:
    return row


def reset(row: dict, reviewer: str) -> dict:
    if not (reviewer or "").strip():
        raise ValueError("reset requires a reviewer id")
    row["review_status"] = "unreviewed"
    row["reviewers"] = []
    row["re_annotated_at"] = None
    row["signed_sha256"] = None
    _append_event(row, f"reset {_now()} by {reviewer}")
    return row


def content_changed(row: dict) -> bool:
    sig = row.get("signed_sha256")
    if row.get("review_status") in ("unreviewed",) or not sig:
        return False
    return sig != signed_content_hash(row)


def unsign_if_changed(row: dict, reviewer: str = "system") -> bool:
    if not content_changed(row):
        return False
    row["review_status"] = "unreviewed"
    row["reviewers"] = []
    row["re_annotated_at"] = None
    row["signed_sha256"] = None
    _append_event(row, f"reset {_now()} by {reviewer} (content changed after signing)")
    return True


def live_reviewed(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if r.get("status") == "active"
            and r.get("review_status") in ("self_reviewed", "peer_reviewed")]


def counts(rows: list[dict]) -> dict:
    live = [r for r in rows if r.get("status") == "active"]
    live_ids = {r["id"] for r in live}
    rejected = sum(1 for r in rows if r.get("status") == "rejected")
    superseded = sum(1 for r in rows if r.get("status") == "superseded")
    reviewed = sum(1 for r in live
                   if r.get("review_status") in ("self_reviewed", "peer_reviewed"))
    two = sum(1 for r in live if len(r.get("reviewers") or []) >= 2)
    return {"total": len(rows), "live": len(live), "rejected": rejected,
            "superseded": superseded, "reviewed": reviewed,
            "two_reviewers": two, "no_separation": len(live) - two}


def no_separation_ids(rows: list[dict], n_signers: int = 1) -> list[str]:
    return [r["id"] for r in rows
            if r.get("status") == "active"
            and len(r.get("reviewers") or []) < 2]
