"""Gold authoring CLI: status, check, drafts, verify, supersede, reset-review, hash.

All commands edit the SAME file (data/eval/datasets/gold_data.jsonl) atomically.
CSV exports nothing; candidates.csv is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .gold_schema import (canonical, GOLD_SET_VERSION, row_hash,
                          signed_content_hash, validate_row)
from .gold_rows import load_rows, persist, append_correction, bump_set_version
from .gold_review import (sign, reject, reset, skip, counts,
                          unsign_if_changed, review_events, live_reviewed)

GOLD = "data/eval/datasets/gold_data.jsonl"
DRAFTS = "data/eval/datasets/packs/drafts.jsonl"
CORRECTIONS = "data/eval/datasets/gold_data.corrections.jsonl"
REVIEW_DIR = Path("data/gold/review")
PAGE_DIR = "data/corpus/extract/page_extracts"


def _page_text(doc_id: str, page: int) -> str:
    path = Path(PAGE_DIR) / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pdf_page_index") == page:
                return rec.get("page_text") or ""
    return ""


def cmd_status(rows: list[dict]) -> int:
    c = counts(rows)
    print("GOLD SET STATUS")
    print(f"  rows {c['total']}  live {c['live']}  rejected {c['rejected']}  "
          f"superseded {c['superseded']}")
    print(f"  reviewed {c['reviewed']}  two_reviewers {c['two_reviewers']}  "
          f"no_separation {c['no_separation']}")
    print(f"  gold_set_version {GOLD_SET_VERSION}")
    return 0


def cmd_check(rows: list[dict], reviewer: str | None) -> int:
    changed = []
    for r in rows:
        if unsign_if_changed(r, reviewer or "system"):
            changed.append(r["id"])
    if changed:
        persist(rows, GOLD)
        print(f"un-signed {len(changed)} rows whose signed fields changed: "
              + ", ".join(changed))
    bad = []
    for r in rows:
        if r.get("review_status") in ("self_reviewed", "peer_reviewed"):
            if not r.get("reviewers") or not r.get("annotated_at"):
                bad.append((r["id"], "signed row missing reviewers or annotated_at"))
        for prob in validate_row(r):
            if prob.startswith("schema"):
                bad.append((r["id"], prob))
    for rid, msg in bad:
        print(f"  [FAIL] {rid}: {msg}")
    print(f"check: {len(changed)} un-signed, {len(bad)} failures")
    return 1 if bad else 0


def cmd_drafts(rows: list[dict], path: str) -> int:
    drafts = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    key = lambda r: (r["evidence"][0]["document_id"], r["evidence"][0]["pdf_page"],
                     r["evidence"][0].get("snippet_hash"), r["category"])
    existing = {key(r) for r in rows}
    folded, already = 0, 0
    nid = max([int(r["id"][2:]) for r in rows if r.get("id", "").startswith("G-")] or [0]) + 1
    for d in drafts:
        k = key(d)
        if k in existing:
            already += 1
            continue
        d = dict(d)
        d["id"] = f"G-{nid:03d}"
        nid += 1
        rows.append(d)
        folded += 1
    persist(rows, GOLD)
    print(f"drafts: folded {folded} new, {already} already present")
    return 0


def cmd_hash(rows: list[dict]) -> int:
    payload = "".join(canonical(r).decode("utf-8") for r in
                      sorted(rows, key=lambda r: r["id"])).encode("utf-8")
    print(f"sha256:{hashlib.sha256(payload).hexdigest()}")
    print(f"gold_set_version {GOLD_SET_VERSION}  rows {len(rows)}")
    return 0


def cmd_supersede(rows: list[dict], args) -> int:
    rid = args.supersede_id
    row = next((r for r in rows if r["id"] == rid), None)
    if row is None:
        print(f"no row {rid}", file=sys.stderr)
        return 1
    old_hash = row_hash(row)
    row["status"] = "superseded"
    row["notes"] = (row.get("notes", "").rstrip()
                    + f" [superseded {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}: {args.reason}]").strip()
    new_version = bump_set_version(rows, rows[0].get("gold_set_version", GOLD_SET_VERSION))
    persist(rows, GOLD)
    record = {"id": rid, "original_row_hash": f"sha256:{old_hash}",
              "reason": args.reason, "approver": args.reviewer or "",
              "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "gold_set_version_after": new_version,
              "metrics": ["n_superseded"]}
    append_correction(CORRECTIONS, record)
    print(f"superseded {rid}; corrections appended; gold_set_version={new_version}")
    return 0


def cmd_reset_review(rows: list[dict], args) -> int:
    reviewer = args.reviewer or ""
    try:
        if not reviewer:
            reviewer = input("reviewer id: ").strip()
    except EOFError:
        reviewer = ""
    if not reviewer:
        print("reset-review requires a reviewer id", file=sys.stderr)
        return 1
    signed = [r for r in rows if r.get("review_status") in ("self_reviewed", "peer_reviewed")]
    if args.reset_category:
        signed = [r for r in signed if r.get("category") == args.reset_category]
    if args.yes is not True:
        ans = input(f"reset {len(signed)} rows to unreviewed? [y/N] ").strip().lower()
        if ans != "y":
            print("aborted")
            return 1
    reason = args.reason or "reset-review"
    reset_at = __import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    signed_ids = [r["id"] for r in signed]
    for r in signed:
        reset(r, reviewer)
    persist(rows, GOLD)
    correction = {"reviewer": reviewer, "reason": reason, "action": "reset_review",
                  "recorded_at": reset_at, "rows_affected": len(signed),
                  "row_ids": signed_ids}
    Path(CORRECTIONS).parent.mkdir(parents=True, exist_ok=True)
    with Path(CORRECTIONS).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(correction, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"reset {len(signed)} rows to unreviewed; correction logged to {CORRECTIONS}")
    return 0


def _parse_ids(spec: str) -> set[str]:
    out: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..", 1)
            try:
                n1, n2 = int(a.split("-")[-1]), int(b.split("-")[-1])
            except ValueError:
                raise ValueError(f"bad id range {part!r}")
            for n in range(min(n1, n2), max(n1, n2) + 1):
                out.add(f"G-{n:03d}")
        else:
            out.add(part)
    return out


def cmd_verify(rows: list[dict], args, gold_path: str = GOLD) -> int:
    reviewer = args.reviewer or None
    try:
        if not reviewer:
            reviewer = input("reviewer id: ").strip()
    except EOFError:
        reviewer = ""
    if not reviewer:
        print("verification requires a reviewer id", file=sys.stderr)
        return 1
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pending = [r for r in rows if r.get("status") == "active"
               and (r.get("review_status") == "unreviewed"
                    or ((args.ids or args.ids_file) and r.get("review_status") in ("self_reviewed", "peer_reviewed")))]
    ids_spec = args.ids
    if getattr(args, "ids_file", None):
        frows = [json.loads(l) for l in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        ids_spec = ",".join(r.get("id") or r.get("row_id") for r in frows if r.get("id") or r.get("row_id"))
    if ids_spec:
        try:
            wanted = _parse_ids(ids_spec)
        except ValueError as e:
            print(f"bad --ids: {e}", file=sys.stderr)
            return 1
        pending = [r for r in pending if r["id"] in wanted]
        missing = wanted - {r["id"] for r in pending}
        if missing:
            print(f"note: {len(missing)} requested ids not pending (already signed or absent): "
                  + ", ".join(sorted(missing)))
    else:
        print(f"{len(pending)} rows pending review")
    dwell: list[tuple[float, str]] = []
    for r in pending:
        if reviewer and reviewer in (r.get("reviewers") or []):
            print(f"skip {r['id']}: already signed by {reviewer}")
            continue
        t0 = __import__("time").time()
        ev = (r.get("evidence") or [None])[0]
        print("=" * 72)
        print(f"{r['id']}  category={r['category']}  company={r.get('company_id')}"
              f"  period={r.get('period')}")
        print(f"question: {r['question']}")
        print(f"expected answer: {r.get('answer_text') or r.get('expected_answer')}")
        if ev is None:
            print("no cited evidence (unanswerable/abstain row) — "
                  "confirm the question is truly unanswerable, then sign")
            text, png = "", None
        else:
            did, page = ev["document_id"], ev["pdf_page"]
            print(f"cited evidence: document={did[:12]} page={page} role={ev.get('evidence_role')}")
            print(f"excerpt: {ev.get('excerpt')}")
            text = _page_text(did, page)
            print(f"--- page {page} text (truncated to 3000 chars) ---")
            print(text[:3000])
            png = REVIEW_DIR / f"{did}_p{page}.png"
        if png is not None and not png.is_file():
            try:
                subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", str(page),
                                "-l", str(page), "-singlefile",
                                f"data/corpus/raw/{did}.pdf",
                                str(REVIEW_DIR / f"{did}_p{page}")],
                               check=True, capture_output=True, timeout=120)
            except Exception as e:
                print(f"(png render failed: {type(e).__name__}: {e})")
        if png is not None and png.is_file():
            print(f"page PNG: {png}")
        try:
            action = input("[enter] sign / x reject / s skip > ").strip().lower()
        except EOFError:
            print("aborted (EOF on stdin)", file=sys.stderr)
            persist(rows, GOLD)
            return 1
        dwell_s = __import__("time").time() - t0
        dwell.append((dwell_s, r["id"]))
        if action in ("", "enter", "sign", "y") and dwell_s < 20 and not args.fast:
            print(f"refused sign of {r['id']}: only {dwell_s:.1f}s spent "
                  f"(need >= 20s, or pass --fast '<reason>' to override)", file=sys.stderr)
            skip(r)
            print("leaving row unreviewed for now")
            persist(rows, gold_path)
            continue
        if dwell_s < 20 and args.fast:
            from .gold_review import _append_event
            _append_event(r, f"fast {dwell_s:.1f}s reason={args.fast!r}")
        try:
            if action in ("", "enter", "sign", "y"):
                sign(r, reviewer)
                print(f"signed by {reviewer}")
            elif action in ("x", "reject", "n"):
                try:
                    reason = input("reason: ").strip() or "no reason given"
                except EOFError:
                    reason = "no reason given"
                reject(r, reviewer, reason)
                print("rejected")
            else:
                skip(r)
                print("skipped")
        except ValueError as e:
            print(f"refused: {e}", file=sys.stderr)
            persist(rows, gold_path)
            return 1
        persist(rows, gold_path)
    if dwell:
        dwell.sort(key=lambda t: t[0])
        print("fastest rows this session:")
        for secs, rid in dwell[:5]:
            print(f"  {rid}: {secs:.1f}s")
    print("verify pass complete")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="src.eval.gold_author")
    ap.add_argument("--input", default=GOLD)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--drafts", type=Path)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reset-review", action="store_true")
    ap.add_argument("--reset-category", default=None)
    ap.add_argument("--ids", default=None,
                    help="row id range G-001..G-034 or comma list G-001,G-005")
    ap.add_argument("--ids-file", default=None,
                    help="jsonl file of rows with id/row_id (e.g. eval subset)")
    ap.add_argument("--supersede", dest="supersede_id", default=None)
    ap.add_argument("--reason", default="")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--reviewer", default=None)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--fast", default=None, help="fast-sign reason; logs the reason on the row")
    args = ap.parse_args()

    rows = load_rows(args.input)
    if args.status:
        return cmd_status(rows)
    if args.check:
        return cmd_check(rows, args.reviewer)
    if args.drafts:
        return cmd_drafts(rows, str(args.drafts))
    if args.verify:
        return cmd_verify(rows, args, args.input)
    if args.reset_review:
        return cmd_reset_review(rows, args)
    if args.supersede_id:
        return cmd_supersede(rows, args)
    if args.hash:
        return cmd_hash(rows)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
