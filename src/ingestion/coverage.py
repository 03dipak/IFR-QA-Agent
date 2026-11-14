"""Coverage-matrix probe: check rows with urls, set PROBED_OK / failure.

Reads a coverage_matrix CSV, probes each row that has a non-empty url and a
status other than PROBED_OK with a bounded ranged GET (<=512 bytes), writes
PROBED_OK or a short failure reason back into the status column, and prints
per-company and total FOUND vs MISSING counts plus the projected document
count (rows whose status is FOUND or PROBED_OK, i.e. expected to produce a
document).

Etiquette: honest client name, <=512 bytes read per probe, one request per
second. Never downloads a whole document.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from .discovery import PROBE_MAX_BYTES, probe_pdf_url

STATUS_OK = "PROBED_OK"
STATUS_FOUND = "FOUND"
STATUS_MISSING = "MISSING"


def failure_reason(probe) -> str:
    if probe.status is None:
        return "TIMEOUT" if probe.error else "ERROR"
    if probe.status not in (200, 206):
        return f"HTTP_{probe.status}"
    return "NOT_PDF"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.ingestion.coverage",
        description="Probe coverage-matrix urls and report FOUND/MISSING.",
    )
    parser.add_argument("--matrix", nargs="?",
                        const="data/seeds/coverage_matrix.csv",
                        default="data/seeds/coverage_matrix.csv",
                        help="coverage matrix CSV (updated in place); "
                             "default: data/seeds/coverage_matrix.csv")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="seconds between probes (default 1.0)")
    args = parser.parse_args(argv)

    path = Path(args.matrix)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys()) if rows else []
    if "status" not in fieldnames:
        raise SystemExit("matrix lacks a 'status' column")

    first = True
    for row in rows:
        url = (row.get("url") or "").strip()
        status = (row.get("status") or "").strip()
        if not url or status == STATUS_OK:
            continue
        if not first:
            time.sleep(args.sleep)
        first = False
        probe = probe_pdf_url(url, max_bytes=PROBE_MAX_BYTES)
        row["status"] = STATUS_OK if probe.ok else failure_reason(probe)

    if not first:  # at least one probe ran: persist statuses
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    found = sum(1 for r in rows if r["status"] in (STATUS_FOUND, STATUS_OK))
    missing = sum(1 for r in rows if r["status"] == STATUS_MISSING)
    other = len(rows) - found - missing
    per_company: dict[str, dict[str, int]] = {}
    for row in rows:
        c = per_company.setdefault(
            row["company_id"], {"found": 0, "missing": 0, "other": 0})
        if row["status"] in (STATUS_FOUND, STATUS_OK):
            c["found"] += 1
        elif row["status"] == STATUS_MISSING:
            c["missing"] += 1
        else:
            c["other"] += 1
    for company in sorted(per_company):
        c = per_company[company]
        print(f"{company} found={c['found']} missing={c['missing']} "
              f"other={c['other']}")
    print(f"summary companies={len(per_company)} rows={len(rows)} "
          f"found={found} missing={missing} other={other} "
          f"projected_document_count={found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
