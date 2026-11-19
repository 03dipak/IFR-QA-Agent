#!/usr/bin/env python3
"""Extract NSE filing rows from a saved company page into _nse_urls_v1.csv.

Input : data/manual_inbox/_nse_pages/<company_code>.html  (page saved by the
        operator from the NSE integrated-filing view)
Output: deletes + dedupe-append of ixbrl Details rows to
        data/manual_inbox/_nse_urls_v1.csv with columns
        company_code,url,quarter_end,variant,audited,consolidated,kind
Cap   : quarter_end >= 30-Jun-2024 (operator-set cap; both variants kept)

usage:
  uv run python scripts/nse_extract_links.py --company adani_green
"""
import argparse
import csv
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "manual_inbox"
CSV_PATH = INBOX / "_nse_urls_v1.csv"
PAGES = INBOX / "_nse_pages"
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
CAP = date(2024, 6, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", required=True)
    a = ap.parse_args()
    page = PAGES / f"{a.company}.html"
    if not page.exists():
        raise SystemExit(f"missing saved page: {page} -- stop and tell the operator")
    doc = page.read_text(encoding="utf-8", errors="replace")
    rows, kept = [], []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", doc, re.S):
        if "iXBRL_WEB.html" not in tr:
            continue
        cols = re.findall(r'data-ws-symbol-col="[^"]*-(\w+?)"[^>]*>(.*?)</td>', tr, re.S)
        c = {k: re.sub(r"<[^>]+>", "", v).strip() for k, v in cols}
        link = re.search(r'href="(https://nsearchives[^"]*iXBRL_WEB\.html)"', tr)
        if not (c.get("qe_Date") and link):
            continue
        rows.append((c["qe_Date"], c.get("type_Sub") or "Original",
                     c.get("audited") or "", c.get("consolidated") or "",
                     link.group(1)))
    seen = set()
    for qe, sub, aud, con, url in rows:
        if url in seen:
            continue
        seen.add(url)
        m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{4})", qe)
        if m and date(int(m.group(3)), MONTHS[m.group(2).upper()], int(m.group(1))) >= CAP:
            kept.append((qe, sub, aud, con, url))
    existing = set()
    changed = False
    if CSV_PATH.exists():
        existing = {p.split(",")[1] for p in CSV_PATH.read_text().splitlines()[1:] if "," in p}
    with CSV_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        for qe, sub, aud, con, url in kept:
            if url in existing:
                continue
            w.writerow([a.company, url, qe, sub, aud, con, "ixbrl_details"])
            changed = True
    print(f"[extract] {a.company}: kept {len(kept)} in-cap rows, "
          f"appended {sum(1 for r in kept if r[4] not in existing)} new, "
          f"{'updated:' if changed else 'no changes to'} {CSV_PATH.name}")


if __name__ == "__main__":
    main()
