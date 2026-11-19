#!/usr/bin/env python3
"""NSE manual-inbox ixbrl -> PDF pipeline (generic, no per-company hardcode).

Rules encoded (project standing decisions):
- input: data/manual_inbox/_nse_urls_v1.csv rows (ixbrl_details kind) filtered by --company
- registry: data/acquisition/company_registry.csv provides the ticker used for
  filenames and the symbol printed inside the ixbrl HTML
- fetch: honest UA, no cookies, 1 request/second, stop run on 403
  (robots decisions recorded: nsearchives=404/allow-all; www=allow / except
  /market-data-test)
- print: headless Chrome --print-to-pdf; HTMLs > 1 MB get a script-stripped
  print copy (tables are server-rendered; JS stripped only for print)
- checks: pdftotext -> "Total income" + quarter-end date string present
- resume: existing scratch HTMLs and existing target PDFs are never redone
- revisions: kept only if normalized text hash differs from the Original of
  the same (quarter_end, consolidated/standalone); identical-hash revisions
  stay on disk but are marked withdrawn in the company manifest.json
- end state: PDFs in data/manual_inbox/<company_code>/, run log appended;
  dry-run ingest is run separately by the caller (no --apply)

usage:
  uv run python scripts/nse_ixbrl_pipeline.py --company adani_green
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "manual_inbox"
SCRATCH = INBOX / "_scratch"
RUNLOG = INBOX / "_run_log.txt"
REGISTRY = ROOT / "data" / "acquisition" / "company_registry.csv"
UA = "DataOSSCapstoneResearch/0.1 (academic capstone; honest client; contact: project owner)"
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
QMOVED = {6: "Q1", 9: "Q2", 12: "Q3", 3: "Q4"}


def log(msg):
    print(msg, flush=True)


def robots_allows(url: str) -> bool:
    host = re.match(r"https://([^/]+)/", url).group(1)
    if host == "nsearchives.nseindia.com":       # robots.txt 404 => allow-all
        return True
    if host == "www.nseindia.com":               # allow / ; disallow /market-data-test
        path = re.match(r"https://[^/]+(/.*)?", url).group(1) or "/"
        return not path.startswith("/market-data-test")
    return False


def load_registry() -> dict[str, dict]:
    out = {}
    for p in csv.DictReader(REGISTRY.open()):
        out[p["code"]] = {"display": p["display_name"], "ticker": p["bse_ticker"]}
    return out


def load_rows(company: str):
    rows = []
    for p in csv.reader((INBOX / "_nse_urls_v1.csv").open()):
        if len(p) >= 7 and p[0] == company and "nsearchives" in p[1] \
                and "iXBRL_WEB.html" in p[1]:
            rows.append({"code": p[0], "url": p[1], "qe": p[2], "variant": p[3],
                         "audited": p[4], "con": p[5], "kind": p[6]})
    return rows


def fetch_if_missing(r, ticker) -> Path:
    seq = re.search(r"INDAS_(\d+)_", r["url"]).group(1)
    out = SCRATCH / f"{ticker}_{r['qe']}_{r['variant'].lower()}_{r['con'].lower()}_{seq}.html"
    if out.exists() and out.stat().st_size > 20000:
        return out                                 # resume: never refetch
    if not robots_allows(r["url"]):
        raise SystemExit(f"robots disallow: {r['url']} -- stopping run")
    time.sleep(1)
    res = subprocess.run(["curl", "-s", "-w", "%{http_code}", "-A", UA,
                          "-o", str(out), r["url"]], capture_output=True, text=True, timeout=180)
    code = res.stdout.strip()[-3:]
    size = out.stat().st_size if out.exists() else 0
    if code == "403":
        raise SystemExit(f"HTTP 403 on {r['url']} -- stopping run (list for manual save)")
    if code != "200" or size <= 20000:
        raise SystemExit(f"fetch failed code={code} size={size} url={r['url']}")
    return out


def header_fields(text: str) -> dict:
    t = re.sub(r"<[^>]+>", "|", text)
    t = html.unescape(t).replace("\n", "|")
    t = re.sub(r"\|+", "|", t)

    def grab(key):
        m = re.search(re.escape(key) + r"[| ]+([^|]{1,80})", t)
        return m.group(1).strip() if m else ""

    return {"symbol": grab("NSE Symbol"),
            "nature": grab("Nature of report standalone or consolidated"),
            "audited": grab("Whether results are audited or unaudited for the quarter ended")}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def pdf_text(pdf: Path) -> str:
    return subprocess.run(["pdftotext", "-q", str(pdf), "-"],
                          capture_output=True, text=True, timeout=180).stdout


def text_hash(pdf: Path) -> str:
    return hashlib.sha256(norm(pdf_text(pdf)).encode()).hexdigest()[:16]


def html_norm_hash(html_path: Path) -> str:
    doc = html_path.read_text(encoding="utf-8", errors="replace")
    doc = re.sub(r"<(script|style)\b.*?(?:</\1>|$)", "", doc, flags=re.S | re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", doc))
    return hashlib.sha256(norm(t).encode()).hexdigest()[:12]


def print_pdf(html_path: Path, out_pdf: Path, strip_scripts=False) -> bool:
    src = html_path
    if strip_scripts:                              # only for print, never for audit
        src = html_path.with_suffix(".printcopy.html")
        doc = html_path.read_text(encoding="utf-8", errors="replace")
        doc = re.sub(r"<script\b.*?</script>", "", doc, flags=re.S | re.I)
        src.write_text(doc, encoding="utf-8")
    cmd = ["google-chrome", "--headless", "--disable-gpu", "--no-first-run",
           "--no-pdf-header-footer", "--virtual-time-budget=15000",
           f"--print-to-pdf={out_pdf}", f"file://{src}"]
    try:
        subprocess.run(cmd, capture_output=True, timeout=420)
    except subprocess.TimeoutExpired:
        return False
    return out_pdf.exists() and out_pdf.stat().st_size > 1000


def big_print(html_path: Path, out_pdf: Path) -> bool:
    """Print a very large iXBRL HTML by slicing it into table slabs and
    uniting the slab PDFs (chrome print stalls on multi-MB single-shot)."""
    doc = html_path.read_text(encoding="utf-8", errors="replace")
    doc = re.sub(r"<script\b.*?</script>", "", doc, flags=re.S | re.I)
    doc = re.sub(r"(?s)[^{}]{0,400}\{[^{}]*\}",
                 lambda m: "" if ":has" in m.group(0) else m.group(0), doc)
    head_end = doc.rfind("</head>")
    head = doc[:head_end + 7] if head_end > 0 else doc
    body_pos = doc.find("<body")
    body = doc[doc.find(">", body_pos) + 1:] if body_pos >= 0 else doc
    body = body.replace("</body></html>", "")
    # cut at row boundaries ~ every max_bytes; browsers tolerate truncated tables
    cuts = [0]
    for m in re.finditer(r"<tr\b", body):
        if m.start() - cuts[-1] >= 700_000:
            cuts.append(m.start())
    slabs = []
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else len(body)
        piece = (head + " <body> " + body[c:end] + " </body></html>")
        p = html_path.with_suffix(f".slab{i}.html")
        p.write_text(piece, encoding="utf-8")
        slabs.append(p)
    parts = []
    ok = True
    for s in slabs:
        sl_pdf = s.with_suffix(".slab.pdf")
        if print_pdf(s, sl_pdf):
            parts.append(sl_pdf)
        else:
            ok = False
            break
    if parts:
        # partial slabs still count; report as failure only if any missing
        out_pdf.unlink(missing_ok=True)
        subprocess.run(["pdfunite", *parts, str(out_pdf)], timeout=300)
        ok = ok and out_pdf.exists() and out_pdf.stat().st_size > 0
    for s in slabs:
        s.unlink(missing_ok=True)
        s.with_suffix(".slab.pdf").unlink(missing_ok=True)
    return ok


def landscape_print(html_path: Path, out_pdf: Path, strip_scripts=False) -> bool:
    wrapped = html_path.with_suffix(".landcopy.html")
    doc = html_path.read_text(encoding="utf-8", errors="replace")
    if strip_scripts:
        doc = re.sub(r"<script\b.*?</script>", "", doc, flags=re.S | re.I)
    inject = "<style>@page{size:A4 landscape;margin:6mm;}</style>"
    doc = re.sub(r"(<head[^>]*>)", r"\1" + inject, doc, count=1, flags=re.I) if "<head" in doc \
        else inject + doc
    wrapped.write_text(doc, encoding="utf-8")
    ok = print_pdf(wrapped, out_pdf)
    wrapped.unlink(missing_ok=True)
    return ok


def pdf_ok(pdf: Path, r) -> tuple[bool, str]:
    txt = pdf_text(pdf)
    low = txt.lower()
    m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{4})", r["qe"])
    dd_mm = f"{m.group(1)}-{MONTHS[m.group(2).upper()]:02d}-{m.group(3)}"
    checks = {"text_nonempty": bool(txt.strip()),
              "total_income": "total income" in low,
              "quarter_end_date": (re.sub(r"\s", "", re.escape(dd_mm)), ) and (
                  re.search(re.escape(dd_mm), txt, re.I)
                  or re.search(re.escape(f"{m.group(1)}-{m.group(2)}-{m.group(3)}"), txt, re.I)) is not None}
    bad = [k for k, v in checks.items() if not bool(v)]
    return (not bad, ",".join(bad) if bad else "all-pass")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--company", required=True, help="company_code folder (registry key)")
    a = ap.parse_args()
    code = a.company
    reg = load_registry().get(code)
    if not reg:
        raise SystemExit(f"company_code '{code}' not in registry -- stopping")
    ticker = reg["ticker"]
    SCRATCH.mkdir(exist_ok=True)
    company_dir = INBOX / code
    company_dir.mkdir(exist_ok=True)
    rows = load_rows(code)
    # rule: a Revision is ingested only if its content hash differs from the
    # Original of the same (quarter_end, consolidated/standalone) -- so every
    # Original must be hashed BEFORE the comparisons -> originals first.
    rows = sorted(load_rows(code),
                  key=lambda r: (r["variant"].strip().lower() != "original",))
    log(f"[pipeline] {len(rows)} ixbrl rows for {code} (ticker {ticker})")

    manifest_changes: dict[str, dict] = {}
    hash_by_orig: dict[tuple[str, str], str] = {}
    failures, done = [], []

    for r in rows:
        try:
            hpath = fetch_if_missing(r, ticker)
        except SystemExit as e:
            log(f"[pipeline] STOP: {e}")
            return 2
        m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{4})", r["qe"])
        qe_d = date(int(m.group(3)), MONTHS[m.group(2).upper()], int(m.group(1)))
        fy = qe_d.year + 1 if qe_d.month >= 4 else qe_d.year          # FY end (Apr-Mar)
        qtr = QMOVED[qe_d.month]
        hdr = header_fields(hpath.read_text(encoding="utf-8", errors="replace"))
        mism = []
        if hdr["symbol"] and hdr["symbol"] != ticker: mism.append(f"symbol {hdr['symbol']}")
        if hdr["nature"] and r["con"].lower() not in hdr["nature"].lower(): mism.append("nature")
        if hdr["audited"] and ("un" in hdr["audited"].lower()) != ("un" in r["audited"].lower()):
            mism.append("audited")

        seq = re.search(r"INDAS_(\d+)_", r["url"]).group(1)
        au = "A" if "un" not in r["audited"].lower() else "U"
        base = f"{ticker}_quarterly_{qtr}FY{fy}_{r['con'].lower()}_{au}_printed.pdf"
        is_rev = "revision" == r["variant"].strip().lower()
        if is_rev:
            base = base.replace("_printed.pdf", "_printed_revision.pdf")
        # content-key based resume: any existing PDF that matches this
        # filing's key+variant means the filing was already printed in a
        # prior run; never reprint the same filing.
        key_pat = re.compile(
            rf"^{re.escape(ticker)}_quarterly_{qtr}FY{fy}_{re.escape(r['con'].lower())}_{au}_printed")
        existing = [p for p in company_dir.glob("*_quarterly_*.pdf")
                    if key_pat.match(p.name) and ("revision" in p.name) == is_rev]
        big = hpath.stat().st_size > 1_000_000
        if existing:
            target = existing[0]
            name = target.name
            log(f"[pipeline] RESUME-SKIP print {name} (key already printed)")
        else:
            name = base
            target = company_dir / name
            okp = big_print(hpath, target) if big else print_pdf(hpath, target)
            if not okp and big:                       # single-shot fallback
                okp = print_pdf(hpath, target, strip_scripts=True)
            if not okp:
                failures.append((name, "print-failed"))
                log(f"[pipeline] PRINT-FAIL {name}")
                continue
        good, reason = pdf_ok(target, r)
        if not good:                                  # one landscape re-print on check failure
            okp = landscape_print(hpath, target, strip_scripts=big)
            good, reason2 = pdf_ok(target, r)
            reason += "|landscape:" + reason2

        thash = text_hash(target)
        hhash = html_norm_hash(hpath)
        srckey = (r["qe"], r["con"].lower())
        if is_rev:
            oh = hash_by_orig.get(srckey)
            if oh is not None and hhash == oh:
                manifest_changes[name] = {"file": name, "action": "withdraw",
                                          "reason": f"identical content to Original "
                                                    f"(strip-tag hash {hhash}); revision rule (row seq {seq})"}
                log(f"[pipeline] HASH-DUP revision -> withdrawn {name} (seq {seq})")
            # else: revision genuinely differs from the Original -> kept
        else:
            hash_by_orig.setdefault(srckey, hhash)
        done.append({"file": name, "qe": r["qe"], "qtr": qtr, "fy": fy,
                     "con": r["con"], "audited": r["audited"], "variant": r["variant"],
                     "seq": seq, "checks_ok": good, "checks": reason,
                     "text_hash": thash, "mismatch": ",".join(mism) or "-",
                     "bigfile": big})
        log(f"[pipeline] {'OK ' if good else 'CHECK-FAIL'} {name} checks={reason} mism={','.join(mism) or '-'}")
        time.sleep(1)

    if manifest_changes:
        mp = company_dir / "manifest.json"
        cur = json.loads(mp.read_text()) if mp.exists() else []
        cur = [e for e in (cur if isinstance(cur, list) else []) if e.get("file") not in manifest_changes]
        cur += list(manifest_changes.values())
        mp.write_text(json.dumps(cur, indent=1))
        log(f"[pipeline] manifest.json updated (+{len(manifest_changes)} withdrawn)")

    with (INBOX / "_run_log.txt").open("a", encoding="utf-8") as f:
        f.write(f"[{date.today().isoformat()}] pipeline {code}: {len(done)} printed, {len(failures)} print-failures\n")
        for d in done:
            f.write("  " + repr(d) + "\n")
        for name, why in failures:
            f.write(f"  FAILURE {name}: {why}\n")
    log(f"[pipeline] done printed={len(done)} failures={len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
