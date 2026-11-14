"""Content-based period detection for manual inbox PDFs.

Weighted scoring of candidate period-end dates:

- permissive phrase match "<unit>[ and <unit>]* ended [on]", plus "ending"
  and "as at"; tolerant date parsing ("quarter ended 31st December 2023",
  "year ended March 31,2023", "31 st March, 2024", "30.09.2024", ...)
- candidates come from the first five pages, weighted:
    statement title sentence / cover-letter Sub|Re|Reg line  +6
    ordinary phrase occurrence  +2 (same-date repeats reinforce)
    table column header 0; NOTES paragraph -2; prior-period cue -3
- sanity window vs filing date (letter / e-signature / PDF metadata;
  never the filename): period end 0-90 days before filing, 0-150 for
  year-end filings
- winner >= 2x runner-up weight -> high, less -> medium, tie -> unparsed
  with a suggested_period; no phrase match -> suggested quarter-per-window,
  low, unparsed. The suggested period is never auto-ingested.

Bundle PDFs (letter + review + statements + press release) are classified
by the highest-value content present; "quarter and year ended 31 March"
yields a Q4 period with covers_full_year=true and doc_type=annual. Tests
use stored text snapshots of the first pages, not whole PDFs.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
          r"july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
          r"dec(?:ember)?)")
_SEP1 = r"[\s.,\-/'’]+"
_SUPER = r"(?:st|nd|rd|th)?"
_YEAR = r"(\d{4}|\d{3}\s?\d|\d{2})(?!\d)"

# tolerant date variants: ordinals incl. stray web text ("31 st March"),
# missing separator ("March 31,2023"), dd/mm/yyyy, dd.mm.yyyy, split year
# ("202 3"), apostrophe year ("31st Dec' 2023", "Dec'23")
_DMY = rf"(\d{{1,2}})\s*{_SUPER}{_SEP1}({_MONTH}){_SEP1}?{_YEAR}"
_MDY = rf"({_MONTH}){_SEP1}?(\d{{1,2}})\s*{_SUPER},?{_SEP1}?{_YEAR}"
_NUM = r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})(?!\d)"
# day-less apostrophe form ("for the quarter ended Dec'23") -> month end
_MOY = rf"({_MONTH})\s*['’]\s?(\d{{2,4}})(?!\d)"

_MONTH_DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
               7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

# "<n>th/ordinal quarter", "third quarter" (ordinal quarters)
_ORDQ = r"(?:\d{1,2}\s*(?:st|nd|rd|th)|first|second|third|fourth)\s+quarter"

_UNITS = (rf"(?:three\s+months|\d{{1,2}}\s+months|{_ORDQ}|"
          r"quarter|(?:six|half)[ \-]?year|nine\s+months|period|year)")

_PHRASE_END = re.compile(
    rf"\b(?:{_UNITS})(?:(?:\s+and\s+|\s*&\s*|\s*,\s*)(?:{_UNITS}))*"
    rf"(?:\s+period)?\s*ended(?:\s+on)?(?![a-zA-Z])", re.I)
_PHRASE_ENDING = re.compile(r"\bending(?:\s+on)?(?![a-zA-Z])", re.I)
_PHRASE_ASAT = re.compile(r"\bas\s+at(?![a-zA-Z])", re.I)

_SUBJECT_RE = re.compile(
    r"^\s*(?:sub(?:ject)?\s*[:.\-]?|re\s*[:.\-]?|reg(?:arding|ference)?\s*"
    r"[:.\-]?|ref\s*[:.\-]?\.?)", re.I)
_STATEMENT_RE = re.compile(r"statement of|financial results", re.I)
_HEADING_RE = re.compile(
    r"(?i)^\s*statement of\b.{0,140}?financial results", re.M)
_LETTER_RE = re.compile(
    r"outcome of (?:the\s+)?board meeting|intimation|press release", re.I)
_OTHER_REPORT_RE = re.compile(r"business responsibility|secretarial audit", re.I)
_CUE_RE = re.compile(
    r"previous|prior|corresponding|comparative|balancing figures|"
    r"refer(?:red)? note|were reviewed|as reported|basis of preparation", re.I)
_NOTES_MARK = re.compile(r"^\s*notes?\s*:?\s*$", re.I | re.M)
_DATE_TOKEN_RE = re.compile(
    rf"(?:{_DMY}|{_MDY}|{_NUM})", re.I)

_W_TITLE = 6
_W_SUBJECT = 6
_W_BODY = 2
_W_HEADER = 0
_W_CUE = -3
_W_NOTES = -2

_TITLE_WINDOW = 160   # chars above a phrase hit that count as title lines
_QUARTER_DAYS = 90
_YEAR_DAYS = 150


# ---------------------------------------------------------------------------
# date and period mapping
# ---------------------------------------------------------------------------

def _valid_date(year: int, month: int, day: int):
    if year < 100:
        year += 2000
    if not (1990 <= year <= 2100):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _year_int(raw: str) -> int:
    return int("".join(str(raw).split()))


def _dates_from_span(flat: str, lo: int, hi: int):
    """Dates whose match starts in [lo, hi); tokens are parsed against the
    full text so window cuts never truncate. Sorted by position."""
    hits = []
    for rx, unpack in (
        (_DMY, lambda m: (int(m.group(1)), MONTHS[m.group(2).lower()[:3]],
                          _year_int(m.group(3)))),
        (_MDY, lambda m: (int(m.group(2)), MONTHS[m.group(1).lower()[:3]],
                          _year_int(m.group(3)))),
        (_NUM, lambda m: None),
        (_MOY, lambda m: None),
    ):
        for m in re.finditer(rx, flat, re.I):
            if not (lo <= m.start() < hi):
                continue
            if rx is _NUM:
                a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if b > 12 >= a:
                    day_num, month = b, a
                else:
                    day_num, month = a, b
            elif rx is _MOY:
                month = MONTHS[m.group(1).lower()[:3]]
                y = _year_int(m.group(2))
                if y < 100:
                    y += 2000
                day_num = _MONTH_DAYS[month]  # month-end for "Dec'23" style
            else:
                day_num, month, y = unpack(m)
            if not (1 <= day_num <= 31 and 1 <= month <= 12):
                continue
            d = _valid_date(y, month, day_num)
            while d is None and day_num > 28:  # clamp Feb 30/31 to month end
                day_num -= 1
                d = _valid_date(y, month, day_num)
            if d:
                hits.append((m.start(), m.end(), d))
    # greedy earliest-start / longest-match: overlapping regex variants of
    # the same token (e.g. split years) must not yield phantom dates
    kept: list = []
    for start, end, d in sorted(hits, key=lambda h: (h[0], -(h[1] - h[0]))):
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, d))
    return [d for _, _, d in kept]


def _fy_period(kind: str, d: date, covers_full_year: bool):
    """Indian FY (Apr-Mar): Q1=Apr-Jun ... Q4=Jan-Mar, FY ends in March."""
    if kind in ("year", "year_q") and d.month == 3:
        # plain "year ended March" -> FY; "quarter and year" -> Q4 of that FY
        if covers_full_year:
            return "Q4FY" + str(d.year), "annual"
        return f"FY{d.year}", "annual"
    q_map = {4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2,
             10: 3, 11: 3, 12: 3, 1: 4, 2: 4, 3: 4}
    q = q_map[d.month]
    fy = d.year + 1 if d.month >= 4 else d.year
    return f"Q{q}FY{fy}", ("annual" if kind == "year" else None)


def _quarter_in_window(filing_dates):
    """No phrase at all: the quarter end within 0-90 days before filing."""
    best = None
    for f in filing_dates:
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            y = f.year if (f.month, f.day) >= (month, day) else f.year - 1
            d = _valid_date(y, month, day)
            if d and 0 <= (f - d).days <= _QUARTER_DAYS \
                    and (best is None or d > best):
                best = d
    if best is None:
        return None
    return _fy_period("quarter", best, False)[0]


# ---------------------------------------------------------------------------
# text layout helpers
# ---------------------------------------------------------------------------

def _page_scopes(text: str):
    """Pages joined with '\\f' kept on its own line; returns (flat, bounds)."""
    raw_pages = (text or "").split("\f") if "\f" in (text or "") else [(text or "")]
    blocks, bounds, pos = [], [], 0
    for p in raw_pages:
        block = "\n".join(" ".join(ln.split()) for ln in p.splitlines())
        blocks.append(block)
        bounds.append((pos, pos + len(block)))
        pos += len(block) + 3  # "\n\f\n"
    return "\n\f\n".join(blocks), bounds


def _line_bounds(flat: str):
    starts = [0]
    for m in re.finditer(r"\n", flat):
        starts.append(m.end())
    return starts


def _line_at(flat: str, bounds, offset: int) -> str:
    import bisect
    i = bisect.bisect_right(bounds, offset) - 1
    if i < 0:
        i = 0
    end = bounds[i + 1] - 1 if i + 1 < len(bounds) else len(flat)
    return flat[bounds[i]:end].replace("\f", " ")


def _is_column_header(line: str) -> bool:
    if len(line) > 200:  # header rows are short; guard prose sentences
        return False
    low = " ".join(line.lower().split())
    if "particulars" in low:
        return True
    numeric_dates = len(re.findall(r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}", low))
    dates = len(_DATE_TOKEN_RE.findall(low))
    period_word = re.search(r"quarter|year|month|period|half|six|nine", low)
    if not period_word:
        return False
    if numeric_dates >= 2 and re.search(r"ended", low):
        return True
    if dates >= 3 and re.search(r"ended", low):
        return True
    return False


def _filing_dates_from_text(flat: str):
    """Letter / e-signature / reference dates in the text (never filename)."""
    cands: list = []
    for m in re.finditer(
            r"(?i)\b(?:dated|date\s*[:\-]|e[- ]?signature|place\s*[:\-]|"
            r"sent on|issued on|dated\s*:)\s*,?\s*", flat):
        for d in _dates_from_span(flat, m.end(), m.end() + 60)[:3]:
            cands.append(d)
    cover = flat.split("\n\f\n", 1)[0] if "\n\f\n" in flat else flat
    taken = 0
    for m in _DATE_TOKEN_RE.finditer(cover):
        if m.start() > 400:
            break
        # a letterhead date is not the tail of an "ended <date>" phrase
        prefix = cover[max(0, m.start() - 60):m.start()]
        if re.search(r"(?i)end(?:ed|ing)\s*[\W_]{0,12}$", prefix):
            continue
        d = _cover_date(m)
        if d is not None:
            cands.append(d)
            taken += 1
        if taken >= 2:
            break
    return cands


def _cover_date(m):
    """Parse the token found by _DATE_TOKEN_RE in the cover into a date."""
    text = m.group(0)
    for rx in (_DMY, _MDY, _NUM):
        m2 = re.search(rx, text, re.I)
        if not m2:
            continue
        if rx is _NUM:
            a, b, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            day_num, month = (b, a) if b > 12 >= a else (a, b)
        elif rx is _DMY:
            day_num = int(m2.group(1))
            month = MONTHS[m2.group(2).lower()[:3]]
            y = int(m2.group(3))
        else:
            day_num = int(m2.group(2))
            month = MONTHS[m2.group(1).lower()[:3]]
            y = int(m2.group(3))
        return _valid_date(y, month, day_num)
    return None


def pdf_metadata_dates(path) -> list:
    """Creation/Modified dates from the PDF metadata (best effort)."""
    try:
        from pypdf import PdfReader
        meta = PdfReader(str(path)).metadata or {}
    except Exception:
        return []
    out = []
    for key in ("/CreationDate", "/ModDate"):
        raw = str(meta.get(key) or "")
        m = re.match(r"D?:?(\d{4})(\d{2})(\d{2})", raw)
        if m:
            d = _valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if d:
                out.append(d)
    return out


def _kind_of(match_text: str):
    low = " ".join(match_text.lower().split())
    is_year = re.search(r"year", low)
    is_quarterly = re.search(r"quarter|three\s+months|\d{1,2}\s+months", low)
    if re.search(r"half|six", low):
        return "half_year", False
    if re.search(r"nine", low):
        return "nine_months", False
    if is_year and is_quarterly:
        return "year_q", True
    if is_year:
        return "year", False
    return "quarter" if is_quarterly else "period", False


def _blank(reason: str) -> dict:
    return {"period": None, "doc_type": None, "variant": None,
            "audit_status": "unclear", "kind": None, "confidence": "low",
            "snippet": "", "dates_found": [], "candidates": "",
            "covers_full_year": False, "suggested_period": None,
            "reason": reason}


# ---------------------------------------------------------------------------
# occurrences and the scorer
# ---------------------------------------------------------------------------

def _occurrences(flat: str, bounds, cutoff=None):
    occ = []
    for rx in (_PHRASE_END, _PHRASE_ENDING, _PHRASE_ASAT):
        for m in rx.finditer(flat):
            if cutoff is not None and m.start() >= cutoff:
                continue
            line = _line_at(flat, bounds, m.start())
            sentence = flat[max(0, m.start() - 140):m.start() + len(m.group(0)) + 140]
            if rx is _PHRASE_END:
                kind, covers = _kind_of(m.group(0))
            else:
                kind, covers = "period", False
            dates = _dates_from_span(flat, m.end(), m.end() + 60)
            if not dates:
                continue
            if _is_column_header(line):
                weight = _W_HEADER
            elif _CUE_RE.search(sentence):
                weight = _W_CUE
            elif _SUBJECT_RE.search(line) and rx is _PHRASE_END:
                weight = _W_SUBJECT
            elif rx is _PHRASE_END and _STATEMENT_RE.search(
                    flat[max(0, m.start() - _TITLE_WINDOW):m.end()]):
                weight = _W_TITLE
            else:
                weight = _W_BODY
            occ.append({
                "offset": m.start(), "kind": kind, "covers": covers,
                "date": dates[0], "line": line, "weight": weight,
                "snippet": line.strip()[:200] if weight >= _W_BODY else
                           f"{m.group(0).strip()[:40]} {dates[0].isoformat()}",
            })
    return occ


def detect_period_from_content(text: str, meta_dates=None) -> dict:
    """Weighted scorer over candidate period-end dates (module docstring)."""
    clean = text or ""
    flat, scope_bounds = _page_scopes(clean)
    if not flat.strip():
        return _blank("empty text")
    bounds = _line_bounds(flat)

    notes_regions = []
    for m in _NOTES_MARK.finditer(flat):
        page_end = next((b[1] for b in scope_bounds if b[0] <= m.start() < b[1]),
                        len(flat))
        notes_regions.append((m.start(), page_end))

    cutoff = scope_bounds[4][1] if len(scope_bounds) > 4 else None
    occ = _occurrences(flat, bounds, cutoff)
    for o in occ:  # NOTES paragraph member after the heading line
        if o["weight"] == _W_HEADER and any(lo <= o["offset"] < hi
                                            for lo, hi in notes_regions):
            o["weight"] = _W_NOTES

    filing_dates = _filing_dates_from_text(flat) + list(meta_dates or [])

    def in_window(d: date, covers: bool) -> bool:
        if not filing_dates:
            return True
        span = _YEAR_DAYS if covers else _QUARTER_DAYS
        return any(0 <= (f - d).days <= span for f in filing_dates)

    scored: dict = {}
    for o in occ:
        slot = scored.setdefault(o["date"], {"weight": 0, "hits": 0,
                                             "covers": False, "top": None})
        slot["weight"] += o["weight"]
        slot["hits"] += 1
        slot["covers"] = slot["covers"] or o["covers"]
        if o["weight"] >= _W_BODY and slot["top"] is None:
            slot["top"] = o

    for d in [d for d, v in scored.items() if not in_window(d, v["covers"])]:
        del scored[d]

    cands_view = "; ".join(
        f"{d.isoformat()}={v['weight']}"
        for d, v in sorted(scored.items(), key=lambda kv: (-kv[1]["weight"], kv[0])))

    def empty_out(reason: str, suggested=None) -> dict:
        out = _blank(reason)
        out["suggested_period"] = suggested
        out["candidates"] = cands_view
        out["dates_found"] = sorted(d.isoformat() for d in scored)
        return out

    if not scored:
        return empty_out(
            "no period-end phrase with a usable date in first pages "
            "(filing dates: "
            + (", ".join(str(d) for d in filing_dates[:3]) or "none") + ")",
            _quarter_in_window(filing_dates))

    ordered = sorted(scored.items(), key=lambda kv: (-kv[1]["weight"], kv[0]))
    top_date, top = ordered[0]
    runner = ordered[1][1]["weight"] if len(ordered) > 1 else 0

    if top["weight"] <= 0:
        # only headers/notes/cue hits: no clean phrase match, never guess
        best_date = max(scored, key=lambda d: (scored[d]["weight"], d))
        suggested = _fy_period("quarter", best_date, False)[0]
        return empty_out("no positively-scored period-end date "
                         f"(candidates: {cands_view})", suggested)

    if len(ordered) > 1 and ordered[0][1]["weight"] == ordered[1][1]["weight"]:
        suggested, _ = _fy_period(top["top"]["kind"], top_date, top["covers"])
        return empty_out("tied top candidates: " + cands_view, suggested)

    anchor = top["top"]
    period, dtype_annual = _fy_period(anchor["kind"], top_date, top["covers"])
    confidence = "high" if (len(ordered) == 1
                            or top["weight"] >= 2 * max(runner, 1)) else "medium"

    if _HEADING_RE.search(flat):
        doc_type = "annual" if dtype_annual == "annual" else "quarterly"
    elif _LETTER_RE.search(flat):
        doc_type = "disclosure"
    elif anchor["kind"] in ("quarter", "period", "half_year", "nine_months"):
        doc_type = "quarterly"
    elif _OTHER_REPORT_RE.search(flat):
        doc_type = "other_report"
    else:
        doc_type = "other_report"

    # variant from the anchor's surrounding sentence (the phrase and the
    # standalone/consolidated words often sit on separate lines) plus the
    # other strong phrase-bearing subject/title lines
    ctx = flat[max(0, anchor["offset"] - 200):anchor["offset"] + 200]
    strong = " ".join(o["line"] for o in occ
                      if o["weight"] >= _W_TITLE or o is anchor)
    probe = (ctx + "\n" + strong).lower()
    has_sa, has_co = "standalone" in probe, "consolidated" in probe
    variant = ("combined" if has_sa and has_co else
               "standalone" if has_sa else
               "consolidated" if has_co else None)

    return {
        "period": period, "doc_type": doc_type, "variant": variant,
        "audit_status": ("unaudited" if "unaudited" in flat.lower()
                         else "audited" if "audited" in flat.lower()
                         else "unclear"),
        "kind": anchor["kind"], "confidence": confidence,
        "snippet": anchor["snippet"], "dates_found": [top_date.isoformat()],
        "candidates": cands_view, "covers_full_year": top["covers"],
        "suggested_period": None, "reason": None,
    }


parse_period_from_content = detect_period_from_content
classify_period = detect_period_from_content
