"""Deterministic slot resolver: company, period, basis, metric.

No model calls. Company aliases come from data/acquisition/company_registry.csv.
Period keys follow the Indian fiscal year (Apr-Mar): FYnn annual, QnFYnn
quarterly, HnFYnn half-year, 9MFYnn nine months.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.entity_registry import normalize_company_name

DEFAULT_REGISTRY = "data/acquisition/company_registry.csv"

# Ind AS line-item synonym classes. Longest phrase wins.
METRIC_SYNONYMS = {
    "revenue from operations": "revenue_from_operations",
    "total income": "total_income",
    "total expenses": "total_expenses",
    "profit before tax": "profit_before_tax",
    "profit for the period": "net_profit",
    "profit for the year": "net_profit",
    "profit after tax": "net_profit",
    "net profit": "net_profit",
    "pat": "net_profit",
    "total assets": "total_assets",
    "total equity": "total_equity",
    "total liabilities": "total_liabilities",
    "earnings per share": "earnings_per_share",
    "eps": "earnings_per_share",
    "ebitda": "ebitda",
    "operating profit": "operating_profit",
    "gross profit": "gross_profit",
    "net sales": "net_sales",
    "sales": "sales",
}

_FY_RE = re.compile(r"\bFY\s*(\d{2}|\d{4})\b", re.I)
_Q_RE = re.compile(r"\bQ\s*([1-4])\s*(?:FY\s*)?(\d{2}|\d{4})\b", re.I)
_H_RE = re.compile(r"\bH\s*([1-2])\s*(?:FY\s*)?(\d{2}|\d{4})\b", re.I)
_NINE_RE = re.compile(r"\bnine\s+months?\b(?:\s+(?:ended|for|FY)\s*(\d{4}))?", re.I)
_NINE_ENDED_RE = re.compile(
    r"\bnine\s+months?\s+ended\s+\d{1,2}\s+(January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+(\d{4})\b", re.I)
_YEAR_ENDED_RE = re.compile(
    r"\byear\s+ended\s+\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(\d{4})\b", re.I)
_BASIS_RE = re.compile(r"\b(standalone|consolidated|combined|single)\b", re.I)


def _fy_year(raw: str) -> int:
    y = int(raw)
    if y < 100:
        y += 2000
    return y


@dataclass
class PeriodSlot:
    token: str
    period_type: str
    fiscal_year: int
    start: str
    end: str
    source: str

    def as_dict(self) -> dict:
        return {"token": self.token, "period_type": self.period_type,
                "fiscal_year": self.fiscal_year, "start": self.start,
                "end": self.end, "source": self.source}


@dataclass
class ResolutionResult:
    company_ids: list[str] = field(default_factory=list)
    periods: list[PeriodSlot] = field(default_factory=list)
    basis: str | None = None
    basis_raw: str | None = None
    basis_policy: str = "unspecified"
    metric_id: str | None = None
    question: str = ""

    def as_dict(self) -> dict:
        return {
            "company_ids": self.company_ids,
            "periods": [p.as_dict() for p in self.periods],
            "basis": self.basis,
            "basis_raw": self.basis_raw,
            "basis_policy": self.basis_policy,
            "metric_id": self.metric_id,
        }


class CompanyRegistry:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.by_id = {r["code"]: r for r in rows if r.get("code")}
        self._aliases: list[tuple[str, str]] = []
        for r in rows:
            cid = r.get("code")
            if not cid:
                continue
            surfaces = [r.get("code", ""), r.get("display_name", ""),
                        r.get("bse_ticker", "")]
            surfaces += [a for a in str(r.get("aliases") or "").split("|") if a]
            for s in surfaces:
                s = (s or "").strip()
                if s:
                    self._aliases.append((s, cid))
        # longest surface first so "Reliance Industries" beats "Reliance"
        self._aliases.sort(key=lambda x: len(x[0]), reverse=True)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _fuzzy(self, surface: str, q_tokens: list[str]) -> bool:
        common = {"company", "companies", "industries", "corporation", "limited",
                  "ltd", "inc", "energy", "india"}
        core = [t for t in self._tokens(surface) if t not in common]
        if not core:
            return False
        n = len(core)
        for i in range(len(q_tokens) - n + 1):
            window = q_tokens[i:i + n]
            ok = True
            for qt, at in zip(window, core):
                if qt == at or (len(qt) >= 4 and (at.startswith(qt) or qt.startswith(at))):
                    continue
                ok = False
                break
            if ok:
                return True
        return False

    def resolve(self, question: str) -> list[str]:
        text = question or ""
        q_tokens = self._tokens(text)
        found: list[str] = []
        spans: list[tuple[int, int]] = []
        for surface, cid in self._aliases:
            if not surface:
                continue
            pat = re.escape(surface)
            if surface[0].isalnum():
                pat = r"\b" + pat
            if surface[-1].isalnum():
                pat = pat + r"\b"
            m = re.search(pat, text, re.I)
            if not m and not self._fuzzy(surface, q_tokens):
                continue
            if m:
                span = m.span()
                if any(span[0] < e and span[1] > s for s, e in spans):
                    continue
                spans.append(span)
            if cid not in found:
                found.append(cid)
        return found


def load_company_registry(path: str | Path = DEFAULT_REGISTRY) -> CompanyRegistry:
    with open(path, newline="", encoding="utf-8") as fh:
        return CompanyRegistry(list(csv.DictReader(fh)))


def resolve_companies(question: str, registry: CompanyRegistry | None = None) -> list[str]:
    registry = registry or load_company_registry()
    return registry.resolve(question)


def _period_from_year(year: int, ptype: str, source: str) -> PeriodSlot:
    if ptype == "annual":
        return PeriodSlot(f"FY{year}", "annual", year,
                          f"{year-1}-04-01", f"{year}-03-31", source)
    if ptype == "quarter":
        q = int(source)
        starts = {1: (4, 1), 2: (7, 1), 3: (10, 1), 4: (1, 1)}
        ends = {1: (6, 30), 2: (9, 30), 3: (12, 31), 4: (3, 31)}
        sm, sd = starts[q]
        em, ed = ends[q]
        sy = year if q != 4 else year - 1
        return PeriodSlot(f"Q{q}FY{year}", "quarter", year,
                          f"{sy}-{sm:02d}-{sd:02d}", f"{year}-{em:02d}-{ed:02d}",
                          f"Q{q}FY{year}")
    if ptype == "half":
        h = int(source)
        if h == 1:
            return PeriodSlot(f"H1FY{year}", "half_year", year,
                              f"{year-1}-04-01", f"{year-1}-09-30", f"H1FY{year}")
        return PeriodSlot(f"H2FY{year}", "half_year", year,
                          f"{year-1}-10-01", f"{year}-03-31", f"H2FY{year}")
    if ptype == "nine_month":
        return PeriodSlot(f"9MFY{year}", "nine_month", year,
                          f"{year-1}-04-01", f"{year-1}-12-31", f"9MFY{year}")
    raise ValueError(ptype)


def resolve_period(question: str) -> list[PeriodSlot]:
    q = question or ""
    out: list[PeriodSlot] = []
    seen: set[str] = set()
    taken: list[tuple[int, int]] = []

    def add(slot: PeriodSlot):
        if slot.token not in seen:
            seen.add(slot.token)
            out.append(slot)

    for m in _Q_RE.finditer(q):
        taken.append(m.span())
        year = _fy_year(m.group(2))
        add(_period_from_year(year, "quarter", m.group(1)))
    for m in _H_RE.finditer(q):
        taken.append(m.span())
        year = _fy_year(m.group(2))
        add(_period_from_year(year, "half", m.group(1)))
    for m in _NINE_ENDED_RE.finditer(q):
        taken.append(m.span())
        months = {m2: i + 1 for i, m2 in enumerate(
            ["january", "february", "march", "april", "may", "june", "july",
             "august", "september", "october", "november", "december"])}
        month = months[m.group(1).lower()]
        year = int(m.group(2))
        fy = year if month <= 3 else year + 1
        add(_period_from_year(fy, "nine_month", "9M"))
    for m in _NINE_RE.finditer(q):
        if any(m.start() < e and m.end() > s for s, e in taken):
            continue
        taken.append(m.span())
        year = int(m.group(1)) if m.group(1) else None
        if year is None:
            fy = _FY_RE.search(q)
            year = _fy_year(fy.group(1)) if fy else None
        if year:
            add(_period_from_year(year, "nine_month", "9M"))
    for m in _FY_RE.finditer(q):
        if any(m.start() < e and m.end() > s for s, e in taken):
            continue
        taken.append(m.span())
        year = _fy_year(m.group(1))
        add(_period_from_year(year, "annual", f"FY{year}"))
    for m in _YEAR_ENDED_RE.finditer(q):
        year = int(m.group(1))
        add(_period_from_year(year, "annual", f"year ended {year}"))
    return out


def resolve_basis(question: str) -> tuple[str | None, str | None, str]:
    m = _BASIS_RE.search(question or "")
    if not m:
        return None, None, "unspecified"
    raw = m.group(1).lower()
    if raw == "consolidated":
        return "consolidated", raw, "explicit"
    if raw in ("standalone", "combined", "single"):
        return "standalone", raw, "explicit"
    return None, raw, "unspecified"


def resolve_metric(question: str) -> str | None:
    q = (question or "").lower()
    for phrase, metric in sorted(METRIC_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(phrase) + r"\b", q):
            return metric
    return None


def resolve_query(question: str, registry: CompanyRegistry | None = None) -> ResolutionResult:
    basis, raw, policy = resolve_basis(question)
    return ResolutionResult(
        company_ids=resolve_companies(question, registry),
        periods=resolve_period(question),
        basis=basis,
        basis_raw=raw,
        basis_policy=policy,
        metric_id=resolve_metric(question),
        question=question,
    )
