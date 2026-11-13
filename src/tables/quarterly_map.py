"""NSE iXBRL quarterly layout: PL spans and revenue/PAT/EPS figures per basis."""

from __future__ import annotations

import re

from .statement_map import ProposedSpan

_LAYOUT_MARKER = re.compile(r"Integrated Filing|Nature of report", re.I)
_NATURE = re.compile(
    r"nature of report\s*(?:standalone or consolidated)?\s*(standalone|consolidated)", re.I)
_REVENUE = re.compile(r"revenue from operations\s*[:\s]*([\d,]+(?:\.\d+)?)", re.I)
_PAT = re.compile(r"total profit\s*\(loss\)\s*for period\s*[:\s]*([\d,]+(?:\.\d+)?)", re.I)
_EPS = re.compile(
    r"(?:basic|diluted) earnings \(loss\) per share[^0-9]{0,120}?([\d,]+(?:\.\d+)?)", re.I)


def _flat(pages: list[tuple[int, str]]) -> dict[int, str]:
    return {index: " ".join(text.split()) for index, text in pages}


def _basis_per_page(text: dict[int, str]) -> dict[int, str]:
    """Carry the last 'Nature of report' marker forward over the following pages."""
    out: dict[int, str] = {}
    basis = ""
    for index in sorted(text):
        match = _NATURE.search(text[index])
        if match:
            basis = match.group(1).lower()
        out[index] = basis
    return out


def _is_quarterly_layout(text: dict[int, str]) -> bool:
    return any(_LAYOUT_MARKER.search(value) for value in text.values())


def quarterly_spans(company: str, checksum: str,
                    pages: list[tuple[int, str]]) -> list[ProposedSpan]:
    """One PL span per basis present in an NSE iXBRL integrated quarterly filing."""
    text = _flat(pages)
    if not _is_quarterly_layout(text):
        return []
    basis_of = _basis_per_page(text)
    spans: list[ProposedSpan] = []
    for basis in ("standalone", "consolidated"):
        hits = [index for index in sorted(text) if basis_of[index] == basis
                and (_REVENUE.search(text[index]) or _PAT.search(text[index])
                     or _EPS.search(text[index]))]
        if len(hits) >= 2:
            spans.append(ProposedSpan(
                company=company, checksum=checksum, basis=basis, statement="PL",
                start_pdf_page=hits[0], end_pdf_page=hits[-1], status="proposed"))
    return spans


def quarterly_figures(pages: list[tuple[int, str]],
                      basis: str) -> dict[str, tuple[str, int]]:
    """Extract revenue/PAT/EPS as (value, pdf_page) for one basis."""
    text = _flat(pages)
    basis_of = _basis_per_page(text)
    out: dict[str, tuple[str, int]] = {}
    for field, pattern in (("revenue", _REVENUE), ("pat", _PAT), ("eps", _EPS)):
        for index in sorted(text):
            if basis_of[index] != basis:
                continue
            match = pattern.search(text[index])
            if match:
                out[field] = (match.group(1).replace(",", ""), index)
                break
    return out
