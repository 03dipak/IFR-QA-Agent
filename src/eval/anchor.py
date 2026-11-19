"""Contiguous-passage anchor — copied from the EDGAR capstone's gold tooling.

Provenance (copied verbatim, 2026-11-08):
    /home/dipak/agentic/DataOSS_Capstone_financial_reporting/src/collector/gold/spans.py
    function ``anchor`` (line 46). Rationale kept as written at the source:
    the anchor is the mechanical fact that makes an evidence pack trustworthy
    (MARKER FAIL catches off-by-one line numbers; NOT CONTIGUOUS catches a
    skipped blank line). Explicit raise, never assert.

Why copied: the evidence-pack discipline it enforces (verbatim contiguous
excerpts anchored to page text) applies unchanged to this corpus's page
extracts. Not copied: the rest of spans.py (XBRL-period attribution, statement
detection tuned to US-GAAP/XBRL line tables).
"""

from __future__ import annotations

from collections.abc import Sequence


def anchor(lines: Sequence[str], full: str, nums: Sequence[int], marker: str) -> str:
    """Extract a contiguous passage and prove it, with two independent checks.

    1. `MARKER FAIL` — the line numbers are off by one. The distinctive phrase you
       expected is not in the assembled text.
    2. `NOT CONTIGUOUS` — a blank line was skipped between anchors.
    """
    try:
        text = "".join(lines[i - 1] for i in nums).rstrip("\n")
    except IndexError as exc:
        raise ValueError(f"line numbers out of range: {list(nums)} ({exc})") from exc

    if marker not in text:
        raise ValueError(f"MARKER FAIL on lines {list(nums)}: {marker!r} not in {text[:70]!r}")
    if text not in full:
        raise ValueError(f"NOT CONTIGUOUS lines {list(nums)} — a blank line was skipped")
    return text
