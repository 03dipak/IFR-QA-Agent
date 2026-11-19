"""Snippet normalisation + hashing — copied from the EDGAR capstone's gold tooling.

Provenance (copied verbatim, 2026-11-08):
    /home/dipak/agentic/DataOSS_Capstone_financial_reporting/src/collector/gold/values.py
    functions ``normalise_for_hash`` and ``snippet_hash``.

Why copied rather than re-implemented: the scorer and any verifier must apply
the IDENTICAL normalisation (NFC -> LF -> whitespace collapse -> lowercase) and
the same snippet hash, or a row verifies in preparation and fails at scoring
(the prior failure mode this module documents: checker vs scorer disagreement).
The docstring rationale is kept as written at the source.

Only these two functions were copied. ``resolve_value`` and ``prefill_traps``
from the same module were NOT copied: they encode XBRL scale/decimals semantics
and USD magnitude traps, which do not map to this corpus (Indian filings print
units as text, typically "Rs. in crore/lakh").
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalise_for_hash(text: str) -> str:
    """NFC → LF → collapse whitespace → lowercase.

    Must be the SAME normalisation the scorer's E7 applies (whitespace and case
    normalisation, decided 2026-10-17). If the pack generator and the scorer ever
    normalise differently, a row verifies at preparation and fails at scoring —
    `lld_eval.md` §5.3.
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return _WS.sub(" ", text).strip().lower()


def snippet_hash(snippet: str) -> str:
    """`sha256:` over the normalised snippet. ADR-005 §8.2."""
    return "sha256:" + hashlib.sha256(normalise_for_hash(snippet).encode("utf-8")).hexdigest()
