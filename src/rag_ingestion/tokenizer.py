"""Versioned tokenizer shared by index build and query (lld_retrieval.txt §4).

One function, one version. Numeric-preserving rules:
  * NFC + lowercase
  * Indian grouping ``1,23,456.78`` collapses to ``123456.78``
  * bracket negatives ``(1,234)`` fold to ``-1234``
  * decimals keep their point (``1.25`` != ``125``)
  * lakh / crore remain tokens on the question side
"""

from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = "tokenizer.v1"

_NUM_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")
_TOKEN_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?|[a-z0-9]+")


def _canonical_number(tok: str) -> str:
    s = tok.strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace(",", "")
    if neg and not s.startswith("-"):
        s = "-" + s
    return s


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if _NUM_RE.fullmatch(tok):
            out.append(_canonical_number(tok))
        else:
            out.append(tok)
    # Emit bare-year alternates for FY/quarter tokens.
    for tok in list(out):
        m = re.fullmatch(r"(?:q[1-4])?fy(\d{4})", tok)
        if m:
            yr = m.group(1)
            if yr not in out:
                out.append(yr)
    return out
