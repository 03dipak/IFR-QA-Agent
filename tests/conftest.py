"""Guard canonical data files from test mutation (gold_data, gate_cache)."""
from __future__ import annotations
from pathlib import Path
import pytest

_GUARDED = [
    Path("data/eval/datasets/gold_data.jsonl"),
    Path("data/corpus/gate_cache.json"),
]

@pytest.fixture(autouse=True)
def _guard_canonical_files():
    before = {p: p.read_bytes() if p.exists() else None for p in _GUARDED}
    yield
    for p, b in before.items():
        if b is None:
            if p.exists():
                p.unlink()
        else:
            if not p.exists() or p.read_bytes() != b:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
