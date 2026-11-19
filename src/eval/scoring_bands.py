"""Tolerance bands + normalisation rules (docs/llds/lld_normalisation.txt
section 5, money bands pinned 2026-11-08). Source of truth: data/eval/scoring_bands.json.

Before any freeze and before any scored run, the fixture must be loaded, its
hash computed and recorded in the split manifest alongside the corpus
freeze_v1 sha256 and registry_version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .gold_schema import canonical

PATH = Path("data/eval/scoring_bands.json")


def load() -> dict:
    return json.load(PATH.open(encoding="utf-8"))


def bands_hash() -> str:
    return hashlib.sha256(canonical(load())).hexdigest()


def normalisation_rules_version() -> str:
    return load()["normalisation_rules_version"]


def registry_version() -> str:
    return load()["registry_version"]
