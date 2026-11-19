"""Unit tests for the per-row freeze_sha field."""

from __future__ import annotations

from src.eval.gold_check import check_freeze_sha
from src.eval.gold_schema import FREEZE_SHA_V1, FREEZE_SHA_V2, freeze_sha_of, validate_row


def test_v14_rows_default_to_freeze_v1() -> None:
    assert freeze_sha_of({}) == FREEZE_SHA_V1
    assert freeze_sha_of({"freeze_sha": ""}) == FREEZE_SHA_V1


def test_v2_rows_keep_their_sha() -> None:
    assert freeze_sha_of({"freeze_sha": FREEZE_SHA_V2}) == FREEZE_SHA_V2


def test_validate_row_accepts_missing_freeze_sha() -> None:
    row = {"id": "G-999", "category": "single_fact_lookup", "status": "active",
           "review_status": "self_reviewed", "split": "dev", "evidence": [{"x": 1}]}
    problems = validate_row(row)
    assert not any("freeze_sha" in p for p in problems), problems


def test_check_freeze_sha_flags_unknown_and_stale() -> None:
    bad = [{"id": "A", "freeze_sha": "deadbeef"}, {"id": "B", "freeze_sha": FREEZE_SHA_V1,
                                                    "gold_set_version": "gold_v2.0"}]
    fails = check_freeze_sha(bad)
    assert {rid for rid, _ in fails} == {"A", "B"}
    ok = [{"id": "C"}, {"id": "D", "freeze_sha": FREEZE_SHA_V2, "gold_set_version": "gold_v2.0"}]
    assert check_freeze_sha(ok) == []
