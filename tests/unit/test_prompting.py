import hashlib
import json
from unittest.mock import patch

import pytest

from src.eval.gold_rows import HoldoutAccessError, assert_no_holdout
from src.eval.prompting import params
from src.eval.prompting.context import pack_pages
from src.eval.prompting.inputs import load_dev_rows
from src.eval.prompting.scoring import score_prediction


def test_holdout_guard_raises():
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout([{"id": "G-1", "split": "holdout"}])


def test_dev_rows_are_active_dev_only():
    rows = load_dev_rows()
    from src.config import GOLD_PATH
    gold_rows = [json.loads(l) for l in open(GOLD_PATH) if l.strip()]
    expected = [r for r in gold_rows if r.get("split") == "dev" and r.get("status", "active") == "active"]
    assert len(rows) == len(expected)
    assert all(r.get("status", "active") == "active" for r in rows)
    assert all(r.get("split") == "dev" for r in rows)
    assert all(r.get("id") for r in rows)


def test_model_params_hash_is_stable():
    h1 = params.model_params_hash()
    h2 = params.model_params_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_score_numeric_norm2():
    row = {
        "expected_behaviour": "answer",
        "expected_answer": "1,234.56",
        "expected_value": 1234.56,
        "expected_unit": "money",
        "expected_unit_raw": "Rs. in crore",
    }
    context = {("doc", 5)}
    good = score_prediction(row, "answer: 1234.56 | page: 5", context)
    assert good["numeric_match"] is True
    assert good["citation_valid"] is True
    assert good["false_answer"] is False
    assert good["false_abstention"] is False

    bad = score_prediction(row, "answer: 1234.57 | page: 5", context)
    assert bad["numeric_match"] is False


def test_score_abstention_errors():
    row = {"expected_behaviour": "abstain", "expected_answer": "ABSTAIN"}
    false_answer = score_prediction(row, "answer: 123 | page: 5", set())
    assert false_answer["false_answer"] is True
    assert false_answer["abstain_correct"] is False

    false_abs = score_prediction(
        {"expected_behaviour": "answer", "expected_answer": "1", "expected_value": 1},
        "ABSTAIN", set())
    assert false_abs["false_abstention"] is True


def test_pack_pages_drops_lowest_ranked():
    pages = [
        {"document_id": "d", "page": 1, "text": "a" * 100, "rank": 1},
        {"document_id": "d", "page": 2, "text": "b" * 100, "rank": 2},
    ]

    def fake_estimate(text):
        return len(text) // 4

    with patch("src.llm.client._estimate_tokens", fake_estimate):
        context, keys, ok = pack_pages(pages, "sys", "q", 10, 50)
    assert ok is True
    assert len(keys) == 1
    assert (pages[0]["document_id"], pages[0]["page"]) in keys


def test_non_numeric_entity_scoring():
    from src.config import GOLD_PATH
    from src.eval.prompting.scoring import expected_entity
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(GOLD_PATH)}
    for rid, page in (("G-003", 177), ("G-010", 205)):
        row = rows[rid]
        ent = expected_entity(row)
        ctx = {(ev["document_id"], ev["pdf_page"]) for ev in row.get("evidence") or []}
        score = score_prediction(row, f"answer: {ent} | page: {page}", ctx)
        assert score["gold_answer_kind"] == "entity"
        assert score["entity_match"] is True
        assert score["numeric_match"] is None
        assert score["parse_ok"] is True


def test_few_shot_has_no_gold_overlap():
    from src.eval.prompting.fewshot import load_few_shot, _gold_cited_pages
    rows = load_few_shot()
    cited = _gold_cited_pages()
    assert len(rows) == 3
    assert all((r["context_document_id"], r["context_page"]) not in cited for r in rows)


def test_entity_scoring_g003_g010():
    from src.config import GOLD_PATH
    from src.eval.prompting.scoring import expected_entity
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(GOLD_PATH)}
    cases = {"G-003": 177, "G-010": 205}
    for rid, page in cases.items():
        row = rows[rid]
        ent = expected_entity(row)
        ctx = {(ev["document_id"], ev["pdf_page"]) for ev in row.get("evidence") or []}
        score = score_prediction(row, f"answer: {ent} | page: {page}", ctx)
        assert score["gold_answer_kind"] == "entity"
        assert score["entity_match"] is True
        assert score["parse_ok"] is True


def test_few_shot_no_gold_overlap():
    from src.eval.prompting.fewshot import load_few_shot, _gold_cited_pages
    rows = load_few_shot()
    cited = _gold_cited_pages()
    assert len(rows) == 3
    assert all((r["context_document_id"], r["context_page"]) not in cited for r in rows)

def test_score_numeric_unit_normalization_crore_lakh():
    row = {
        "expected_behaviour": "answer",
        "expected_answer": "826189",
        "expected_value": 826189.0,
        "expected_unit": "money",
        "expected_unit_raw": "Rs. in lakh",
    }
    context = {("doc", 1)}
    good = score_prediction(row, "answer: 8261.89 crore | page: 1", context)
    assert good["numeric_match"] is True
    rev = score_prediction(
        {"expected_behaviour": "answer", "expected_answer": "8261.89", "expected_value": 8261.89,
         "expected_unit": "money", "expected_unit_raw": "Rs. in crore"},
        "answer: 826189 lakh | page: 1", context)
    assert rev["numeric_match"] is True

def test_score_applies_corrections_overlay(monkeypatch, tmp_path):
    # corrections overlay (HIGH) must be scored against corrected value
    from src.eval.prompting import inputs
    from src.eval.prompting.scoring import score_prediction
    corr = tmp_path / "gold_data.corrections.jsonl"
    corr.write_text(json.dumps({
        "row_id":"G-X","issue":"note_number_column_bug_relaxed","confidence":"HIGH",
        "proposed":{"expected_value":293.75,"expected_answer":"293.75"}
    }) + "\n")
    monkeypatch.setattr(inputs, "CORRECTIONS", corr)
    row = {"id":"G-X","split":"dev","status":"active","expected_behaviour":"answer",
           "expected_answer":"6","expected_value":6.0,"expected_unit":"money","expected_unit_raw":"Rs. in crore"}
    corrected = inputs.apply_corrections_overlay([row])[0]
    assert corrected["expected_value"] == 293.75
    s = score_prediction(corrected, "answer: 293.75 crore | page: 1", {("d",1)})
    assert s["numeric_match"] is True
