from unittest.mock import patch

from src.eval.prompting.runner import run_arm
from src.llm.client import LLMResponse


def _fake_chat(messages, **kwargs):
    return LLMResponse("answer: 1 | page: 5", {"prompt_tokens": 10, "completion_tokens": 2},
                       200, False, "abc")


def test_runner_checkpoint_and_no_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr("src.eval.prompting.runner.chat", _fake_chat)
    monkeypatch.setattr("src.eval.prompting.runner.OUT_DIR", tmp_path)
    rows = [{"id": "G-001", "row_id": "G-001", "split": "dev", "status": "active",
             "expected_behaviour": "answer", "expected_answer": "1",
             "expected_value": 1, "question_text": "q", "flow_bucket": "b"}]
    manifest = run_arm("B0", rows=rows, smoke=True, live=True, limit=1,
                       run_id="testrun")
    assert manifest["n_processed_this_run"] == 1
    out = tmp_path / "testrun_B0.jsonl"
    assert out.is_file()
    # resume: second run skips the completed row
    manifest2 = run_arm("B0", rows=rows, smoke=True, live=True, limit=1,
                        run_id="testrun")
    assert manifest2["n_processed_this_run"] == 0
    assert manifest2["n_completed_before"] == 1


def test_context_cache_is_cached():
    from src.eval.prompting import context
    cache = context.load_context_cache()
    assert context._CONTEXT_CACHE is cache
    rid = next(iter(cache))
    pages = context.candidate_pages({"id": rid})
    assert pages == cache[rid]
