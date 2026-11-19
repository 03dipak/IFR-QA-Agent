from src.eval.baselines import run_b0, run_b1, run_b2, MODEL_PARAMS
from src.eval.prompts import PROMPT_SET_VERSION, prompt_set_hash, render_prompt


def test_pinned_params():
    assert MODEL_PARAMS["temperature"] == 0.0
    assert MODEL_PARAMS["max_tokens"] == 512
    assert MODEL_PARAMS["disable_thinking"] is True
    assert MODEL_PARAMS["cache"] is False


def test_prompt_set_hash_stable():
    assert prompt_set_hash() == prompt_set_hash()
    assert PROMPT_SET_VERSION.startswith("prompt_baselines")


def test_smoke_outputs_label_not_scored():
    row = {"row_id": "X1", "question_text": "What was total assets?"}
    for out in (run_b0(row), run_b1(row, context="ctx"),
                run_b2(row, context="ctx", retrieval_variant="global")):
        assert out["label"] == "SMOKE_NOT_SCORED"
        assert "answer:" in out["rendered_prompt"]
