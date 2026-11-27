"""Tests for runner secret redaction and exception logging."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

from src.eval.prompting.runner import redact_secrets, run_arm
from src.llm.client import LLMError


def test_redact_secrets_redacts_environment_secret(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "secret-value")
    assert "secret-value" not in redact_secrets("failed: secret-value")


def test_runner_logs_redacted_exception(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setenv("LLM_API_KEY", "secret-value")
    monkeypatch.setattr("src.eval.prompting.runner.OUT_DIR", tmp_path)

    def fake_chat(messages, **kwargs):
        raise LLMError("boom secret-value")

    with patch("src.eval.prompting.runner.chat", fake_chat):
        caplog.set_level(logging.WARNING)
        manifest = run_arm(
            "B0",
            rows=[{
                "id": "G-001", "row_id": "G-001", "split": "dev", "status": "active",
                "expected_behaviour": "answer", "expected_answer": "1",
                "question_text": "q", "flow_bucket": "b",
            }],
            smoke=False, live=True, limit=1, run_id="logtest",
        )

    assert manifest["n_processed_this_run"] == 1
    record = json.loads((tmp_path / "logtest_B0.jsonl").read_text().strip())
    assert record["status"] == "error"
    assert "secret-value" not in record["error"]
    assert "LLMError" in record["error"]
    assert "secret-value" not in caplog.text
