import json
import os
import urllib.error
import urllib.request

import pytest

import src.llm.client as lc


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._p).encode("utf-8")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.test.internal/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setattr(lc, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(lc, "_recent_calls", [])
    monkeypatch.setattr(lc, "_ENDPOINT_LOCK_PATH", tmp_path / "llm_endpoint.lock")
    monkeypatch.setattr(lc, "_endpoint_lock_fd", None)
    monkeypatch.setattr(lc, "_endpoint_lock_pid", None)
    yield
    monkeypatch.setattr(lc, "_endpoint_lock_fd", None)
    monkeypatch.setattr(lc, "_endpoint_lock_pid", None)


def _ok_payload(text="hello", ptok=10, ctok=5):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok}}


def test_url_appends_chat_completions_once(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.test.internal/v1")
    assert lc._chat_url() == "https://llm.test.internal/v1/chat/completions"


def test_max_tokens_clamped(monkeypatch, capsys):
    captured = {}

    def fake(req, timeout=None, context=None):
        captured["body"] = json.loads(req.data.decode())
        return FakeResponse(_ok_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    lc.chat([{"role": "user", "content": "hi"}], max_tokens=5000)
    assert captured["body"]["max_tokens"] == lc.MAX_TOKENS_CAP


def test_retry_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake(req, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate limit", {}, None)
        return FakeResponse(_ok_payload())

    sleeps = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    resp = lc.chat([{"role": "user", "content": "hi"}], cache=False)
    assert resp.status == 200
    assert calls["n"] == 3
    assert sleeps == [2, 4]


@pytest.mark.parametrize("code", [400, 401, 404])
def test_no_retry_on_client_errors(monkeypatch, code):
    calls = {"n": 0}

    def fake(req, timeout=None, context=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, code, "bad request", {}, None)

    sleeps = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(lc.LLMError):
        lc.chat([{"role": "user", "content": "hi"}], cache=False)
    assert calls["n"] == 1
    assert sleeps == []


def test_503_exhausts_retries(monkeypatch):
    def fake(req, timeout=None, context=None):
        raise urllib.error.HTTPError(req.full_url, 503, "down", {}, None)

    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(lc.LLMError):
        lc.chat([{"role": "user", "content": "hi"}], cache=False)


def test_think_tags_stripped(monkeypatch):
    payload = _ok_payload('<think>let me reason</think>{"answer": 42}')
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResponse(payload))
    resp = lc.chat([{"role": "user", "content": "hi"}], cache=False)
    assert resp.json() == {"answer": 42}
    assert "let me reason" not in resp.json().__repr__()


def test_cache_hit_skips_http(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: FakeResponse(_ok_payload()))
    r1 = lc.chat([{"role": "user", "content": "cached question"}])
    assert r1.cached is False
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("HTTP must not be called")))
    r2 = lc.chat([{"role": "user", "content": "cached question"}])
    assert r2.cached is True
    assert r2.text == "hello"


def test_rate_limit_sleeps_at_60_per_min(monkeypatch):
    sleeps = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(lc.time, "time", lambda: 1000.0)
    monkeypatch.setattr(lc, "_recent_calls", [999.0] * 60)
    lc._rate_window()
    assert sleeps and sleeps[0] > 0
