"""The ONLY module that talks to the college Qwen3-8B endpoint.

Rules live in AGENTS.md ("## LLM endpoint (college Qwen3-8B)").

Config (env / .env, git-ignored):
  LLM_BASE_URL  (already ends with /v1; this module appends /chat/completions)
  LLM_API_KEY
  LLM_CA_CERT   (CA bundle path; TLS verification is ALWAYS on)
  LLM_MODEL     (default Qwen/Qwen3-8B)

Never print, log, commit or paste LLM_API_KEY anywhere. If a key appears in
any file or output: stop and report. No embeddings calls live here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover — dotenv is a runtime dep
    pass

log = logging.getLogger("llm.client")

CACHE_DIR = Path("data/llm_cache")
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS_CAP = 1024
DEFAULT_MAX_TOKENS = 512
MAX_TOTAL_TOKENS = 8192
RETRY_STATUSES = (429, 503)
BACKOFF_S = (2, 4, 8)
RATE_LIMIT_PER_MIN = 60

_lock = threading.Lock()
_recent_calls: list[float] = []
_think_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ENDPOINT_LOCK_PATH = Path("/tmp/opencode/llm_endpoint.lock")
_endpoint_lock_fd = None
_endpoint_lock_pid: int | None = None

def _acquire_endpoint_lock() -> None:
    """Process-level lock: refuse to run a second endpoint client."""
    global _endpoint_lock_fd, _endpoint_lock_pid
    import fcntl
    with _lock:
        if _endpoint_lock_fd is not None and _endpoint_lock_pid == os.getpid():
            return
        _ENDPOINT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd = open(_ENDPOINT_LOCK_PATH, "a+", encoding="utf-8")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fd.seek(0)
            holder = fd.read().strip() or "unknown"
            fd.close()
            raise LLMError(f"endpoint lock held by another process: {holder} ({_ENDPOINT_LOCK_PATH})")
        fd.seek(0)
        fd.truncate()
        fd.write(f"pid={os.getpid()}\nstarted={time.time()}\n")
        fd.flush()
        _endpoint_lock_fd = fd
        _endpoint_lock_pid = os.getpid()


class LLMError(RuntimeError):
    pass


class LLMResponse:
    def __init__(self, text: str, usage: dict, status: int,
                 cached: bool, prompt_version: str):
        self.text = text
        self.usage = usage or {}
        self.status = status
        self.cached = cached
        self.prompt_version = prompt_version

    def __repr__(self) -> str:
        tok = self.usage.get("prompt_tokens", 0), self.usage.get("completion_tokens", 0)
        return f"<LLMResponse status={self.status} cached={self.cached} tokens={tok}>"

    def json(self) -> Any:
        body = _think_re.sub("", self.text).strip()
        if not body:
            raise LLMError("empty response after stripping <think> tags")
        return json.loads(body)


def _base_url() -> str:
    base = os.environ.get("LLM_BASE_URL")
    if not base:
        raise LLMError("LLM_BASE_URL not set")
    return base.rstrip("/")


def _chat_url() -> str:
    base = _base_url()
    if base.endswith("/v1"):
        return base + CHAT_PATH
    return base + CHAT_PATH


def _models_url() -> str:
    base = _base_url()
    return base + (MODELS_PATH if base.endswith("/v1") else "/v1" + MODELS_PATH)


def _ssl_context() -> ssl.SSLContext:
    ca = os.environ.get("LLM_CA_CERT")
    ctx = ssl.create_default_context(cafile=ca or None)
    # Never verify=False.
    return ctx


def _estimate_tokens(text: str) -> int:
    """Conservative fixed chars-per-token estimate (len/3.0)."""
    return max(1, int(len(text) / 3.0))


def _rate_window() -> None:
    now = time.time()
    cutoff = now - 60.0
    while _recent_calls and _recent_calls[0] < cutoff:
        _recent_calls.pop(0)
    if len(_recent_calls) >= RATE_LIMIT_PER_MIN:
        wait = 60.0 - (now - _recent_calls[0]) + 0.05
        log.warning("rate limit: sleeping %.2fs (60/min cap)", wait)
        time.sleep(wait)


def _record_call() -> None:
    _recent_calls.append(time.time())


def _params_digest(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def _cache_key(model: str, messages: list[dict], params: dict) -> str:
    payload = {"model": model, "messages": messages, "params": params}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _cache_put(key: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8")


def strip_thinking(text: str) -> str:
    return _think_re.sub("", text).strip()


def chat(messages: list[dict], *, model: str | None = None,
         max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.0,
         top_p: float | None = None, cache: bool = True, timeout: float = 60.0,
         disable_thinking: bool = True) -> LLMResponse:
    """Single chat completion call. temperature 0, serialised, cached."""
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    max_tokens = min(int(max_tokens), MAX_TOKENS_CAP)
    params = {"max_tokens": max_tokens, "temperature": temperature}
    params_digest = _params_digest(params)
    prompt_version = hashlib.sha256(
        json.dumps({"model": model, "messages": messages, "params": params},
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    est = _estimate_tokens("".join(m.get("content", "") for m in messages))
    if est + max_tokens > MAX_TOTAL_TOKENS:
        raise LLMError(f"estimated prompt tokens {est} + max_tokens {max_tokens} "
                       f"exceeds {MAX_TOTAL_TOKENS} cap")

    if cache:
        key = _cache_key(model, messages, params)
        hit = _cache_get(key)
        if hit is not None:
            log.info("cache hit prompt_version=%s model=%s", prompt_version, model)
            return LLMResponse(hit.get("text", ""), hit.get("usage") or {},
                               int(hit.get("status", 200)), True, prompt_version)
    else:
        key = None

    _acquire_endpoint_lock()

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    with _lock:
        _rate_window()
        retries = 0
        t0 = time.time()
        status = 0
        usage: dict = {}
        text = ""
        while True:
            req = urllib.request.Request(
                _chat_url(),
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"})
            try:
                with urllib.request.urlopen(
                        req, timeout=timeout, context=_ssl_context()) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    status = resp.status
                    usage = payload.get("usage") or {}
                    choices = payload.get("choices") or []
                    text = (choices[0].get("message", {}) or {}).get("content", "") if choices else ""
                    break
            except urllib.error.HTTPError as e:
                status = e.code
                if status in RETRY_STATUSES and retries < len(BACKOFF_S):
                    s = BACKOFF_S[retries]
                    log.warning("HTTP %d: retry %d in %ds (prompt_version=%s)",
                                status, retries + 1, s, prompt_version)
                    time.sleep(s)
                    retries += 1
                    continue
                log.error("HTTP %d: failing (prompt_version=%s retries=%d)",
                          status, prompt_version, retries)
                raise LLMError(f"endpoint returned HTTP {status}") from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                raise LLMError(f"endpoint call failed: {type(e).__name__}") from e
        _record_call()

    latency_ms = int((time.time() - t0) * 1000)
    log.info("llm call status=%d latency_ms=%d prompt_tokens=%s completion_tokens=%s "
             "retries=%d prompt_version=%s model=%s cached=False",
             status, latency_ms, usage.get("prompt_tokens"),
             usage.get("completion_tokens"), retries, prompt_version, model)

    if cache and key is not None:
        _cache_put(key, {"text": text, "usage": usage, "status": status,
                         "prompt_version": prompt_version})
    return LLMResponse(text, usage, status, False, prompt_version)


def list_models(timeout: float = 30.0) -> list[str]:
    req = urllib.request.Request(
        _models_url(), method="GET",
        headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=timeout,
                                context=_ssl_context()) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else payload
    return [m.get("id", "?") if isinstance(m, dict) else str(m) for m in (data or [])]
