"""Configuration module — reads environment variables only.

Implements the cloud-agnostic rule: configuration through environment variables,
no cloud SDK in core code. All settings are loaded from .env via python-dotenv
or directly from os.environ. No secrets are stored in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover — dotenv is a runtime dep
    pass


@dataclass(frozen=True)
class LLMConfig:
    """LLM stack configuration (<=8B parameters, OpenAI-compatible)."""
    chat_url: str
    embeddings_url: str
    model: str
    embeddings_model: str
    max_tokens_chat: int
    max_tokens_embedding: int


@dataclass(frozen=True)
class SessionConfig:
    """Session and cache configuration."""
    ttl_seconds: int
    cache_ttl_seconds: int


@dataclass(frozen=True)
class CorpusConfig:
    """Corpus configuration."""
    freeze_version: str
    prompt_set_hash: str
    model_params_hash: str


@dataclass(frozen=True)
class ReflectionConfig:
    """Reflection loop settings (Track B skeleton)."""
    max_iterations: int
    max_total_tokens: int


def load_reflection_config() -> ReflectionConfig:
    """Load reflection loop settings from environment variables with defaults."""
    return ReflectionConfig(
        max_iterations=int(os.environ.get("REFLECTION_MAX_ITERATIONS", "2")),
        max_total_tokens=int(os.environ.get("REFLECTION_MAX_TOTAL_TOKENS", "4000")),
    )


@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""
    log_level: str
    api_host: str
    api_port: int
    streamlit_port: int
    streamlit_address: str
    llm: LLMConfig
    session: SessionConfig
    corpus: CorpusConfig


def load_config() -> AppConfig:
    """Load configuration from environment variables.

    Returns an AppConfig with all values resolved from env vars.
    Raises ValueError if a required variable is missing.
    """
    required_str = [
        "LLM_CHAT_URL", "LLM_EMBEDDINGS_URL", "LLM_MODEL",
        "EMBEDDINGS_MODEL", "INDEX_FREEZE_VERSION",
    ]
    missing = [v for v in required_str if not os.environ.get(v)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return AppConfig(
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        api_host=os.environ.get("API_HOST", "0.0.0.0"),
        api_port=int(os.environ.get("API_PORT", "8000")),
        streamlit_port=int(os.environ.get("STREAMLIT_SERVER_PORT", "8501")),
        streamlit_address=os.environ.get(
            "STREAMLIT_SERVER_ADDRESS", "0.0.0.0"
        ),
        llm=LLMConfig(
            chat_url=os.environ["LLM_CHAT_URL"],
            embeddings_url=os.environ["LLM_EMBEDDINGS_URL"],
            model=os.environ["LLM_MODEL"],
            embeddings_model=os.environ["EMBEDDINGS_MODEL"],
            max_tokens_chat=int(os.environ.get("MAX_TOKENS_CHAT", "4096")),
            max_tokens_embedding=int(
                os.environ.get("MAX_TOKENS_EMBEDDING", "512")
            ),
        ),
        session=SessionConfig(
            ttl_seconds=int(os.environ.get("SESSION_TTL_SECONDS", "3600")),
            cache_ttl_seconds=int(
                os.environ.get("CACHE_TTL_SECONDS", "3600")
            ),
        ),
        corpus=CorpusConfig(
            freeze_version=os.environ["INDEX_FREEZE_VERSION"],
            prompt_set_hash=os.environ.get("PROMPT_SET_HASH", "pending"),
            model_params_hash=os.environ.get("MODEL_PARAMS_HASH", "pending"),
        ),
    )
