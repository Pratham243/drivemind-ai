"""Environment-based application settings and static YAML config loading.

Two layers of configuration are used deliberately:
  * ``Settings`` (this module) — secrets and per-environment values, read
    from environment variables / ``.env``. Never committed.
  * ``configs/config.yaml`` and ``configs/intents.yaml`` — static,
    version-controlled project configuration (paths, hyperparameters,
    intent taxonomy). Loaded via :func:`load_yaml_config`.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings sourced from environment variables / .env file.

    All fields have safe defaults so the application runs without a
    populated .env file (see PHASE 32 requirement: no external API
    dependency for core functionality).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    nlu_model: str = "baseline"  # baseline | transformer | llm

    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    @property
    def llm_available(self) -> bool:
        """Whether a real LLM provider can be constructed."""
        return bool(self.openai_api_key)


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache
def load_yaml_config(name: str = "config.yaml") -> dict[str, Any]:
    """Load a static YAML config file from configs/ by filename."""
    path = PROJECT_ROOT / "configs" / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache
def load_intents() -> dict[str, Any]:
    return load_yaml_config("intents.yaml")["intents"]
