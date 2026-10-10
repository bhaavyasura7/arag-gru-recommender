"""Central configuration loader.

Loads ``configs/config.yaml`` (relative to the repo root), exposes typed-ish
attribute access, resolves paths relative to the repo root, and reads any
environment secrets (never logging their values).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Config:
    def __init__(self, raw: dict[str, Any], repo_root: Path = REPO_ROOT):
        self.raw = raw
        self.repo_root = repo_root

    # -- dict access -------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # -- path helpers ------------------------------------------------------
    def path(self, dotted: str) -> Path:
        """Resolve a config path value to an absolute path under the repo root.

        ``dotted`` may be a bare key under ``paths.`` (e.g. ``"artifacts"``) or a
        full dotted key (e.g. ``"paths.lists"``).
        """
        if "." not in dotted:
            dotted = f"paths.{dotted}"
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"config path key not found: {dotted}")
        p = Path(value)
        if not p.is_absolute():
            p = self.repo_root / p
        return p

    # -- common shortcuts --------------------------------------------------
    @property
    def category(self) -> str:
        return self.raw["dataset"]["category"]

    @property
    def seed(self) -> int:
        return int(self.raw["sampling"]["seed"])

    @property
    def llm_provider(self) -> str:
        return self.raw["llm"]["provider"]

    @property
    def llm_model(self) -> str:
        return self.raw["llm"]["model"]

    def llm_base_url(self) -> str:
        provider = self.llm_provider
        return self.raw["llm"][provider]["base_url"]

    def llm_api_key(self) -> str:
        provider = self.llm_provider
        if provider == "ollama":
            return self.raw["llm"]["ollama"]["api_key"]
        env = self.raw["llm"]["openai"]["api_key_env"]
        return os.environ.get(env, "")

    def llm_has_key(self) -> bool:
        """Whether a usable API key is present (Ollama always 'has' one)."""
        if self.llm_provider == "ollama":
            return True
        return bool(self.llm_api_key())


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = REPO_ROOT / "configs" / "config.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw, repo_root=path.resolve().parent.parent)


def get_config() -> Config:
    """Cached singleton for the default config."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


_CONFIG: Config | None = None
