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
        """Key for the active provider: a literal ``api_key`` (Ollama) or the env var named by
        ``api_key_env``, falling back to the repo's git-ignored ``.env`` file."""
        section = self.raw["llm"].get(self.llm_provider, {})
        if "api_key" in section:
            return section["api_key"]
        env = section.get("api_key_env", "OPENAI_API_KEY")
        return os.environ.get(env, "") or _dotenv(self.repo_root).get(env, "")

    def llm_has_key(self) -> bool:
        """Whether a usable API key is present (Ollama always 'has' one)."""
        if self.llm_provider == "ollama":
            return True
        return bool(self.llm_api_key())


def _dotenv(root: Path) -> dict[str, str]:
    """KEY=VALUE pairs from ``<root>/.env`` (never logged)."""
    p = root / ".env"
    out: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Load ``path``, else ``$ARAG_CONFIG`` (e.g. a per-category config), else ``configs/config.yaml``."""
    if path is None:
        path = os.environ.get("ARAG_CONFIG") or REPO_ROOT / "configs" / "config.yaml"
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    # repo root = the directory holding the ``configs/`` folder this file lives in
    resolved = path.resolve()
    root = next((p.parent for p in resolved.parents if p.name == "configs"), resolved.parent.parent)
    return Config(raw, repo_root=root)


def get_config() -> Config:
    """Cached singleton for the default config."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


_CONFIG: Config | None = None
