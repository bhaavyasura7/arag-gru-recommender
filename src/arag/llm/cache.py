"""Disk cache for LLM calls, keyed by SHA-256 of (model, prompt, schema).

A cache hit makes no API call. Re-running the evaluation with the same prompts must
therefore make zero new calls.
"""
from __future__ import annotations

import json
from pathlib import Path

from arag import util


class DiskCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, prompt: str, schema: dict) -> str:
        payload = json.dumps(
            {"model": model, "prompt": prompt, "schema": schema},
            sort_keys=True,
            ensure_ascii=False,
        )
        return util.sha256_text(payload)

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return util.read_json(p)

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        util.write_json(p, value)

    def contains(self, key: str) -> bool:
        return self._path(key).exists()
