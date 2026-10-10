"""Prompt loading and rendering helpers."""
from __future__ import annotations

from pathlib import Path

from arag.config import Config


def load_prompt(cfg: Config, name: str) -> str:
    p: Path = cfg.path("prompts") / f"{name}.txt"
    return p.read_text(encoding="utf-8")


def render(template: str, **kwargs) -> str:
    out = template
    for key, value in kwargs.items():
        out = out.replace(f"[[{key}]]", str(value))
    return out


def format_candidate(item: dict) -> str:
    """Render one candidate as 'item_id | title | text'."""
    return f"{item['item_id']} | {item.get('title', '')} | {item.get('text', '')}"


def format_candidates(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. {format_candidate(item)}")
    return "\n".join(lines)


def format_accepted(items: list[dict]) -> str:
    """Accepted items ordered by score: 'item_id | score | text'."""
    lines = []
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. {item['item_id']} | score={item.get('score', 0)} | {item.get('text', '')}")
    return "\n".join(lines)
