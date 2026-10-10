"""Build item text from metadata and parse fields (price, categories)."""
from __future__ import annotations

from typing import Any


def flatten_strings(value: Any) -> list[str]:
    """Flatten a possibly-nested list of strings/numbers into a flat list of str."""
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(flatten_strings(item))
    else:
        s = str(value).strip()
        if s:
            out.append(s)
    return out


def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("$", "").replace(",", "")
    if not s or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def build_item_text(meta: dict, max_words: int = 120) -> str:
    parts: list[str] = []
    title = meta.get("title")
    if title:
        parts.append(str(title).strip())
    main_cat = meta.get("main_category")
    if main_cat:
        parts.append(str(main_cat).strip())
    for key in ("categories", "features", "description"):
        parts.extend(flatten_strings(meta.get(key)))
    text = " | ".join(p for p in parts if p)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text
