"""Extract display-only product fields (image, star rating, store, platform) for the demo UI.

Reads the raw Amazon metadata once and writes artifacts/item_extras.parquet. These fields
are only shown in the frontend; no model or agent prompt uses them.

    python scripts/build_item_extras.py
"""
from __future__ import annotations

import json
import re

import pandas as pd

from arag.config import get_config


def _image(images: list[dict]) -> str | None:
    if not images:
        return None
    main = next((im for im in images if im.get("variant") == "MAIN"), images[0])
    return main.get("large") or main.get("hi_res") or main.get("thumb")


# store navigation also has nodes like "Game Genre of the Month"; keep only real platforms
_PLATFORM = re.compile(r"PC|Mac|Linux|Nintendo|PlayStation|PS Vita|Xbox|Sega|Wii|Game Boy|Atari|Virtual Reality|Online Game")


def _platform(categories: list[str]) -> str | None:
    # categories look like ["Video Games", "PC", "Games"]; the second entry is the platform,
    # except under "Legacy Systems", where the console is the third entry
    if len(categories) > 2 and categories[1] == "Legacy Systems":
        name = categories[2]
    elif len(categories) > 1:
        name = categories[1]
    else:
        return None
    return name if _PLATFORM.search(name) else None


def _clip(text: str, limit: int = 800) -> str | None:
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def main() -> None:
    cfg = get_config()
    src = cfg.path("data_raw") / "meta_Video_Games.jsonl"
    rows = []
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "item_id": r["parent_asin"],
                    "image": _image(r.get("images") or []),
                    "avg_rating": r.get("average_rating"),
                    "n_ratings": r.get("rating_number"),
                    "store": r.get("store") or None,
                    "platform": _platform(r.get("categories") or []),
                    "features": [f for f in (r.get("features") or []) if f][:6],
                    "about": _clip(" ".join(d for d in (r.get("description") or []) if d)),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates("item_id")
    out = cfg.path("artifacts") / "item_extras.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} items to {out} ({df['image'].notna().mean():.0%} with an image)")


if __name__ == "__main__":
    main()
