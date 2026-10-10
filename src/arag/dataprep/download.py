"""Download the Amazon Reviews 2023 benchmark + metadata files for the category.

Saves raw files under ``data/raw/`` and records them (with SHA-256) in the
manifest. Skips files already present so re-running is cheap.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from arag import util
from arag.config import get_config
from arag.dataprep import manifest as manifest_mod

CHUNK = 1 << 20


def _download_url(url: str, dest: Path) -> dict:
    """Stream ``url`` to ``dest`` atomically; return {sha256, size_bytes}."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    hasher = hashlib.sha256()
    size = 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(CHUNK):
                if not block:
                    continue
                fh.write(block)
                hasher.update(block)
                size += len(block)
    os.replace(tmp, dest)
    return {"sha256": hasher.hexdigest(), "size_bytes": size}


def remote_files(cfg) -> dict[str, str]:
    """Map local filename -> remote path (relative to the dataset repo root)."""
    category = cfg.category
    return {
        f"{category}.train.csv": cfg.get("dataset.benchmark_files.train").format(category=category),
        f"{category}.valid.csv": cfg.get("dataset.benchmark_files.valid").format(category=category),
        f"{category}.test.csv": cfg.get("dataset.benchmark_files.test").format(category=category),
        f"meta_{category}.jsonl": cfg.get("dataset.meta_file").format(category=category),
    }


def main() -> None:
    logger = util.setup_logging("arag.dataprep.download")
    cfg = get_config()
    repo = cfg.get("dataset.id")
    raw_dir: Path = cfg.path("data_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    base = f"https://huggingface.co/datasets/{repo}/resolve/main/"

    manifest = manifest_mod.load(cfg)

    for local, remote in remote_files(cfg).items():
        dest = raw_dir / local
        url = base + remote
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("skip (exists) %s", local)
            continue
        logger.info("downloading %s ...", local)
        info = _download_url(url, dest)
        logger.info(
            "done %s (%.1f MB, sha256=%s)",
            local,
            info["size_bytes"] / 1e6,
            info["sha256"][:12],
        )
        manifest["files"][local] = {"remote": remote, **info}

    manifest["download_date"] = datetime.now(timezone.utc).isoformat()
    manifest_mod.save(cfg, manifest)
    logger.info("manifest -> %s", manifest_mod.manifest_path(cfg))


if __name__ == "__main__":
    main()
