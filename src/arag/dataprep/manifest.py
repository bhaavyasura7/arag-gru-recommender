"""Manifest: records dataset provenance (files, SHA-256, row counts, dates).

Written to ``data/manifest.json`` with a tracked copy in ``results/manifest.json``.
"""
from __future__ import annotations

from pathlib import Path

from arag import util

MANIFEST_FILENAME = "manifest.json"


def manifest_path(cfg) -> Path:
    return cfg.path("data_processed").parent / MANIFEST_FILENAME


def new(cfg) -> dict:
    return {"dataset_id": cfg.get("dataset.id"), "category": cfg.category, "files": {}}


def load(cfg) -> dict:
    p = manifest_path(cfg)
    return util.read_json(p) if p.exists() else new(cfg)


def save(cfg, manifest: dict) -> None:
    util.write_json(manifest_path(cfg), manifest)


def write_tracked_copy(cfg, manifest: dict) -> None:
    util.write_json(cfg.path("results") / MANIFEST_FILENAME, manifest)
