"""Audit: recompute metrics from raw predictions and diff against results/.

Exit non-zero if any metric, list hash, or run-record invariant is inconsistent.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from arag import util
from arag.config import get_config

METRIC_NAMES = ["hit@5", "ndcg@5", "hit@10", "ndcg@10", "mrr"]


def recompute_from_rank(rank: np.ndarray) -> dict[str, float]:
    r = rank.astype(float)
    return {
        "hit@5": float((r <= 5).mean()),
        "ndcg@5": float(np.where(r <= 5, 1.0 / np.log2(r + 1), 0.0).mean()),
        "hit@10": float((r <= 10).mean()),
        "ndcg@10": float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0).mean()),
        "mrr": float((1.0 / r).mean()),
    }


def main() -> int:
    cfg = get_config()
    results_dir = cfg.path("results")
    pred_dir = cfg.path("predictions")
    metrics = util.read_json(results_dir / "metrics.json")
    errors: list[str] = []
    sampled = set(util.read_json(cfg.path("lists") / "users.json")["test"])

    # "metrics": sampled test users; "metrics_all_users": every test user
    for key, users in (("metrics", sampled), ("metrics_all_users", None)):
        for sys_name, summary in metrics.get(key, {}).items():
            if "_vs_" in sys_name:
                continue
            path = pred_dir / f"{sys_name}_test.parquet"
            if not path.exists():
                errors.append(f"missing predictions: {sys_name}")
                continue
            df = pd.read_parquet(path)
            if users is not None:
                df = df[df["user_id"].isin(users)]
            rec = recompute_from_rank(df["rank"].to_numpy())
            for mm in METRIC_NAMES:
                expected = summary[mm]["mean"]
                if abs(rec[mm] - expected) > 1e-6:
                    errors.append(
                        f"{key}.{sys_name} {mm}: results={expected:.6f} recomputed={rec[mm]:.6f}"
                    )

    lists_dir = cfg.path("lists")
    hashes = util.read_json(lists_dir / "lists_hashes.json")
    for fname, sha in hashes.items():
        actual = util.sha256_file(lists_dir / fname)
        if actual != sha:
            errors.append(f"list hash mismatch: {fname}")

    rr_path = results_dir / "run_record.json"
    if not rr_path.exists():
        errors.append("missing run_record.json")
    else:
        rr = util.read_json(rr_path)
        if not rr.get("git_commit"):
            errors.append("run_record missing git_commit")

    if errors:
        print("VERIFY FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("VERIFY OK: metrics, list hashes, and run record are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
