"""Non-LLM evaluation (zero API calls): every trained / non-learned system on the frozen lists.

Systems: Random, Popularity, Content similarity, ItemKNN, ID-only GRU and text-aware GRU (all
seeds found on disk). Recency decay for content / ItemKNN is tuned on all validation users.
Metrics + paired bootstrap are computed on all test users and on the same sampled test users
the LLM systems use. Writes ``<results>/metrics_nonllm.json`` and ``<results>/tables_nonllm.md``;
does not touch ``metrics.json`` (the LLM evaluation's output).

    ARAG_CONFIG=configs/categories/Electronics.yaml python -m arag.eval.run_nonllm
"""
from __future__ import annotations

import time

import numpy as np

from arag import util
from arag.config import get_config
from arag.eval.metrics import ndcg_at_k, rank_of
from arag.eval.run import EvalData, compute_all_metrics, save_predictions
from arag.models.score import GRUScorer
from arag.ranking.content import ItemKNN, content_ranking
from arag.ranking.simple import popularity_ranking, random_ranking, seed_for

NAMES = {
    "random": "Random", "popularity": "Popularity", "content": "Content similarity", "itemknn": "ItemKNN",
    "gru_seed0": "GRU, ID only (seed 0)", "gru_seed1": "GRU, ID only (seed 1)", "gru_seed2": "GRU, ID only (seed 2)",
    "gru_text_seed0": "Text-aware GRU (seed 0)", "gru_text_seed1": "Text-aware GRU (seed 1)",
    "gru_text_seed2": "Text-aware GRU (seed 2)",
}
METRIC_ORDER = ["hit@5", "ndcg@5", "hit@10", "ndcg@10", "mrr"]


def _table(metrics: dict) -> list[str]:
    out = ["| System | " + " | ".join(METRIC_ORDER) + " |", "|" + "---|" * (len(METRIC_ORDER) + 1)]
    for sys, label in NAMES.items():
        if sys in metrics:
            out.append(f"| {label} | " + " | ".join(f"{metrics[sys][m]['mean']:.3f}" for m in METRIC_ORDER) + " |")
    return out


def _paired(metrics: dict) -> list[str]:
    out = ["| Contrast | ΔNDCG@5 | 95% CI | Significant |", "|---|---|---|---|"]
    for key, v in metrics.items():
        if "_vs_" not in key:
            continue
        d = v["ndcg@5"]
        lo, hi = d["ci"]
        sys, ref = key.split("_vs_")
        out.append(f"| {NAMES.get(sys, sys)} − {NAMES.get(ref, ref)} | {d['mean_diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    return out


def main() -> None:
    logger = util.setup_logging("arag.eval.run_nonllm")
    cfg = get_config()
    t0 = time.time()
    data = EvalData(cfg)
    data.load_embeddings()
    logger.info("[%s] loaded data (%.0fs)", cfg.category, time.time() - t0)

    # recency decay for content / ItemKNN on all validation users
    knn = ItemKNN([list(r["train_items"]) for r in data.seq_by_user.values()])
    val = data.valid_lists
    val_hist = {u: data.history_for(u, "valid")["item_id"].astype(str).tolist() for u in val["user_id"]}
    val_cands = dict(zip(val["user_id"], (list(c) for c in val["candidates"])))
    val_tgt = {u: str(data.seq_by_user[u]["valid_item"]) for u in val["user_id"]}

    def content_fn(h, c, d):
        return content_ranking(h, c, data.emb, data.emb_idx, d)

    def tune(fn, name):
        curve = {}
        for d in cfg.get("content.decay_grid", [0.0, 0.3, 0.6, 1.0]):
            curve[d] = float(np.mean([ndcg_at_k(rank_of(val_tgt[u], fn(val_hist[u], c, d)), 5) for u, c in val_cands.items()]))
            logger.info("  %s decay=%.1f val NDCG@5=%.4f", name, d, curve[d])
        return max(curve, key=curve.get), curve

    d_content, curve_content = tune(content_fn, "content")
    d_knn, curve_knn = tune(knn.rank, "itemknn")
    logger.info("tuned decay: content=%.1f itemknn=%.1f (%.0fs)", d_content, d_knn, time.time() - t0)

    # GRU scorers present on disk
    seeds = cfg.get("gru.seeds", [0, 1, 2])
    scorers = {}
    for kind, key in (("gru", "gru_seed"), ("gru_text", "gru_text_seed")):
        for s in seeds:
            p = cfg.path(kind) / str(s)
            if (p / "model.pt").exists():
                scorers[f"{key}{s}"] = GRUScorer(str(p))
    logger.info("GRU models found: %s", sorted(scorers))

    test = data.test_lists
    systems: dict[str, dict[str, list[str]]] = {k: {} for k in ["random", "popularity", "content", "itemknn", *scorers]}
    n = len(test)
    for i, (u, cands) in enumerate(zip(test["user_id"], test["candidates"])):
        cand = list(cands)
        hist = data.history_for(u, "test")["item_id"].astype(str).tolist()
        systems["random"][u] = random_ranking(cand, seed_for(cfg.seed, "random", u))
        systems["popularity"][u] = popularity_ranking(cand, data.popularity)
        systems["content"][u] = content_fn(hist, cand, d_content)
        systems["itemknn"][u] = knn.rank(hist, cand, d_knn)
        for k, sc in scorers.items():
            systems[k][u] = sc.rank(hist, cand)
        if (i + 1) % 10000 == 0 or i + 1 == n:
            logger.info("  ranked %d / %d test users (%.0fs)", i + 1, n, time.time() - t0)

    targets = {u: str(data.seq_by_user[u]["test_item"]) for u in test["user_id"]}
    sampled = set(data.users["test"])
    cfg.path("predictions").mkdir(parents=True, exist_ok=True)
    cfg.path("predictions_csv").mkdir(parents=True, exist_ok=True)
    for sys, rk in systems.items():
        save_predictions(cfg, sys, "test", rk, targets, sampled)

    logger.info("metrics + paired bootstrap ...")
    m_all = compute_all_metrics(systems, targets, cfg)
    m_sample = compute_all_metrics({s: {u: r for u, r in rk.items() if u in sampled} for s, rk in systems.items()}, targets, cfg)

    res = cfg.path("results")
    res.mkdir(parents=True, exist_ok=True)
    util.write_json(res / "metrics_nonllm.json", {
        "category": cfg.category,
        "n_test_users_all": n,
        "n_test_users_sampled": len(sampled),
        "content_decay": {"best": d_content, "curve": {str(k): v for k, v in curve_content.items()}},
        "itemknn_decay": {"best": d_knn, "curve": {str(k): v for k, v in curve_knn.items()}},
        "metrics_all_users": m_all,
        "metrics_sampled": m_sample,
        "elapsed_s": time.time() - t0,
    })
    lines = [f"# Non-LLM results — {cfg.category}", "", f"## All {n:,} test users", "", *_table(m_all), "",
             "### Paired bootstrap (ΔNDCG@5)", "", *_paired(m_all), "",
             f"## The {len(sampled)} sampled test users (same users as the LLM systems)", "", *_table(m_sample), ""]
    (res / "tables_nonllm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s and tables_nonllm.md (%.0fs total)", res / "metrics_nonllm.json", time.time() - t0)


if __name__ == "__main__":
    main()
