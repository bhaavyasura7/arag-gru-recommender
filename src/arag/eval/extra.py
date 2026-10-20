"""Extra analyses on the saved rankings (zero LLM calls).

1. History length: NDCG@5 per history-length bucket (does a method still work for users
   with only 4 past items?).
2. SASRec: the standard self-attentive baseline vs the GRUs and ARAG (if trained by
   ``arag.models.train_sasrec``).
3. Beyond accuracy: popularity bias, intra-list diversity and coverage of each top-5.

Reads ``artifacts/predictions/*_test.parquet``; writes ``results/extra_metrics.json``,
``results/extra_tables.md`` and ``docs/figures/fig12-14``.

    python -m arag.eval.extra
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arag import util
from arag.config import REPO_ROOT, get_config
from arag.eval.bootstrap import paired_bootstrap
from arag.eval.metrics import hit_at_k, mrr, ndcg_at_k

SEEDS = [0, 1, 2]
# systems averaged over their 3 seeds (per user) to remove seed noise
SEEDED = {"gru": "gru_seed{}", "gru_text": "gru_text_seed{}", "sasrec": "sasrec_seed{}"}
LLM = ["recency", "vanilla_rag", "arag", "arag_gru", "arag_gru_text"]
NON_LLM = ["random", "popularity", "content", "itemknn", "gru", "gru_text", "sasrec"]
NAMES = {
    "random": "Random", "popularity": "Popularity", "content": "Content (text match)",
    "itemknn": "ItemKNN", "gru": "GRU (ID)", "gru_text": "Text-aware GRU", "sasrec": "SASRec",
    "recency": "LLM-Recency", "vanilla_rag": "LLM-Vanilla-RAG", "arag": "ARAG",
    "arag_gru": "ARAG + GRU", "arag_gru_text": "ARAG + text-GRU",
}
# history length = interactions before the test target (train + validation item)
BUCKETS_ALL = [(4, 4, "4 items"), (5, 6, "5–6"), (7, 10, "7–10"), (11, 10**9, "11+")]
BUCKETS_100 = [(4, 4, "4 items"), (5, 7, "5–7"), (8, 10**9, "8+")]


def mean_ci(x: np.ndarray) -> dict:
    m, lo, hi = paired_bootstrap(x, np.zeros_like(x))
    return {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n": int(len(x))}


def diff_ci(a: np.ndarray, b: np.ndarray) -> dict:
    m, lo, hi = paired_bootstrap(a, b)
    return {"mean_diff": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "significant": bool(lo > 0 or hi < 0)}


class Rankings:
    """Per-user rank of the target and top-5 lists for every saved system."""

    def __init__(self, pred_dir: Path):
        self.rank: dict[str, pd.Series] = {}
        self.top5: dict[str, pd.Series] = {}
        for path in sorted(pred_dir.glob("*_test.parquet")):
            name = path.name.removesuffix("_test.parquet")
            df = pd.read_parquet(path).set_index("user_id")
            self.rank[name] = df["rank"].astype(int)
            self.top5[name] = df["ranked_ids"].map(lambda r: [str(i) for i in r[:5]])
        self.available = {s for s in NON_LLM + LLM if self._present(s)}

    def _present(self, system: str) -> bool:
        if system in SEEDED:
            return all(SEEDED[system].format(s) in self.rank for s in SEEDS)
        return system in self.rank

    def metric(self, system: str, users: list[str], fn) -> np.ndarray:
        """Per-user metric; seeded systems are averaged over seeds."""
        names = [SEEDED[system].format(s) for s in SEEDS] if system in SEEDED else [system]
        return np.mean([self.rank[n].loc[users].map(fn).to_numpy(dtype=float) for n in names], axis=0)

    def lists(self, system: str, users: list[str]) -> list[list[list[str]]]:
        """Top-5 lists per seed (one entry for unseeded systems)."""
        names = [SEEDED[system].format(s) for s in SEEDS] if system in SEEDED else [system]
        return [self.top5[n].loc[users].tolist() for n in names]


def ndcg5(r):
    return ndcg_at_k(r, 5)


def hit5(r):
    return hit_at_k(r, 5)


def ndcg10(r):
    return ndcg_at_k(r, 10)


def hit10(r):
    return hit_at_k(r, 10)


# ---------------------------------------------------------------------------
# 1. history length
# ---------------------------------------------------------------------------
def history_buckets(R: Rankings, users: list[str], hist_len: pd.Series, systems: list[str], buckets) -> dict:
    out = {}
    for lo, hi, label in buckets:
        us = [u for u in users if lo <= hist_len[u] <= hi]
        out[label] = {"n_users": len(us), **{s: mean_ci(R.metric(s, us, ndcg5)) for s in systems}}
    return out


def bucket_diffs(R: Rankings, users, hist_len, pairs, buckets) -> dict:
    out = {}
    for lo, hi, label in buckets:
        us = [u for u in users if lo <= hist_len[u] <= hi]
        out[label] = {f"{a}_vs_{b}": diff_ci(R.metric(a, us, ndcg5), R.metric(b, us, ndcg5)) for a, b in pairs}
    return out


# ---------------------------------------------------------------------------
# 3. beyond accuracy
# ---------------------------------------------------------------------------
class ItemSpace:
    def __init__(self, cfg):
        ids = np.load(cfg.path("faiss") / "item_ids.npy", allow_pickle=True)
        emb = np.load(cfg.path("faiss") / "item_embeddings.npy").astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        self.pos = {str(i): k for k, i in enumerate(ids)}
        self.emb = emb
        inter = pd.read_parquet(cfg.path("data_processed") / "interactions.parquet")
        counts = inter.loc[inter["split"] == "train", "item_id"].astype(str).value_counts()
        self.logpop = np.log1p(counts).to_dict()

    def ild(self, items: list[str]) -> float:
        """Intra-list diversity: 1 - mean pairwise cosine similarity of the item texts."""
        v = self.emb[[self.pos[i] for i in items if i in self.pos]]
        if len(v) < 2:
            return float("nan")
        sim = v @ v.T
        n = len(v)
        return float(1 - (sim.sum() - n) / (n * (n - 1)))

    def pop(self, items: list[str]) -> float:
        return float(np.mean([self.logpop.get(i, 0.0) for i in items]))


def beyond_accuracy(R: Rankings, S: ItemSpace, users: list[str], cand: pd.Series, systems: list[str]) -> dict:
    pool_pop = np.array([S.pop(cand[u]) for u in users])
    pool_items = {i for u in users for i in cand[u]}
    out = {}
    for s in systems:
        per_seed = R.lists(s, users)
        lift = np.mean([[S.pop(t) for t in lists] for lists in per_seed], axis=0) - pool_pop
        ild = np.nanmean([[S.ild(t) for t in lists] for lists in per_seed], axis=0)
        cov = np.mean([len({i for t in lists for i in t}) / len(pool_items) for lists in per_seed])
        out[s] = {"popularity_lift": mean_ci(lift), "diversity_ild": mean_ci(ild), "coverage": round(float(cov), 4)}
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
COLORS = {"gru": "#4285f4", "gru_text": "#1a73e8", "sasrec": "#a142f4", "content": "#34a853",
          "itemknn": "#81c995", "popularity": "#9aa0a6", "random": "#dadce0", "arag": "#ea4335",
          "arag_gru_text": "#c5221f", "recency": "#fbbc05", "vanilla_rag": "#f9ab00", "arag_gru": "#ee675c"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def grouped_bars(ax, table: dict, systems: list[str], title: str):
    labels = list(table)
    w = 0.8 / len(systems)
    x = np.arange(len(labels))
    for k, s in enumerate(systems):
        means = [table[b][s]["mean"] for b in labels]
        err = np.array([[table[b][s]["mean"] - table[b][s]["ci"][0], table[b][s]["ci"][1] - table[b][s]["mean"]] for b in labels]).T
        ax.bar(x + (k - (len(systems) - 1) / 2) * w, means, w, yerr=err, capsize=2, label=NAMES[s],
               color=COLORS.get(s, "#5f6368"), edgecolor="white", linewidth=0.5, error_kw={"lw": 0.8, "alpha": 0.6})
    ax.set_xticks(x, [f"{b}\n(n={table[b]['n_users']:,})" for b in labels])
    ax.set_ylabel("NDCG@5")
    ax.set_xlabel("Items in the user's history before the test item")
    ax.set_title(title)


def fig_history(res: dict, out: Path):
    sys_all = [s for s in ["popularity", "content", "gru", "gru_text", "sasrec"] if s in res["history_all"]["4 items"]]
    sys_100 = [s for s in ["gru", "gru_text", "content", "arag", "arag_gru_text"] if s in res["history_100"]["4 items"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [4, 3]})
    grouped_bars(axes[0], res["history_all"], sys_all, "All 94,762 test users (non-LLM systems)")
    grouped_bars(axes[1], res["history_100"], sys_100, "The 100 LLM test users")
    for ax in axes:
        ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    fig.suptitle("Does it still work for users with very short histories?", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fig12_history_length_performance.png", bbox_inches="tight")
    plt.close(fig)


def fig_sasrec(res: dict, out: Path):
    t = res["overall_100"]
    order = [s for s in ["random", "popularity", "gru", "sasrec", "itemknn", "content", "gru_text",
                         "recency", "vanilla_rag", "arag", "arag_gru", "arag_gru_text"] if s in t]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    means = [t[s]["ndcg@5"]["mean"] for s in order]
    err = np.array([[t[s]["ndcg@5"]["mean"] - t[s]["ndcg@5"]["ci"][0], t[s]["ndcg@5"]["ci"][1] - t[s]["ndcg@5"]["mean"]] for s in order]).T
    y = np.arange(len(order))
    ax.barh(y, means, xerr=err, color=[COLORS.get(s, "#5f6368") for s in order], capsize=3,
            error_kw={"lw": 0.9, "alpha": 0.7})
    for yi, m, hi in zip(y, means, err[1]):
        ax.text(m + hi + 0.008, yi, f"{m:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, [NAMES[s] for s in order])
    ax.set_xlabel("NDCG@5 on the 100 test users (95% CI)")
    ax.set_xlim(0, 0.72)
    ax.set_title("All systems on the same 100 test users (sorted by family)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig13_sasrec_baseline.png", bbox_inches="tight")
    plt.close(fig)


def fig_beyond(res: dict, out: Path):
    t = res["beyond_accuracy_100"]
    order = [s for s in ["popularity", "gru", "sasrec", "itemknn", "content", "gru_text",
                         "recency", "vanilla_rag", "arag", "arag_gru", "arag_gru_text"] if s in t]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, key, title, xlabel in [
        (axes[0], "popularity_lift", "Popularity bias of the top-5",
         "log-popularity of top-5 minus the 20-item list average\n(> 0 = prefers popular items)"),
        (axes[1], "diversity_ild", "Diversity of the top-5",
         "intra-list diversity (1 - mean text similarity)"),
    ]:
        means = [t[s][key]["mean"] for s in order]
        err = np.array([[t[s][key]["mean"] - t[s][key]["ci"][0], t[s][key]["ci"][1] - t[s][key]["mean"]] for s in order]).T
        y = np.arange(len(order))
        ax.barh(y, means, xerr=err, color=[COLORS.get(s, "#5f6368") for s in order], capsize=3,
                error_kw={"lw": 0.9, "alpha": 0.7})
        ax.set_yticks(y, [NAMES[s] for s in order])
        ax.axvline(0, color="#202124", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Beyond accuracy (the 100 test users)", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fig14_beyond_accuracy.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def fmt(d):
    return f"{d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def fmt_diff(d):
    return f"{d['mean_diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] {'✓' if d['significant'] else '–'}"


def tables_md(res: dict) -> str:
    L = ["# Extra analyses (zero LLM calls)", "",
         "Generated by `python -m arag.eval.extra` from the saved per-user rankings. "
         "GRU, text-aware GRU and SASRec are averaged over 3 seeds per user. "
         "Brackets = 95% paired-bootstrap CI (10,000 resamples); ✓ = CI excludes 0.", ""]

    def bucket_table(title, t, systems):
        L.extend([f"## {title}", "", "| History | Users | " + " | ".join(NAMES[s] for s in systems) + " |",
                  "|---|---:|" + "---:|" * len(systems)])
        for b, row in t.items():
            L.append(f"| {b} | {row['n_users']:,} | " + " | ".join(fmt(row[s]) for s in systems) + " |")
        L.append("")

    def diff_table(title, t):
        pairs = list(next(iter(t.values())))
        L.extend([f"### {title}", "", "| History | " + " | ".join(p.replace("_vs_", " vs ") for p in pairs) + " |",
                  "|---|" + "---|" * len(pairs)])
        for b, row in t.items():
            L.append(f"| {b} | " + " | ".join(fmt_diff(row[p]) for p in pairs) + " |")
        L.append("")

    bucket_table("1a. NDCG@5 by history length — all test users", res["history_all"], res["systems_all"])
    diff_table("Paired differences (all users)", res["history_all_diffs"])
    bucket_table("1b. NDCG@5 by history length — the 100 LLM test users", res["history_100"], res["systems_100"])
    diff_table("Paired differences (100 users)", res["history_100_diffs"])

    for scope, key in [("100 test users", "overall_100"), ("all test users (non-LLM)", "overall_all")]:
        L.extend([f"## 2. Overall — {scope}", "", "| System | NDCG@5 | Hit@5 | MRR |", "|---|---:|---:|---:|"])
        for s, m in res[key].items():
            L.append(f"| {NAMES[s]} | {fmt(m['ndcg@5'])} | {fmt(m['hit@5'])} | {fmt(m['mrr'])} |")
        L.append("")
    if res.get("sasrec_tests"):
        L.extend(["### Key paired tests (NDCG@5)", "", "| Comparison | Users | Δ [95% CI] |", "|---|---:|---|"])
        for k, d in res["sasrec_tests"].items():
            L.append(f"| {k} | {d['n']:,} | {fmt_diff(d)} |")
        L.append("")

    L.extend(["## 3. Beyond accuracy — 100 test users", "",
              "| System | Popularity lift | Diversity (ILD) | Coverage |", "|---|---:|---:|---:|"])
    for s, m in res["beyond_accuracy_100"].items():
        L.append(f"| {NAMES[s]} | {fmt(m['popularity_lift'])} | {fmt(m['diversity_ild'])} | {m['coverage']:.3f} |")
    L.extend(["", "Popularity lift = mean log(1+train count) of the top-5 minus the 20-candidate average "
              "(> 0 favours popular items). ILD = 1 − mean pairwise cosine similarity of the top-5 item "
              "texts (MiniLM). Coverage = distinct top-5 items / distinct candidate items.", ""])
    return "\n".join(L)


def main() -> None:
    logger = util.setup_logging("arag.eval.extra")
    cfg = get_config()
    R = Rankings(cfg.path("artifacts") / "predictions")
    seqs = pd.read_parquet(cfg.path("lists") / "sequences.parquet").set_index("user_id")
    hist_len = seqs["n_train"] + 1  # train items + the validation item
    users_100 = util.read_json(cfg.path("lists") / "users.json")["test"]
    users_all = R.rank["random"].index.tolist()
    cand = pd.read_parquet(cfg.path("lists") / "test_lists.parquet").set_index("user_id")["candidates"].map(
        lambda c: [str(i) for i in c])

    systems_all = [s for s in NON_LLM if s in R.available]
    systems_100 = systems_all + [s for s in LLM if s in R.available]
    logger.info("systems: %s", systems_100)

    res: dict = {"systems_all": systems_all, "systems_100": systems_100}
    res["history_all"] = history_buckets(R, users_all, hist_len, systems_all, BUCKETS_ALL)
    pairs_all = [("gru_text", "gru"), ("content", "gru")] + ([("sasrec", "gru"), ("gru_text", "sasrec")] if "sasrec" in R.available else [])
    res["history_all_diffs"] = bucket_diffs(R, users_all, hist_len, pairs_all, BUCKETS_ALL)
    res["history_100"] = history_buckets(R, users_100, hist_len, systems_100, BUCKETS_100)
    res["history_100_diffs"] = bucket_diffs(
        R, users_100, hist_len, [("arag", "gru"), ("arag", "gru_text"), ("arag_gru_text", "arag")], BUCKETS_100)
    logger.info("history-length analysis done")

    for key, users, systems in [("overall_100", users_100, systems_100), ("overall_all", users_all, systems_all)]:
        res[key] = {s: {name: mean_ci(R.metric(s, users, fn)) for name, fn in [("ndcg@5", ndcg5), ("hit@5", hit5), ("ndcg@10", ndcg10), ("hit@10", hit10), ("mrr", mrr)]}
                    for s in systems}
    if "sasrec" in R.available:
        tests = {}
        for a, b, users in [("sasrec", "gru", users_all), ("sasrec", "gru_text", users_all),
                            ("sasrec", "itemknn", users_all), ("arag", "sasrec", users_100),
                            ("arag", "gru", users_100), ("arag", "gru_text", users_100),
                            ("arag_gru_text", "arag", users_100), ("arag_gru_text", "sasrec", users_100)]:
            tests[f"{NAMES[a]} vs {NAMES[b]}"] = {**diff_ci(R.metric(a, users, ndcg5), R.metric(b, users, ndcg5)), "n": len(users)}
        res["sasrec_tests"] = tests
    logger.info("overall + SASRec comparisons done")

    S = ItemSpace(cfg)
    res["beyond_accuracy_100"] = beyond_accuracy(R, S, users_100, cand, [s for s in systems_100 if s != "random"])
    logger.info("beyond-accuracy done")

    util.write_json(cfg.path("results") / "extra_metrics.json", res)
    (cfg.path("results") / "extra_tables.md").write_text(tables_md(res), encoding="utf-8")
    fig_dir = REPO_ROOT / "docs" / "figures"
    _style()
    fig_history(res, fig_dir)
    fig_sasrec(res, fig_dir)
    fig_beyond(res, fig_dir)
    logger.info("wrote results/extra_metrics.json, results/extra_tables.md and docs/figures/fig12-14")


if __name__ == "__main__":
    main()
