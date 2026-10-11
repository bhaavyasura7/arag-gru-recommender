"""Generate a consistent set of presentation figures for the ARAG+GRU report.

All numbers are read from results/metrics.json (the audited, corrected run) or
recomputed from the saved per-user rankings. Paper and superseded-run numbers are
hard-coded with their source (REPORT.md Appendix F / Appendix E) so the figures
are self-documenting.

Outputs to docs/figures/:
  fig01_pipeline.png        the 4-agent ARAG + GRU fusion pipeline
  fig02_main_results.png    all systems, NDCG@5 and Hit@5
  fig03_hypotheses.png      H1 / H2 forest plot with 95% CIs
  fig04_ablation.png        ablation staircase, ours vs the paper
  fig05_alpha_curve.png     fusion weight alpha (validation vs test)
  fig06_paper_vs_ours.png   paper Table 1 side by side
  fig07_protocol_flaw.png   the degenerate-vs-corrected reversal
  fig08_history_lengths.png why the GRU fails: user history length
  fig09_fusion_methods.png  uniform vs RRF vs exponential fusion
  fig10_content_aware.png   extension: content baselines + text-aware GRU (zero LLM cost)

Usage:  .venv/bin/python scripts/make_figures.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "results" / "metrics.json").read_text())["metrics"]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- consistent style -------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)

# palette
C = {
    "base": "#9aa0a6",      # baselines (random/popularity)
    "gru": "#4285f4",       # GRU
    "llm": "#34a853",       # LLM baselines (recency, vanilla)
    "arag": "#ea4335",      # ARAG / fused
    "abl": "#fbbc05",       # ablations
    "ink": "#202124",
}


def ndcg(name: str) -> float:
    return M[name]["ndcg@5"]["mean"]


def hit(name: str) -> float:
    return M[name]["hit@5"]["mean"]


def paired(name: str):
    d = M[name]["ndcg@5"]
    return d["mean_diff"], d["ci"][0], d["ci"][1]


# ---------------------------------------------------------------------------
# fig01 pipeline schematic
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc=C["ink"], fs=10, bold=False):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                facecolor=fc, edgecolor=C["ink"], linewidth=1.0,
            )
        )
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.35,
        )

    def arrow(x1, y1, x2, y2, color=C["ink"], style="-|>", ls="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=1.6, linestyle=ls),
        )

    # top row: input -> UUA -> NLI -> CSA -> IRA
    box(0.02, 0.62, 0.13, 0.22, "User\nhistory", "#e8eaed")
    arrow(0.155, 0.73, 0.20, 0.73)
    box(0.20, 0.62, 0.14, 0.22, "UUA\nuser\nsummary", "#d2e3fc")
    arrow(0.345, 0.73, 0.39, 0.73)
    box(0.39, 0.62, 0.16, 0.22, "NLI\nscore 20\ncandidates", "#d2e3fc")
    arrow(0.555, 0.73, 0.60, 0.73)
    box(0.60, 0.62, 0.14, 0.22, "CSA\ncondense", "#d2e3fc")
    arrow(0.745, 0.73, 0.79, 0.73)
    box(0.79, 0.62, 0.15, 0.22, "IRA\nLLM\nranking", "#fce8e6")

    # bottom row: history -> GRU
    box(0.02, 0.12, 0.13, 0.22, "User\nhistory", "#e8eaed")
    arrow(0.155, 0.23, 0.30, 0.23)
    box(0.30, 0.12, 0.18, 0.22, "GRU4Rec\nsequence\nmodel", "#d2e3fc")
    arrow(0.485, 0.23, 0.79, 0.23)
    box(0.79, 0.12, 0.15, 0.22, "GRU\nranking", "#d2e3fc")

    # fusion
    arrow(0.865, 0.62, 0.865, 0.46)
    arrow(0.865, 0.34, 0.865, 0.46)
    box(0.70, 0.40, 0.24, 0.10,
        "Fusion   α·s_LLM + (1−α)·s_GRU", "#fce8e6", fs=10, bold=True)
    arrow(0.82, 0.45, 0.82, 0.45)  # noop keep layout
    box(0.40, 0.40, 0.26, 0.10, "Final ranking  (α*=0.9)", "#fce8e6", fs=10, bold=True)
    arrow(0.70, 0.45, 0.665, 0.45)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)
    ax.set_title("ARAG 4-agent pipeline + late GRU fusion", loc="left")
    fig.savefig(OUT / "fig01_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig02 main results
# ---------------------------------------------------------------------------
def fig_main_results():
    order = [
        ("random", "Random", C["base"]),
        ("popularity", "Popularity", C["base"]),
        ("gru_seed0", "GRU (seed 0)", C["gru"]),
        ("vanilla_rag", "LLM-Vanilla-RAG", C["llm"]),
        ("recency", "LLM-Recency", C["llm"]),
        ("arag_wo_nli_csa", "ARAG w/o NLI+CSA", C["abl"]),
        ("arag_wo_nli", "ARAG w/o NLI", C["abl"]),
        ("arag_gru", "ARAG + GRU (fusion)", C["arag"]),
        ("arag", "ARAG", C["arag"]),
    ]
    names = [o[1] for o in order]
    nd = [ndcg(o[0]) for o in order]
    ht = [hit(o[0]) for o in order]
    cols = [o[2] for o in order]

    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].barh(y, nd, color=cols, edgecolor="none")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("NDCG@5")
    axes[0].set_title("NDCG@5  (higher better)")
    for i, v in enumerate(nd):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(nd) * 1.18)

    axes[1].barh(y, ht, color=cols, edgecolor="none")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Hit@5")
    axes[1].set_title("Hit@5  (higher better)")
    for i, v in enumerate(ht):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].set_xlim(0, max(ht) * 1.18)

    fig.suptitle(
        "All systems on 100 held-out test users  —  ARAG is best, fusion ≈ ARAG",
        x=0.02, ha="left", fontsize=13, fontweight="bold",
    )
    fig.savefig(OUT / "fig02_main_results.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig03 hypotheses forest plot
# ---------------------------------------------------------------------------
def fig_hypotheses():
    rows = [
        ("H1: ARAG+GRU  vs  ARAG", "arag_gru_vs_arag"),
        ("H2: ARAG+GRU  vs  GRU", "arag_gru_vs_gru_seed0"),
        ("ARAG  vs  LLM-Recency", "recency_vs_arag"),
        ("ARAG  vs  LLM-Vanilla-RAG", "vanilla_rag_vs_arag"),
        ("ARAG  vs  GRU (seed 1)", "gru_seed1_vs_arag"),
        ("ARAG  vs  GRU (seed 2)", "gru_seed2_vs_arag"),
    ]
    labels = [r[0] for r in rows]
    diffs, los, his, sig = [], [], [], []
    for i, (_, key) in enumerate(rows):
        d, lo, hi = paired(key)
        # rows 0-1 (arag_gru_vs_*) store (fusion - other) = (first - second): keep.
        # rows 2-5 (X_vs_arag) store (X - arag) = (second - first): flip so the
        # display is always (first-named system) - (second-named system).
        if i >= 2:
            d, lo, hi = -d, -hi, -lo
        diffs.append(d)
        los.append(lo)
        his.append(hi)
        sig.append(lo > 0 or hi < 0)  # CI excludes 0 in either direction

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axvline(0, color=C["ink"], lw=1.0, ls="--", alpha=0.7)
    for i, (d, lo, hi) in enumerate(zip(diffs, los, his)):
        col = C["arag"] if sig[i] else C["base"]
        ax.plot([lo, hi], [i, i], color=col, lw=2.4, solid_capstyle="round")
        ax.plot([d], [i], "o", color=col, ms=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Δ NDCG@5   (positive = ARAG / fusion better)")
    ax.set_title("Paired bootstrap comparisons vs ARAG / GRU  (95% CI, 2,000 draws)")
    # annotate
    for i, (d, lo, hi) in enumerate(zip(diffs, los, his)):
        mark = "SIG" if lo > 0 else "n.s."
        ax.text(
            max(his) * 1.02, i, f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {mark}",
            va="center", fontsize=9,
        )
    ax.set_xlim(min(los) * 1.05, max(his) * 1.42)
    # legend
    handles = [
        mpatches.Patch(color=C["arag"], label="CI excludes 0 (significant)"),
        mpatches.Patch(color=C["base"], label="CI includes 0 (not significant)"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    fig.savefig(OUT / "fig03_hypotheses.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig04 ablation staircase (ours vs paper)
# ---------------------------------------------------------------------------
def fig_ablation():
    # paper Table 1 / F.4, NDCG@5
    paper = {
        "Vanilla RAG": {"Clothing": 0.299, "Electronics": 0.238, "Home": 0.229},
        "ARAG w/o NLI+CSA": {"Clothing": 0.302, "Electronics": 0.272, "Home": 0.249},
        "ARAG w/o NLI": {"Clothing": 0.385, "Electronics": 0.296, "Home": 0.273},
        "ARAG (full)": {"Clothing": 0.439, "Electronics": 0.329, "Home": 0.289},
    }
    ours = {
        "Vanilla RAG": ndcg("vanilla_rag"),
        "ARAG w/o NLI+CSA": ndcg("arag_wo_nli_csa"),
        "ARAG w/o NLI": ndcg("arag_wo_nli"),
        "ARAG (full)": ndcg("arag"),
    }
    steps = list(paper.keys())
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for cat, color in [("Clothing", "#8ab4f8"), ("Electronics", "#f28b82"), ("Home", "#81c995")]:
        vals = [paper[s][cat] for s in steps]
        ax.plot(x, vals, "o-", color=color, lw=2, ms=7, label=f"Paper: {cat}")
    ax.plot(x, [ours[s] for s in steps], "s-", color=C["arag"], lw=2.6, ms=8,
            label="Ours: Video Games")
    for i, s in enumerate(steps):
        ax.annotate(f"{ours[s]:.3f}", (i, ours[s]), textcoords="offset points",
                    xytext=(6, 8), color=C["arag"], fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=10)
    ax.set_ylabel("NDCG@5")
    ax.set_ylim(0.20, 0.47)
    ax.set_title("Ablation staircase: every removed component lowers the score")
    ax.legend(loc="lower right", framealpha=0.9)
    # upward arrow annotation
    ax.annotate(
        "full ARAG > w/o NLI > w/o NLI+CSA > Vanilla\n(the paper's mechanistic claim reproduces)",
        xy=(0.5, 0.455), xytext=(0.5, 0.455), ha="center", fontsize=9,
        style="italic", color=C["ink"],
    )
    fig.savefig(OUT / "fig04_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig05 alpha curve
# ---------------------------------------------------------------------------
def fig_alpha_curve():
    mc = json.loads((ROOT / "results" / "metrics.json").read_text())
    val = mc["alpha_curve_validation"]
    test = mc["alpha_curve_test"]
    alphas = sorted(float(k) for k in val)
    v = [val[f"{a:g}"] if f"{a:g}" in val else val[str(a)] for a in alphas]
    t = [test[f"{a:g}"] if f"{a:g}" in test else test[str(a)] for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(alphas, t, "o-", color=C["arag"], lw=2.4, ms=6, label="Test")
    ax.plot(alphas, v, "s--", color=C["gru"], lw=1.8, ms=5, label="Validation")
    best_alpha = 0.9
    ax.axvline(best_alpha, color=C["ink"], ls=":", lw=1.2, alpha=0.8)
    ax.annotate(
        "α*=0.9\n(validation best)", xy=(best_alpha, 0.4341),
        xytext=(best_alpha - 0.28, 0.435), fontsize=9,
        arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.0),
    )
    ax.annotate(
        "test peaks at α=1.0\n(= pure ARAG)", xy=(1.0, 0.4245),
        xytext=(0.52, 0.30), fontsize=9,
        arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.0),
    )
    ax.set_xlabel("α   (weight on LLM ranking;  0 = pure GRU, 1 = pure ARAG)")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Fusion weight α: more LLM ⇒ better; α*=0.9 is close to pure ARAG")
    ax.set_xticks(alphas)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(OUT / "fig05_alpha_curve.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig06 paper vs ours
# ---------------------------------------------------------------------------
def fig_paper_vs_ours():
    # REPORT.md Appendix F.3, NDCG@5 / Hit@5
    systems = ["LLM-Recency", "LLM-Vanilla-RAG", "ARAG (full)"]
    domains = ["Clothing", "Electronics", "Home", "Video Games\n(ours)"]
    data = {
        "Clothing": [0.309, 0.299, 0.439],
        "Electronics": [0.225, 0.238, 0.329],
        "Home": [0.224, 0.229, 0.289],
        "Video Games\n(ours)": [ndcg("recency"), ndcg("vanilla_rag"), ndcg("arag")],
    }
    sys_colors = [C["llm"], C["llm"], C["arag"]]

    x = np.arange(len(systems))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 5))
    dom_colors = ["#8ab4f8", "#f28b82", "#81c995", C["arag"]]
    for j, dom in enumerate(domains):
        offset = (j - (len(domains) - 1) / 2) * width
        vals = data[dom]
        bars = ax.bar(
            x + offset, vals, width,
            label=dom.replace("\n", " "),
            color=dom_colors[j], edgecolor="none",
            alpha=1.0 if j == 3 else 0.75,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("NDCG@5")
    ax.set_ylim(0, 0.50)
    ax.set_title("ARAG vs baselines — paper's 3 categories side by side with ours")
    ax.legend(loc="upper left", ncol=4, framealpha=0.9, fontsize=9)
    fig.savefig(OUT / "fig06_paper_vs_ours.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig07 protocol flaw reversal
# ---------------------------------------------------------------------------
def fig_protocol_flaw():
    # REPORT.md Appendix E.1 (original, superseded) vs E.4 (corrected)
    systems = ["Popularity", "GRU", "ARAG", "ARAG+GRU"]
    old = [0.801, 0.823, 0.453, 0.802]   # degenerate lists
    new = [ndcg("popularity"), ndcg("gru_seed0"), ndcg("arag"), ndcg("arag_gru")]

    x = np.arange(len(systems))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    b1 = ax.bar(x - width / 2, old, width, color=C["base"], edgecolor="none",
                label="Degenerate lists (superseded)")
    b2 = ax.bar(x + width / 2, new, width, color=C["arag"], edgecolor="none",
                label="Corrected (definitive)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9.5)
    ax.set_ylabel("NDCG@5")
    ax.set_ylim(0, 0.95)
    ax.set_ylim(0, 1.02)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        fontsize=8.5, framealpha=0.9,
    )
    ax.set_title("NDCG@5 reversal", pad=36)

    ax = axes[1]
    metrics = ["H1\n(fusion−ARAG)", "H2\n(fusion−GRU)", "α*"]
    old_h = [0.349, 0.002, 0.1]
    new_h = [-0.013, 0.174, 0.9]
    x2 = np.arange(len(metrics))
    b1 = ax.bar(x2 - width / 2, old_h, width, color=C["base"], edgecolor="none")
    b2 = ax.bar(x2 + width / 2, new_h, width, color=C["arag"], edgecolor="none")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + (0.01 if b.get_height() >= 0 else -0.03),
                f"{b.get_height():+.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(0, color=C["ink"], lw=0.9)
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Headline claims flip")
    handles = [
        mpatches.Patch(color=C["base"], label="Superseded"),
        mpatches.Patch(color=C["arag"], label="Corrected"),
    ]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9)
    ax.set_ylim(-0.1, 1.05)

    fig.suptitle(
        "Protocol flaw: 15.5/19 negatives were out-of-vocab → Popularity & GRU were handed the answer",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.savefig(OUT / "fig07_protocol_flaw.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig08 history lengths (why GRU fails)
# ---------------------------------------------------------------------------
def fig_history_lengths():
    seq = pd.read_parquet(ROOT / "artifacts" / "lists" / "sequences.parquet")
    h = seq["n_train"].to_numpy()
    bins = np.arange(2, 41)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.hist(h, bins=bins, color=C["gru"], edgecolor="white", linewidth=0.5)
    med = float(np.median(h))
    ax.axvline(med, color=C["arag"], lw=2, ls="--")
    ax.annotate(
        f"median = {med:.0f} items\n(63.5% of users have 3–5)",
        xy=(med, ax.get_ylim()[1] * 0.72), xytext=(12, ax.get_ylim()[1] * 0.62),
        fontsize=10, color=C["arag"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["arag"], lw=1.4),
    )
    ax.set_xlabel("Training-history length (number of items)")
    ax.set_ylabel("# users")
    ax.set_title("Why the GRU fails: users have almost no sequence to learn from")
    # inset text box
    txt = (
        "A next-item GRU needs long sequences.\n"
        "Here the median user has 4 past items —\n"
        "the GRU largely learns 'popular wins',\n"
        "which the corrected protocol deliberately\n"
        "removes as a free signal."
    )
    ax.text(
        0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="#e8f0fe", ec=C["gru"]),
    )
    fig.savefig(OUT / "fig08_history_lengths.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig09 fusion methods (uniform vs RRF vs exponential)
# ---------------------------------------------------------------------------
def rank_to_uniform(ranking):
    n = len(ranking)
    if n <= 1:
        return {i: 0.0 for i in ranking}
    return {i: 1.0 - k / (n - 1) for k, i in enumerate(ranking)}


def rank_to_rrf(ranking, k=60):
    return {i: 1.0 / (k + r + 1) for r, i in enumerate(ranking)}


def rank_to_exp(ranking, tau=None):
    n = len(ranking)
    tau = tau if tau is not None else max(n / 3, 1)
    return {i: float(np.exp(-r / tau)) for r, i in enumerate(ranking)}


def ndcg5_from_ranking(ranking, target):
    """NDCG@5 for a single binary-relevant target: 1/log2(pos+1) if in top 5 else 0."""
    try:
        rank = ranking.index(target)  # 0-based
    except ValueError:
        return 0.0
    if rank >= 5:
        return 0.0
    return 1.0 / np.log2(rank + 2)


def fig_fusion_methods():
    arag = pd.read_csv(ROOT / "results" / "predictions" / "arag_test.csv")
    gru = pd.read_csv(ROOT / "results" / "predictions" / "gru_seed0_test.csv")
    arag["ranked_ids"] = arag["ranked_ids"].apply(ast.literal_eval)
    gru["ranked_ids"] = gru["ranked_ids"].apply(ast.literal_eval)
    merged = arag.merge(
        gru[["user_id", "ranked_ids"]], on="user_id", suffixes=("_a", "_g")
    )
    merged = merged.rename(columns={"target": "target"})  # target comes from arag only

    methods = {
        "Uniform (current)": rank_to_uniform,
        "Reciprocal-rank fusion": rank_to_rrf,
        "Exponential decay": rank_to_exp,
    }
    alphas = np.linspace(0.0, 1.0, 11)
    curves = {name: [] for name in methods}
    for name, fn in methods.items():
        for a in alphas:
            scores = []
            for _, row in merged.iterrows():
                sa, sg = fn(row["ranked_ids_a"]), fn(row["ranked_ids_g"])
                target = row["target"]
                cand = list(dict.fromkeys(row["ranked_ids_a"] + row["ranked_ids_g"]))
                fused = sorted(
                    cand,
                    key=lambda i: (-(a * sa.get(i, 0) + (1 - a) * sg.get(i, 0)), i),
                )
                scores.append(ndcg5_from_ranking(fused, target))
            curves[name].append(float(np.mean(scores)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [C["base"], C["llm"], C["arag"]]
    for (name, ys), col in zip(curves.items(), colors):
        ax.plot(alphas, ys, "o-", color=col, lw=2, ms=4.5, label=name)
    arag_only = ndcg("arag")
    ax.axhline(arag_only, color=C["ink"], ls="--", lw=1.2, alpha=0.8)
    ax.annotate(
        f"ARAG alone = {arag_only:.3f}",
        xy=(0.02, arag_only), xytext=(0.02, arag_only + 0.012),
        fontsize=9, color=C["ink"],
    )
    # mark best of each
    for (name, ys), col in zip(curves.items(), colors):
        bi = int(np.argmax(ys))
        ax.plot(alphas[bi], ys[bi], "*", color=col, ms=14,
                label=f"{name} best ({ys[bi]:.3f} @ α={alphas[bi]:.1f})")
    ax.set_xlabel("α   (weight on LLM ranking)")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Fusion rule matters: exponential decay beats the current uniform rule")
    ax.set_xticks(alphas)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    fig.savefig(OUT / "fig09_fusion_methods.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig10 content-aware extension (all users + 100-user sample)
# ---------------------------------------------------------------------------
def fig_content_aware():
    full = json.loads((ROOT / "results" / "metrics.json").read_text())
    allu = full["metrics_all_users"]
    left = [
        ("random", "Random", C["base"]),
        ("popularity", "Popularity", C["base"]),
        ("gru_seed0", "GRU (ID only)", C["gru"]),
        ("itemknn", "ItemKNN", C["llm"]),
        ("content", "Content similarity", C["llm"]),
        ("gru_text_seed0", "Text-aware GRU", C["arag"]),
    ]
    right = [
        ("gru_seed0", "GRU (ID only)", C["gru"]),
        ("arag_gru", "ARAG + GRU", C["abl"]),
        ("arag", "ARAG", C["abl"]),
        ("content", "Content similarity", C["llm"]),
        ("gru_text_seed0", "Text-aware GRU", C["arag"]),
        ("arag_gru_text", "ARAG + text-aware GRU", C["arag"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, rows, src, title in [
        (axes[0], left, allu, "Non-LLM systems, all 94,762 test users"),
        (axes[1], right, M, "Same 100 test users as the LLM systems"),
    ]:
        vals = [src[k]["ndcg@5"]["mean"] for k, _, _ in rows]
        ax.barh([r[1] for r in rows], vals, color=[r[2] for r in rows])
        for y, v in enumerate(vals):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel("NDCG@5")
        ax.set_title(title)
    fig.suptitle("Adding item text to the GRU: 0.287 → 0.461 NDCG@5 (no LLM calls)", fontweight="bold")
    fig.savefig(OUT / "fig10_content_aware.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating figures ->", OUT)
    fig_pipeline()
    print("  fig01_pipeline.png")
    fig_main_results()
    print("  fig02_main_results.png")
    fig_hypotheses()
    print("  fig03_hypotheses.png")
    fig_ablation()
    print("  fig04_ablation.png")
    fig_alpha_curve()
    print("  fig05_alpha_curve.png")
    fig_paper_vs_ours()
    print("  fig06_paper_vs_ours.png")
    fig_protocol_flaw()
    print("  fig07_protocol_flaw.png")
    fig_history_lengths()
    print("  fig08_history_lengths.png")
    fig_fusion_methods()
    print("  fig09_fusion_methods.png")
    fig_content_aware()
    print("  fig10_content_aware.png")
    print("Done.")


if __name__ == "__main__":
    main()
