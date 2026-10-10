"""Generate results/*.md|json|png from the saved metrics.json and manifest.

Conclusions are computed from the numbers, never hand-written.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arag import util
from arag.config import get_config

ORDER = ["random", "popularity", "gru_seed0", "recency", "vanilla_rag", "arag", "arag_wo_nli", "arag_wo_nli_csa", "arag_gru"]
METRIC_ORDER = ["hit@5", "ndcg@5", "hit@10", "ndcg@10", "mrr"]


def _sig(ci: list[float]) -> str:
    lo, hi = ci
    return "yes" if (lo > 0 or hi < 0) else "no"


def main() -> None:
    logger = util.setup_logging("arag.eval.report")
    cfg = get_config()
    results_dir = cfg.path("results")
    m = util.read_json(results_dir / "metrics.json")
    manifest = util.read_json(results_dir / "manifest.json")

    metrics = m["metrics"]
    cost = m["cost"]

    # cost.json
    util.write_json(results_dir / "cost.json", cost)

    # ---- tables.md ----
    lines = ["# Results", ""]
    lines.append(f"Category: {manifest.get('category')} | model: {cfg.llm_model} "
                 f"| θ* = {m['best_theta']} | α* = {m['best_alpha']}")
    lines.append("")
    lines.append("| System | " + " | ".join(METRIC_ORDER) + " |")
    lines.append("|" + "---|" * (len(METRIC_ORDER) + 1))
    for sys in ORDER:
        if sys not in metrics:
            continue
        cells = []
        for mm in METRIC_ORDER:
            v = metrics[sys].get(mm, {})
            cells.append(f"{v.get('mean', 0):.3f} ± {v.get('std', 0):.3f}")
        lines.append(f"| {sys} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Paired differences vs reference (mean diff, 95% CI, significant)")
    lines.append("")
    lines.append("| System | vs | " + " | ".join(METRIC_ORDER) + " |")
    lines.append("|" + "---|" * (len(METRIC_ORDER) + 3))
    for key in metrics:
        if "_vs_" not in key:
            continue
        sys, ref = key.split("_vs_")
        cells = []
        for mm in METRIC_ORDER:
            v = metrics[key].get(mm)
            if v is None:
                cells.append("-")
            else:
                cells.append(f"{v['mean_diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] {_sig(v['ci'])}")
        lines.append(f"| {sys} | {ref} | " + " | ".join(cells) + " |")
    lines.append("")
    util.write_text(results_dir / "tables.md", "\n".join(lines) + "\n")

    # ---- alpha_curve.png ----
    av = {float(k): v for k, v in m["alpha_curve_validation"].items()}
    at = {float(k): v for k, v in m["alpha_curve_test"].items()}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted(av), [av[k] for k in sorted(av)], "o-", label="validation")
    ax.plot(sorted(at), [at[k] for k in sorted(at)], "s-", label="test")
    ax.axvline(m["best_alpha"], color="gray", linestyle="--", label=f"α*={m['best_alpha']}")
    ax.set_xlabel("α (ARAG weight)")
    ax.set_ylabel("NDCG@5")
    ax.set_title("ARAG + GRU fusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "alpha_curve.png", dpi=120)
    plt.close(fig)

    # ---- report.md ----
    concl = _conclusions(metrics, m)
    provider_note = (
        "Local qwen3:8b output differs from gpt-4o-mini; results are ARAG-inspired, not a paper reproduction."
        if cfg.llm_provider == "ollama"
        else "OpenAI gpt-4o-mini was used; results are ARAG-inspired, not a paper reproduction."
    )
    r = [
        "# ARAG + GRU — Results Report",
        "",
        f"**Model:** {cfg.llm_model} ({cfg.llm_provider})  ",
        f"**Category:** {manifest.get('category')}  ",
        f"**θ*:** {m['best_theta']}  **α*:** {m['best_alpha']}  ",
        f"**LLM cost:** ${cost['cost_usd']:.4f}  **Last run elapsed:** {m['elapsed_s']/60:.1f} min",
        "",
        "## Main results (test users)",
        "",
        _fmt_table(metrics),
        "",
        "## Conclusions (computed)",
        "",
        *[f"- {c}" for c in concl],
        "",
        "## Limitations",
        "",
        "- Leave-last-out means other users' later interactions can appear in GRU training.",
        "- 100 test users ⇒ wide CIs (roughly ±10 points on Hit@5).",
        "- Uniform negatives are easier than real retrieval; not an exact paper reproduction.",
        f"- {provider_note}",
        "",
        "## Protocol",
        "",
        "Frozen 20-item candidate lists (target + 19 uniform negatives) shared by all systems. ",
        "θ tuned on validation NDCG@5 over {4,5,6,7}; α tuned on validation NDCG@5 over [0,1]. ",
        "Non-LLM systems (Random, Popularity, GRU) reported on all test users; LLM systems on 100 sampled test users. ",
        "Paired-bootstrap 95% CIs (10,000 resamples).",
        "",
    ]
    util.write_text(results_dir / "report.md", "\n".join(r))
    logger.info("report artifacts written")


def _fmt_table(metrics: dict) -> str:
    out = ["| System | " + " | ".join(METRIC_ORDER) + " |", "|" + "---|" * (len(METRIC_ORDER) + 1)]
    for sys in ORDER:
        if sys not in metrics:
            continue
        cells = [f"{metrics[sys].get(mm, {}).get('mean', 0):.3f}" for mm in METRIC_ORDER]
        out.append(f"| {sys} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _conclusions(metrics: dict, m: dict) -> list[str]:
    out = []
    def diff(sys, ref, mm="ndcg@5"):
        v = metrics.get(f"{sys}_vs_{ref}", {}).get(mm)
        return v if v else None

    for sys, ref, label in [
        ("arag_gru", "arag", "ARAG+GRU vs ARAG (H1)"),
        ("arag_gru", "gru_seed0", "ARAG+GRU vs GRU (H2)"),
        ("recency", "arag", "LLM-Recency vs ARAG"),
        ("vanilla_rag", "arag", "LLM-Vanilla-RAG vs ARAG"),
        ("arag_wo_nli", "arag", "ARAG w/o NLI vs ARAG"),
        ("arag_wo_nli_csa", "arag", "ARAG w/o NLI&CSA vs ARAG"),
    ]:
        v = diff(sys, ref, "ndcg@5")
        if v is None:
            continue
        sig = "significant" if _sig(v["ci"]) == "yes" else "not significant"
        out.append(
            f"{label}: Δ NDCG@5 = {v['mean_diff']:+.3f} "
            f"(95% CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]) — {sig}"
        )
    return out


if __name__ == "__main__":
    main()
