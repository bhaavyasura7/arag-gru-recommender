"""Generate the LaTeX tables for docs/paper/ from results/metrics.json.

Every number in the paper's results tables is read from the audited metrics file, so the
paper cannot drift from the pipeline. Re-run after `make eval && make report`.

Usage:  .venv/bin/python scripts/make_paper_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MJ = json.loads((ROOT / "results" / "metrics.json").read_text())
M, ALL = MJ["metrics"], MJ["metrics_all_users"]
EXTRA = json.loads((ROOT / "results" / "extra_metrics.json").read_text())  # make extra
OUT = ROOT / "docs" / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = ["hit@5", "ndcg@5", "hit@10", "ndcg@10", "mrr"]
NAMES = {
    "random": "Random",
    "popularity": "Popularity",
    "content": "Content similarity",
    "itemknn": "ItemKNN",
    "gru_seed0": "GRU, ID only (seed 0)",
    "gru_seed1": "GRU, ID only (seed 1)",
    "gru_seed2": "GRU, ID only (seed 2)",
    "gru_text_seed0": "Text-aware GRU (seed 0)",
    "gru_text_seed1": "Text-aware GRU (seed 1)",
    "gru_text_seed2": "Text-aware GRU (seed 2)",
    "recency": "LLM-Recency",
    "vanilla_rag": "LLM-Vanilla-RAG",
    "arag": "ARAG",
    "arag_wo_nli": "ARAG w/o NLI",
    "arag_wo_nli_csa": "ARAG w/o NLI \\& CSA",
    "arag_gru": "ARAG + GRU",
    "arag_gru_text": "ARAG + text-aware GRU",
    "arag_content": "ARAG + content",
}
GROUPS = [
    ("Non-learned", ["random", "popularity", "content", "itemknn"]),
    ("Trained sequential models", ["gru_seed0", "gru_seed1", "gru_seed2", "gru_text_seed0", "gru_text_seed1", "gru_text_seed2"]),
    ("LLM baselines and ablations", ["recency", "vanilla_rag", "arag_wo_nli_csa", "arag_wo_nli"]),
    ("Agentic LLM and fusion", ["arag", "arag_gru", "arag_content", "arag_gru_text"]),
]


def mean(src, sys, m):
    return src[sys][m]["mean"]


def write(name, body):
    (OUT / f"{name}.tex").write_text(body + "\n", encoding="utf-8")
    print("  tables/" + name + ".tex")


def main_table():
    """All systems on the same 100 test users; best value per column in bold."""
    systems = [s for _, g in GROUPS for s in g if s in M]
    best = {m: max(mean(M, s, m) for s in systems) for m in METRICS}
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "System & Hit@5 & NDCG@5 & Hit@10 & NDCG@10 & MRR \\\\",
        "\\midrule",
    ]
    for gi, (label, group) in enumerate(GROUPS):
        if gi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for s in group:
            if s not in M:
                continue
            cells = []
            for m in METRICS:
                v = mean(M, s, m)
                cells.append(f"\\textbf{{{v:.3f}}}" if abs(v - best[m]) < 1e-9 else f"{v:.3f}")
            lines.append(f"\\quad {NAMES[s]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("main_results", "\n".join(lines))


def all_users_table():
    import numpy as np

    rows = [("random", ["random"]), ("popularity", ["popularity"]), ("itemknn", ["itemknn"]),
            ("content", ["content"]), ("gru", ["gru_seed0", "gru_seed1", "gru_seed2"]),
            ("gru_text", ["gru_text_seed0", "gru_text_seed1", "gru_text_seed2"])]
    label = {"random": "Random", "popularity": "Popularity", "itemknn": "ItemKNN",
             "content": "Content similarity", "gru": "GRU, ID only (3 seeds)",
             "gru_text": "Text-aware GRU (3 seeds)"}
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "System & Hit@5 & NDCG@5 & MRR \\\\", "\\midrule"]
    for key, seeds in rows:
        cells = []
        for m in ["hit@5", "ndcg@5", "mrr"]:
            v = [ALL[s][m]["mean"] for s in seeds]
            txt = f"{np.mean(v):.3f}" + (f" $\\pm$ {np.std(v):.3f}" if len(v) > 1 else "")
            cells.append(f"\\textbf{{{txt}}}" if key == "gru_text" else txt)
        lines.append(f"{label[key]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("all_users", "\n".join(lines))


def paired_table():
    """Key paired-bootstrap contrasts, ΔNDCG@5 = first − second."""
    def get(sys, ref, flip=False):
        d = M[f"{sys}_vs_{ref}"]["ndcg@5"]
        md, lo, hi = d["mean_diff"], d["ci"][0], d["ci"][1]
        if flip:
            md, lo, hi = -md, -hi, -lo
        return md, lo, hi

    rows = [
        ("\\textbf{H1}: ARAG + GRU $-$ ARAG", get("arag_gru", "arag")),
        ("\\textbf{H2}: ARAG + GRU $-$ GRU", get("arag_gru", "gru_seed0")),
        ("ARAG $-$ LLM-Recency", get("recency", "arag", True)),
        ("ARAG $-$ LLM-Vanilla-RAG", get("vanilla_rag", "arag", True)),
        ("ARAG $-$ ARAG w/o NLI", get("arag_wo_nli", "arag", True)),
        ("ARAG $-$ ARAG w/o NLI \\& CSA", get("arag_wo_nli_csa", "arag", True)),
        ("ARAG $-$ GRU (seed 1)", get("gru_seed1", "arag", True)),
        ("ARAG $-$ Popularity", get("popularity", "arag", True)),
        ("\\midrule", None),
        ("Text-aware GRU $-$ GRU", get("gru_text_seed0", "gru_seed0")),
        ("Text-aware GRU $-$ ARAG", get("gru_text_seed0", "arag")),
        ("Content similarity $-$ ARAG", get("content", "arag")),
        ("ItemKNN $-$ ARAG", get("itemknn", "arag")),
        ("\\textbf{H1$'$}: ARAG + text-GRU $-$ ARAG", get("arag_gru_text", "arag")),
        ("\\textbf{H2$'$}: ARAG + text-GRU $-$ text-GRU", get("arag_gru_text", "gru_text_seed0")),
        ("ARAG + content $-$ ARAG", get("arag_content", "arag")),
    ]
    lines = ["\\begin{tabular}{lrcc}", "\\toprule",
             "Contrast & $\\Delta$NDCG@5 & 95\\% CI & Sig. \\\\", "\\midrule"]
    for label, v in rows:
        if v is None:
            lines.append(label)
            continue
        md, lo, hi = v
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"{label} & ${md:+.3f}$ & $[{lo:+.3f}, {hi:+.3f}]$ & {sig} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("paired", "\n".join(lines))


def tuning_table():
    tv = MJ["theta_grid"]
    av = MJ["alpha_curve_validation"]
    at = MJ["alpha_curve_test"]
    avt = MJ["alpha_curve_validation_text"]
    avc = MJ["alpha_curve_validation_content"]
    keys = ["0.0", "0.2", "0.3", "0.4", "0.6", "0.8", "0.9", "1.0"]
    lines = ["\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
             "$\\alpha$ & " + " & ".join(keys) + " \\\\", "\\midrule"]
    for label, curve, best in [
        ("ARAG + GRU, validation", av, MJ["best_alpha"]),
        ("ARAG + GRU, test", at, None),
        ("ARAG + text-GRU, validation", avt, MJ["best_alpha_text"]),
        ("ARAG + content, validation", avc, MJ["best_alpha_content"]),
    ]:
        cells = []
        for k in keys:
            v = curve.get(k)
            if v is None:
                cells.append("--")
            elif best is not None and abs(float(k) - best) < 1e-9:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("alpha", "\n".join(lines))
    th = ", ".join(f"$\\theta={k}$: {v:.4f}" for k, v in sorted(tv.items(), key=lambda t: int(t[0])))
    (OUT / "theta.tex").write_text(th + "\n", encoding="utf-8")
    print("  tables/theta.tex")


XN = {"gru": "GRU (ID only)", "gru_text": "Text-aware GRU", "sasrec": "SASRec", "content": "Content similarity",
      "itemknn": "ItemKNN", "popularity": "Popularity", "recency": "LLM-Recency", "vanilla_rag": "LLM-Vanilla-RAG",
      "arag": "ARAG", "arag_gru": "ARAG + GRU", "arag_gru_text": "ARAG + text-GRU"}


def extra_tables():
    """History-length, SASRec and beyond-accuracy tables from results/extra_metrics.json."""
    sys_h = ["content", "itemknn", "gru", "gru_text", "sasrec"]
    lines = ["\\begin{tabular}{lr" + "c" * len(sys_h) + "}", "\\toprule",
             "History & Users & " + " & ".join(XN[x] for x in sys_h) + " \\\\", "\\midrule"]
    for label, row in EXTRA["history_all"].items():
        lines.append(label.replace("–", "--") + f" & {row['n_users']:,}".replace(",", "{,}") + " & "
                     + " & ".join(f"{row[x]['mean']:.3f}" for x in sys_h) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("history", "\n".join(lines))

    order = ["popularity", "itemknn", "content", "gru", "gru_text", "sasrec", "recency", "vanilla_rag",
             "arag", "arag_gru", "arag_gru_text"]
    b = EXTRA["beyond_accuracy_100"]
    o = EXTRA["overall_100"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "System & NDCG@5 & Popularity lift & Diversity (ILD) & Coverage \\\\", "\\midrule"]
    for x in order:
        if x in b:
            lines.append(f"{XN[x]} & {o[x]['ndcg@5']['mean']:.3f} & {b[x]['popularity_lift']['mean']:.3f} & "
                         f"{b[x]['diversity_ild']['mean']:.3f} & {b[x]['coverage']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("beyond", "\n".join(lines))

    lines = ["\\begin{tabular}{lrc}", "\\toprule", "Contrast ($\\Delta$NDCG@5) & Users & 95\\% CI \\\\", "\\midrule"]
    for name, t in EXTRA["sasrec_tests"].items():
        lo, hi = t["ci"]
        sig = "$^{*}$" if t["significant"] else ""
        lines.append(f"{name} & {t['n']:,}".replace(",", "{,}") + f" & ${t['mean_diff']:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]{sig} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("sasrec", "\n".join(lines))


if __name__ == "__main__":
    print("Generating paper tables ->", OUT)
    main_table()
    all_users_table()
    paired_table()
    tuning_table()
    extra_tables()
