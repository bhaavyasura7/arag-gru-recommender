# ARAG + GRU — Project Overview

A one-page guide to the whole project: what it asks, how it works, what it found, and where
everything lives. Details: [`REPORT.md`](REPORT.md) (research write-up) ·
[`PROCESS.md`](PROCESS.md) (engineering log with every real number) · [`../SPEC.md`](../SPEC.md)
(specification and deviations).

## 1. The question

Does a **trained deep-learning model** (a GRU over the user's item sequence) improve an
**agentic LLM recommender** (ARAG, Maragheh et al., 2025)? And how does each compare to simple
baselines when every system ranks exactly the same candidates?

## 2. Data and protocol

- **Data:** Amazon Reviews 2023, Video Games, 5-core, official leave-last-out split.
  94,762 users, 137,269 items, 814,586 interactions (each interaction is a **review**, the standard stand-in for a purchase). Median **4** training items per user.
- **Task:** for each user, rank a **frozen list of 20 items**: the true next item plus 19
  negatives drawn to be about as popular as targets, so popularity alone cannot win.
- **Metrics:** Hit@5/10, NDCG@5/10, MRR. Significance: paired bootstrap, 95% CI, 10,000 resamples.
- **Users:** LLM systems on 100 sampled test users (cost); non-LLM systems also on all 94,762.
- **Tuning:** every hyperparameter (θ, α, recency decay, GRU checkpoint) on validation only.

## 3. Systems

| Family | Systems | LLM calls / user |
|---|---|---|
| Trivial | Random, Popularity | 0 |
| Classic (extension) | Content similarity (MiniLM), ItemKNN (co-occurrence) | 0 |
| Trained deep models | **GRU** (ID embeddings), **text-aware GRU** (ID + MiniLM item vectors, extension), **SASRec** (Transformer) | 0 |
| LLM baselines | LLM-Recency, LLM-Vanilla-RAG | 1 |
| Agentic LLM | **ARAG**: UUA ∥ NLI → CSA → IRA, plus two ablations | 7 |
| Fusion | ARAG + GRU, ARAG + text-aware GRU, ARAG + content (late fusion, α on validation) | 0 extra |

```
history ──► ARAG agents (gpt-4o-mini) ──► LLM ranking ─┐
                                                       ├─► α·LLM + (1−α)·GRU ──► final top-5
history ──► GRU (trained, PyTorch)   ──► GRU ranking ─┘
```

## 4. Results (NDCG@5)

Final run (October 2026, fixed GRU initialisation — see Finding 5).

| System | All 94,762 users | Same 100 users |
|---|---|---|
| Random / Popularity | 0.149 / 0.128 | 0.125 / 0.109 |
| LLM-Recency / Vanilla-RAG | — | 0.285 / 0.263 |
| ItemKNN | 0.406 | 0.399 |
| Content similarity | 0.411 | 0.439 |
| ARAG | — | 0.424 |
| GRU, ID only (3 seeds) | 0.505 ± 0.001 | 0.520 |
| Text-aware GRU (3 seeds) | **0.530** ± 0.001 | 0.501 |
| SASRec (3 seeds) | 0.508 | **0.536** |
| ARAG + GRU (α = 0.2) | — | 0.523 |
| ARAG + text-aware GRU (α = 0.1) | — | 0.532 |

![All systems on the 100 test users](figures/fig13_sasrec_baseline.png)

## 5. Findings

1. **H1 holds:** fusing the GRU into ARAG significantly improves ARAG (+0.098, CI [+0.013, +0.182]).
   **H2 does not:** the fusion is no better than the GRU alone (+0.004, n.s.).
2. **The paper's claims reproduce:** ARAG significantly beats LLM-Recency (+0.139) and
   Vanilla-RAG (+0.162), and removing NLI+CSA significantly hurts (−0.062).
3. **Trained sequence models are more accurate than ARAG here** — a comparison the paper did not
   make. SASRec beats ARAG significantly on the 100 users (+0.112); the GRU is +0.095 (borderline).
   Text-aware GRU is the best single model on all users (+0.026 over the GRU, significant).
4. **ARAG's strengths are elsewhere:** almost no popularity bias (lift 0.13 vs ~0.77 for trained
   models), higher catalogue coverage, natural-language explanations and search.
5. **Two mistakes caught and fixed:** (a) candidate lists that let popularity win (REPORT
   Appendix E); (b) a GRU weight-initialisation bug (default N(0,1) embeddings with tied output)
   that made the GRU look weak (0.279) — we had wrongly blamed short histories. Performance is in
   fact flat across history lengths (fig12). All re-runs cost $0 (LLM cache, `REPLAY_ONLY=1`).

Limits: 100 LLM-evaluated users cannot separate the top systems (CIs ≈ ±0.08); sampled
negatives are easier than real retrieval; a single category and a single LLM.

## 6. Reproduce

```bash
make setup                      # venv + Python deps + npm
make data lists index           # dataset -> frozen lists -> MiniLM + FAISS
make gru gru-text sasrec        # ID-only GRU, text-aware GRU, SASRec (grid + 3 seeds each; CPU)
make eval                       # all systems; resumable, LLM disk-cached
make report verify              # results/*; re-derive every metric from raw predictions
make extra                      # history length, SASRec, beyond-accuracy (no LLM calls)
make app                        # demo on http://127.0.0.1:8000
```

**Zero-cost re-run:** `REPLAY_ONLY=1 OPENAI_API_KEY=replay-only-dummy make eval` replays every
LLM call from the cache and raises on any uncached call (~5 min, $0). Total project LLM spend:
**$2.04**.

## 7. Where things live

| Path | Contents |
|---|---|
| `configs/config.yaml` | every path, seed, model name, grid |
| `src/arag/dataprep/` | download, item text, sequences, frozen lists |
| `src/arag/agents/` | ARAG agents (UUA, NLI, CSA, IRA) and LLM baselines |
| `src/arag/models/` | GRU / text-aware GRU, SASRec, training, scoring |
| `src/arag/ranking/` | fusion, LLM output parser, random/popularity, content/ItemKNN |
| `src/arag/eval/` | evaluation driver, metrics, bootstrap, report, verify, extra analyses |
| `src/arag/api/`, `frontend/` | FastAPI backend and React demo |
| `results/` | `tables.md`, `report.md`, `metrics.json`, `extra_tables.md`, per-user predictions |
| `docs/figures/` | 14 figures (`scripts/make_figures.py`, `arag.eval.extra`) |
| `docs/PROJECT_GUIDE.docx` | study guide with viva Q&A (`scripts/make_guide.js`) |
