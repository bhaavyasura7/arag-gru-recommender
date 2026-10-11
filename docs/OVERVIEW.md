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
  94,762 users, 137,269 items, 814,586 interactions. Median **4** training items per user.
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
| Trained deep models | **GRU** (ID embeddings), **text-aware GRU** (ID + MiniLM item vectors, extension) | 0 |
| LLM baselines | LLM-Recency, LLM-Vanilla-RAG | 1 |
| Agentic LLM | **ARAG**: UUA ∥ NLI → CSA → IRA, plus two ablations | 7 |
| Fusion | ARAG + GRU, ARAG + text-aware GRU, ARAG + content (late fusion, α on validation) | 0 extra |

```
history ──► ARAG agents (gpt-4o-mini) ──► LLM ranking ─┐
                                                       ├─► α·LLM + (1−α)·GRU ──► final top-5
history ──► GRU (trained, PyTorch)   ──► GRU ranking ─┘
```

## 4. Results (NDCG@5)

| System | All 94,762 users | Same 100 users |
|---|---|---|
| Random / Popularity | 0.149 / 0.128 | 0.125 / 0.109 |
| GRU, ID only | 0.279 (3 seeds) | 0.238 |
| LLM-Recency / Vanilla-RAG | — | 0.285 / 0.263 |
| ItemKNN | 0.406 | 0.399 |
| ARAG + GRU | — | 0.412 |
| Content similarity | 0.411 | 0.439 |
| ARAG | — | 0.424 |
| **Text-aware GRU** | **0.461** (3 seeds, ±0.002) | 0.462 |
| **ARAG + text-aware GRU** | — | **0.468** |

![Content-aware extension](figures/fig10_content_aware.png)

## 5. Findings

1. **Original study:** fusing the ID-only GRU into ARAG does **not** help (H1: −0.013, n.s.).
   ARAG significantly beats the ID-only GRU, LLM-Recency, Vanilla-RAG, Popularity and Random, and
   the paper's NLI+CSA ablation reproduces significantly.
2. **Why the GRU lost:** with ~4 items per user, item IDs are too sparse. The LLM reads item text.
3. **Extension:** adding MiniLM item text to the *same* GRU raises it **0.279 → 0.461** on all
   users (+0.174, CI [+0.172, +0.176]). It beats content similarity and ItemKNN significantly on
   all users and is numerically above ARAG on the 100-user sample (+0.038, n.s.).
4. **Cheap beats expensive here:** content similarity (no training, no API) ties with ARAG, and
   the text-aware GRU scores every user in seconds for $0. ARAG costs 7 LLM calls per user.
5. **Methodology matters:** an earlier version of the candidate lists made popularity look
   strong and produced a false "H1 supported" result; it was caught, fixed, and documented
   (REPORT Appendix E).

Limits: 100 LLM-evaluated users cannot separate the top systems (CIs ≈ ±0.08–0.10); sampled
negatives are easier than real retrieval; a single category.

## 6. Reproduce

```bash
make setup                      # venv + Python deps + npm
make data lists index           # dataset -> frozen lists -> MiniLM + FAISS
make gru gru-text               # ID-only GRU (grid + 3 seeds), text-aware GRU (3 seeds, ~15 min)
make eval                       # all systems; resumable, LLM disk-cached
make report verify              # results/*; re-derive every metric from raw predictions
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
| `src/arag/models/` | GRU / text-aware GRU, training, scoring |
| `src/arag/ranking/` | fusion, LLM output parser, random/popularity, content/ItemKNN |
| `src/arag/eval/` | evaluation driver, metrics, bootstrap, report, verify |
| `src/arag/api/`, `frontend/` | FastAPI backend and React demo |
| `results/` | `tables.md`, `report.md`, `metrics.json`, per-user predictions |
| `docs/figures/` | 10 presentation figures (`scripts/make_figures.py`) |
