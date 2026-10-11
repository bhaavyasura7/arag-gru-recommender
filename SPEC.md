# ARAG + GRU — Project Specification (v3, final)

Self-contained build spec. A developer or coding agent should be able to build the whole project from this file alone. It consolidates every decision made so far; where it differs from earlier drafts, this file wins.

## 0. Instructions for the agent building this

1. Read this whole file before writing code.
2. Build the **entire system in one pass** (Phase A research core + Phase B product demo). There is no separate "debug run" phase.
3. **Never fabricate results.** Every number in `results/` must come from a real run on real data with the real LLM. If the dataset cannot be downloaded or the model is unreachable, build and test the code, then stop and say exactly what is missing. Do not commit stub or fake metrics, traces or demo data.
4. Unit and integration tests use small hand-made fixtures. Tests hit a local model (see deviations); their outputs must never be written to `results/` or `artifacts/`.
5. Work on the branch you were given; commit in logical steps with clear messages.
6. When done, report: what was built, test results, the actual metrics table (if the real run happened), actual LLM cost, and anything not done.

## 1. What the project is

An ARAG-style agentic recommender (Maragheh et al., 2025, arXiv:2506.21931) on Amazon Reviews 2023 (Video Games), extended with **one deep-learning contribution: a trained PyTorch GRU sequential user model fused into the final ranking**. Plus a web demo with a personalised feed, product search, feedback and persistent memory.

- **Primary question:** on identical frozen candidate lists, does fusing a GRU score into ARAG's LLM ranking improve Hit@5 / NDCG@5 over ARAG alone and over the GRU alone?
- **H1:** ARAG+GRU > ARAG (α = 1). **H2:** ARAG+GRU > GRU alone (α = 0).
- **Secondary:** check the paper's ordering (LLM-Recency, LLM-Vanilla-RAG < ARAG) and its two ablations on our data; report cost/latency.
- This is **ARAG-inspired, not an exact reproduction**: the paper does not publish prompts, NLI threshold, candidate-list size or code. Never compare our numbers directly with the paper's Table 1.
- Negative or mixed results are valid outcomes and must be reported as measured.

## 2. Fixed decisions

| Topic | Decision |
|---|---|
| Dataset | Amazon Reviews 2023 (McAuley Lab), category **Video_Games**. Fallback **All_Beauty** only if Video_Games 5-core is unusable (record why). |
| Split | The dataset's official **rating-only 5-core, leave-last-out** benchmark split (last item = test, second-to-last = validation, rest = train). |
| Item text | Product metadata only (title, main_category, categories, features, description), truncated to ~120 words. No review text. |
| LLM | Local **qwen3:8b** via Ollama (see deviations). Temperature 0, structured JSON output (JSON schema), pinned model string in config. |
| Embeddings | sentence-transformers **all-MiniLM-L6-v2**, normalised, FAISS `IndexFlatIP`. |
| Deep learning | PyTorch **GRU4Rec-style** model, trained on train sequences only. |
| Evaluation users | **100 test users** and **100 validation users** for every LLM-based system (fixed seed). Non-LLM systems (Random, Popularity, GRU) are additionally reported on **all** test users. |
| Candidate lists | Per user: **target + 19 negatives = 20 items**, frozen and shared by every system. |
| Primary metrics | Hit@5, NDCG@5; also Hit@10, NDCG@10, MRR; paired-bootstrap 95% CIs. |
| Orchestration | Plain async Python for the agent pipeline (LangGraph not required). |
| API / storage / UI | FastAPI + Pydantic, **SQLite**, **React + Vite** frontend. |
| Experiment log | JSON run records (MLflow not used). |
| Python | 3.11+ (3.13 fine), dependencies pinned (`requirements.txt`). |
| Budget | Local model ⇒ $0 LLM cost. Hard stop if spend estimate exceeds $5 (OpenAI provider only). |

## 3. Repository layout

```
README.md  SPEC.md  Makefile  requirements.txt  .env.example  .gitignore
configs/
  config.yaml              # all paths, seeds, sizes, model names, θ grid, α grid
  prompts/                 # uua.txt nli.txt csa.txt ira.txt recency.txt vanilla_rag.txt
data/                      # git-ignored: raw/ processed/
artifacts/                 # git-ignored: lists/ gru/ faiss/ llm_cache/ predictions/ app.db
results/                   # tracked: metrics.json, tables.md, alpha_curve.png, cost.json, run_record.json, report.md
src/arag/
  config.py
  dataprep/  download.py  load.py  items.py  lists.py  manifest.py
  models/    gru.py  train_gru.py  score.py
  retrieval/ embed.py  index.py
  llm/       client.py  cache.py  schemas.py
  agents/    context.py  uua.py  nli.py  csa.py  ira.py  arag.py  baselines.py
  ranking/   parser.py  fusion.py  simple.py      # random, popularity
  eval/      metrics.py  bootstrap.py  run.py  report.py  verify.py
  memory/    db.py  repo.py                        # SQLite
  api/       app.py  routes.py  schemas.py  services.py
frontend/                  # React + Vite
tests/
```

**.gitignore rule:** anchor ignores to the repo root (`/data/`, `/artifacts/`, `/.venv/`, `node_modules/`, `.env`). A bare `data/` rule silently ignores *any* folder named `data`, including source packages. Name the code package `dataprep/`, not `data/`. Add a test that every module under `src/` is tracked by git.

## 4. Data pipeline (`src/arag/dataprep`)

1. **Download** from Hugging Face dataset `McAuley-Lab/Amazon-Reviews-2023`:
   - the rating-only 5-core leave-last-out benchmark files for Video_Games (`benchmark/5core/last_out/Video_Games.{train,valid,test}.csv`), and
   - the item metadata for Video_Games (`raw/meta_categories/meta_Video_Games.jsonl`).
   Verify exact file names/configs against the dataset card before coding the loader; do not guess silently. Save under `data/raw/`.
2. **Manifest** (`data/manifest.json`, tracked copy in `results/`): dataset id, category, files, download date, row counts, SHA-256 per file.
3. **Load** into Parquet tables:
   - `interactions(user_id, item_id, timestamp, rating, split)`; item_id = `parent_asin`.
   - `items(item_id, title, text, main_category, price)`.
   Log row counts at every step; drop rows with missing ids/timestamps and log how many (no silent row loss). Items without metadata/title: keep them in interactions but mark `has_text=false`; log the count.
4. **Sequences** per user, sorted by (timestamp, item_id):
   - train sequence = train-split interactions;
   - validation target = the valid-split item; its history = train sequence;
   - test target = the test-split item; its history = train sequence + validation item.
   Assert for every user: max(train ts) ≤ valid ts ≤ test ts, and targets are not in their own history.
5. **Context for prompts** (`agents/context.py`), built only from history strictly before the target.
6. **User samples** (seed 42): 100 validation users and 100 test users, drawn from users whose target item has text. Save ids to `artifacts/lists/users.json`.
7. **Frozen candidate lists** (`dataprep/lists.py`), for every validation and test user (all users):
   - 19 negatives sampled (seed 42, **popularity-weighted** — proportional to train interaction count) from items **with train interactions** that the user **never** interacted with in any split;
   - add the target, shuffle with a seeded RNG, store the order;
   - save `artifacts/lists/{valid,test}_lists.parquet` + SHA-256; never regenerate after evaluation starts.
   - Random-ranking reference: Hit@5 = 0.25.
   - **Amendment (post-review fix).** The original run sampled negatives uniformly from *all catalogue items with text*. ~81% of those items have zero interactions, so ~15.5 of 19 negatives per list were unknown to Popularity/the GRU (shared OOV embedding / tied zero counts) and the task collapsed to a ~4.5-way choice — making Popularity Hit@5 = 0.97 and inflating any GRU-vs-LLM gap. Negatives are now drawn from the 25,527 items with train interactions, weighted by popularity so they are about as popular as targets. See `docs/REPORT.md` Appendix E (errata).

## 5. Systems compared

All systems rank the **same frozen 20-item list per user**.

| System | Definition | LLM calls/user |
|---|---|---|
| Random | Seeded random permutation | 0 |
| Popularity | Order by train interaction count | 0 |
| GRU (α = 0) | Order by GRU score | 0 |
| LLM-Recency (paper baseline) | Last 10 history items + 20 candidates in one ranking prompt | 1 |
| LLM-Vanilla-RAG (paper baseline) | The 10 history items most similar (MiniLM cosine) to the mean candidate embedding + 20 candidates in one ranking prompt | 1 |
| ARAG (α = 1) | UUA ∥ NLI → CSA → IRA (§6) | 7 |
| ARAG w/o NLI (paper ablation) | CSA summarises all 20 candidates; then IRA | 2 (UUA reused) |
| ARAG w/o NLI & CSA (paper ablation) | IRA gets S_user + raw candidate list | 1 (UUA reused) |
| **ARAG + GRU (proposed)** | Late fusion of ARAG ranking and GRU score (§7.3) | 0 extra (reuses ARAG) |

## 6. ARAG agents (`src/arag/agents`)

**6.1 UUA** — input C_lt, C_st. Output `{long_term_interests, current_intent, avoid, summary}`.
**6.2 NLI** — input raw user context + 5 candidates per call (4 calls). Output per candidate `{item_id, label, score, rationale}`. Accepted set I⁺ = score ≥ θ, θ ∈ {4,5,6,7} (default 6). Missing/unknown ids: re-ask once; if still missing, score 5 + `nli_missing`.
**6.3 CSA** — input S_user, item text + NLI scores of I⁺. Output `{summary}`. Empty I⁺ → top 3 by score, `csa_empty_fallback = true`.
**6.4 IRA** — input S_user, S_ctx, all 20 candidates. Output `{ranking: [20 ids], reasons}`.
**6.5 Ranking parser** (`ranking/parser.py`) — drop unknown ids, drop duplicates (keep first), append missing ids (by NLI score, else list order). Count each repair type.
**6.6 Execution**: `parallel: S_user = UUA(...), nli = NLI(...)` then `S_ctx = CSA(...)` then `π = IRA(...)`. Up to 8 users concurrently.
**6.7 LLM client** (`llm/client.py`): OpenAI SDK, temperature 0, JSON-schema response format, timeout, exponential backoff on 429/5xx. On invalid JSON/schema: one retry with the validation error appended; then the documented fallback, counted. Disk cache keyed by SHA-256 of (model, prompt, schema). Per-call log: agent, tokens in/out, latency, cache hit, error.

## 7. GRU model and fusion

**7.1 Model** — item embedding 64-d → 1-layer GRU (hidden 128) → dropout 0.2 → user vector h_u. Index 0 = padding, 1 = OOV. Input embeddings tied to output embeddings.
**7.2 Training** — last 50 train items, left-padded, masked. Full-softmax next-item prediction over the vocab, every prefix. Adam lr 1e-3, batch 256, up to 30 epochs, early stopping (patience 3) on validation NDCG@5. Grid: hidden {64,128}, dropout {0.2,0.5}. 3 seeds (0,1,2). Save `artifacts/gru/{seed}/`.
**7.3 Fusion** — `s_llm = 1 - (rank_IRA-1)/19`, `s_gru = 1 - (rank_GRU-1)/19`, `final = α·s_llm + (1-α)·s_gru`, α ∈ {0.0…1.0}, ties by NLI score then item_id. α chosen on validation NDCG@5.

## 8. Evaluation

- Hit@K, NDCG@K, MRR. Paired bootstrap: 10,000 resamples, 95% CI of per-user metric difference.
- Retrieval Recall@{20,50,100} (no LLM, all test users).
- Cost: calls, tokens, USD, seconds per user per system; parser repair rate and fallback counts.
- Per-user predictions `artifacts/predictions/{system}.parquet` + tracked CSV copy.
- `results/report.md` generated from saved artifacts only. Text conclusions computed from numbers.
- Limitations: leave-last-out leakage into GRU training; 100 users ⇒ wide CIs; sampled negatives easier than real retrieval candidates; not an exact reproduction.

## 9. Phase B — product demo

As in the interactive demo description: SQLite memory, search/feed/feedback/profile/replay flows, FastAPI routes, React + Vite frontend.

## 10. Budget

Local Ollama qwen3:8b ⇒ $0 LLM cost. OpenAI provider pricing kept in config for optional use.

## 11. Tests

See `tests/`. Deterministic logic is unit-tested with hand-made fixtures; integration tests hit the local Ollama model.

## 12. Commands (`Makefile`)

`setup freeze data lists index gru eval report verify test app all clean`.

## 13. Definition of done

`make all` on a fresh clone (network + local Ollama) produces `results/report.md`, `metrics.json`, tables, α curve, cost table, run record. All systems evaluated on the same frozen lists. `make test` passes (requires local Ollama). `make app` runs the demo. No fabricated or stub-generated numbers.

---

## Recorded deviations from the original gpt-4o-mini spec

1. **LLM provider:** configurable — the `llm/client.py` supports both OpenAI and local Ollama; `config.yaml` selects the provider. The committed results were produced with **`gpt-4o-mini`** (~$0.99, §10). Local **`qwen3:8b`** via Ollama is also supported (zero cost; used by the test-suite, which forces the local provider so `make test` needs no API key).
2. **Budget:** `budget.max_usd` guards the OpenAI provider; the local provider is $0. Evaluation cost is reported in `results/cost.json` from real cached usage.
3. **`llm/fake.py` removed:** no fake LLM client. Unit tests use hand-made fixtures for deterministic logic; integration tests call the local Ollama model directly. `make test` therefore requires Ollama running (still no API key).
4. **Data paths verified** against the dataset card: benchmark files are `benchmark/5core/last_out/Video_Games.{train,valid,test}.csv` (columns `user_id,parent_asin,rating,timestamp`); metadata is `raw/meta_categories/meta_Video_Games.jsonl` (the `raw_meta_Video_Games` config does not exist for Video_Games).
5. **`make verify` added:** an audit target that recomputes metrics from `artifacts/predictions/` and diffs against `results/`, checks list SHA-256 and run-record consistency, to guarantee no fabricated numbers can survive.
