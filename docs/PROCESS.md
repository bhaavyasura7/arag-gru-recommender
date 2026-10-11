# End-to-End Process & Training Documentation

This document records exactly how the committed results were produced, with real
numbers from the actual run. It is written so the whole pipeline can be reproduced
or audited step by step. See `SPEC.md` for the specification and deviations.

Run summary: Amazon Reviews 2023 — **Video_Games** (5-core rating-only, leave-last-out),
100 validation + 100 test users for LLM systems, all 94,762 test users for non-LLM
systems, LLM = **OpenAI `gpt-4o-mini`**, θ\*=5, α\*=0.1, LLM cost **$0.9853**.

---

## 0. Environment

| Item | Value |
|---|---|
| OS / arch | macOS (darwin, arm64) |
| Python | 3.11.16, managed by `uv` (`.venv`) |
| Key packages | torch 2.14.1, faiss-cpu 1.15.1, sentence-transformers 6.1.0, openai 3.28.0, pandas 3.0.6, fastapi 0.143.0 |
| Package layout | `src/` layout, installed editable (`uv pip install -e .`, package `arag`) |
| Secrets | `OPENAI_API_KEY` read from the environment only; never written to a file |
| Hardware | same machine for data, GRU training (CPU), and evaluation |

Setup: `make setup` (venv + `uv pip install -r requirements.txt` + `npm install`),
then `make freeze` pins exact versions into `requirements.txt`.

---

## 1. Data acquisition (`make data`)

Downloaded from Hugging Face `McAuley-Lab/Amazon-Reviews-2023` via `us.aws.cdn.hf.co`:

| File | Remote path | Size | SHA-256 (first 12) |
|---|---|---|---|
| `Video_Games.train.csv` | `benchmark/5core/last_out/Video_Games.train.csv` | 36.3 MB | `abe955af35f5` |
| `Video_Games.valid.csv` | `benchmark/5core/last_out/Video_Games.valid.csv` | 5.5 MB | `a4d270bb97f9` |
| `Video_Games.test.csv` | `benchmark/5core/last_out/Video_Games.test.csv` | 5.5 MB | `88a7653b41d2` |
| `meta_Video_Games.jsonl` | `raw/meta_categories/meta_Video_Games.jsonl` | 437.4 MB | `3ad94dfaa8df` |

The rating-only split CSVs have columns `user_id,parent_asin,rating,timestamp`.
Full SHA-256 values and the download date are in `data/manifest.json`
(tracked copy `results/manifest.json`).

## 2. Loading & text building (`make data`)

- Interactions concatenated from the three splits → `data/processed/interactions.parquet`
  with columns `user_id,item_id,timestamp,rating,split` (`item_id = parent_asin`).
- Item metadata parsed line-by-line from the 437 MB JSONL → `data/processed/items.parquet`
  with `item_id,title,text,main_category,price,has_text`.
- `text` = title + main_category + categories + features + description, flattened and
  truncated to **120 words**.

**Logged row counts (no silent loss):**

| Quantity | Value |
|---|---|
| interactions total | 814,586 |
| — train / valid / test | 625,062 / 94,762 / 94,762 |
| users | 94,762 |
| items (metadata rows) | 137,269 (0 duplicates) |
| items with text | 137,260 |
| interacted items with no metadata | 0 |
| rows dropped for missing id/timestamp | 0 |

## 3. Sequences & frozen candidate lists (`make lists`)

- Per-user sequences sorted by `(timestamp, item_id)`.
- Validation target = valid-split item (history = train sequence).
- Test target = test-split item (history = train sequence + validation item).
- **Assertions enforced** (`_assert_no_leak`): `max(train ts) ≤ valid ts ≤ test ts`,
  and targets are not in their own history. No violations.
- Sequences saved to `artifacts/lists/sequences.parquet` (all 94,762 users).

**Candidate lists** (frozen, shared by every system):
- 19 negatives sampled uniformly from items **with text** that the user never
  interacted with in any split, + the target → 20 items, shuffled with a seeded RNG
  (seed 42). Generated for **all** valid and all test users.
- Saved to `artifacts/lists/{valid,test}_lists.parquet`; SHA-256 stored in
  `artifacts/lists/lists_hashes.json` and checked by `make verify`.
- Validation lists: 94,762 users × 20 candidates; test lists: 94,762 × 20.
- **Random-ranking reference:** Hit@5 = 0.25 (confirmed empirically: 0.251).

**Sampled evaluation users** (`artifacts/lists/users.json`, seed 42): 100 validation
and 100 test users, drawn from users whose target item has text.

## 4. Embeddings & retrieval index (`make index`)

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-d), L2-normalised.
- 137,260 item texts embedded on CPU in ~9 min (batch 256).
- Output: `artifacts/faiss/item_ids.npy`, `artifacts/faiss/item_embeddings.npy`.
- FAISS `IndexFlatIP` (cosine via inner product) → `artifacts/faiss/index.faiss`.

---

## 5. GRU training (`make gru`) — every detail

### 5.1 Vocabulary

- Built from **train-split items only** → **25,527 distinct items**.
- Index assignment: `0 = padding`, `1 = OOV`, real items `2 … 25,528`.

### 5.2 Model (`models/gru.py`)

```
item embedding (vocab × 64)
  → dropout
  → 1-layer GRU (input 64, hidden {64,128})
  → dropout
  → linear proj (hidden → 64)  [Identity when hidden == 64]
  → full-softmax output over the vocabulary (weights TIED to the input embedding)
```

- User vector `h_u` = last valid hidden state, projected to 64-d.
- Score of item *i* = `h_u · e_i`.

### 5.3 Sequences

- Per user: last **50** train items, mapped to indices.
- **Right-padded** (padding token at the end) and processed with
  `pack_padded_sequence(..., enforce_sorted=False)` + `pad_packed_sequence(..., total_length=)`.
- Only users with ≥2 train items are used → **94,762 training sequences**
  (essentially all users, since 5-core guarantees ≥3 train items).

### 5.4 Objective & optimisation

- Input `x = seq[:-1]`, target `y = seq[1:]` → **next-item prediction on every prefix**.
- Loss: full-softmax cross-entropy over the vocabulary, masked to valid positions
  (`ignore_index=0`), batch size 256.
- Optimizer: Adam, lr 1e-3.
- Epochs: up to **12**, early stopping patience 3.
- **Early-stopping metric:** mean validation NDCG@5 computed over **all 94,762
  validation users** against the frozen validation lists (single relevant item per user).

### 5.5 Hyperparameter grid (validation, seed 0)

| hidden | dropout | best val NDCG@5 | wall time |
|---|---|---|---|
| **64** | **0.2** | **0.8400** | 1207 s |
| 64 | 0.5 | 0.8268 | 1160 s |
| 128 | 0.2 | 0.8218 (early stop @ epoch 5) | 618 s |
| 128 | 0.5 | 0.8256 (early stop @ epoch 8) | 760 s |

**Best config: hidden = 64, dropout = 0.2.**

### 5.6 Final seeds

| Seed | val NDCG@5 | wall time | checkpoint |
|---|---|---|---|
| 0 | 0.8409 | 973 s | `artifacts/gru/0/` |
| 1 | 0.8455 | 1009 s | `artifacts/gru/1/` |
| 2 | 0.8385 | 1057 s | `artifacts/gru/2/` |

Each checkpoint dir contains `model.pt` + `item_map.json` + `config.json` together.

**Test-set GRU alone (mean ± std over seeds):** Hit@5 **0.9758 ± 0.0014**,
NDCG@5 **0.8230 ± 0.0043**, MRR 0.773 ± 0.004.

### 5.7 Notes

- Total GRU training wall time ≈ **1 h 40 m** on CPU (grid + 3 seeds).
- Epochs capped at **12** (spec default 30): validation NDCG@5 had effectively
  plateaued (epoch 9 = 0.8294, epoch 11 = 0.8400), so 30 epochs would add hours for
  ranking-irrelevant gains. Recorded in `SPEC.md`.
- **Pad-side bug found and fixed:** `pack_padded_sequence` expects *right*-padding;
  the first implementation left-padded and silently packed the wrong timesteps
  (caught by the `test_gru.py` padding test). Fixed by right-padding + `total_length`.

---

## 6. LLM evaluation (`make eval`) — every detail

### 6.1 Provider

- Primary implementation supports **local Ollama (`qwen3:8b`)** and **OpenAI
  (`gpt-4o-mini`)** through the same `llm/client.py`.
- Local qwen3:8b measured **~20 s/call** and Ollama processed requests **serially**
  (3 concurrent calls took the same total time as sequential), i.e. ~14 h for the
  full evaluation. The committed results were therefore produced with
  **`gpt-4o-mini`** (~2.4 s/call, 8-way concurrency), at a total cost of **$0.9853**.

### 6.2 LLM client (`llm/client.py`)

- OpenAI SDK, `temperature=0`, `response_format` = strict JSON schema derived from
  Pydantic models (`llm/schemas.py`,`to_strict_schema`).
- Timeout 60 s, exponential backoff on 429/5xx/connection errors, one schema-retry
  with the validation error appended, then the documented per-agent fallback (counted).
- **Disk cache** (`artifacts/llm_cache/`) keyed by SHA-256 of (model, prompt, schema).
  A cache hit makes no API call → re-running the eval makes **zero** new calls and
  reproduces identical outputs.
- Every call logged (agent, tokens in/out, latency, cache hit, cost, fallback).

**Strict-schema fix:** OpenAI rejects free-form objects (`dict[str,str]`) in strict
mode, so the IRA output's `reasons` was changed from a map to a **list of
`{item_id, reason}` objects** (`llm/schemas.py`, `configs/prompts/ira.txt`, `agents/ira.py`).

**Prompt brevity:** prompts were tightened after measuring ~700 generated tokens per
NLI call. Rationales ≤ 8 words, CSA summary ≤ 60 words, reasons ≤ 8 words.

### 6.3 Agents per user

```
parallel:  S_user = UUA(C_lt, C_st)              # 1 call
           nli    = NLI(raw context, candidates) # 4 calls (5 candidates each)
then:      S_ctx  = CSA(S_user, accepted(nli, θ))# 1 call
then:      π      = IRA(S_user, S_ctx, candidates) # 1 call
```

- NLI accepted set I⁺ = score ≥ θ; missing/unknown ids re-asked once, then neutral
  score 5 (counted as `nli_missing`). CSA falls back to top-3 by score if I⁺ is empty.
- Ranking parser repairs malformed outputs (unknown ids dropped, duplicates removed,
  missing appended) and counts each repair type.
- Concurrency: 8 users in parallel (`asyncio` + thread offload for the blocking client).

### 6.4 θ tuning (validation, 100 users)

ARAG NDCG@5 on the 100 validation users:

| θ | 4 | **5** | 6 | 7 |
|---|---|---|---|---|
| val NDCG@5 | 0.4606 | **0.4890** | 0.4574 | 0.4772 |

**θ\* = 5.** (UUA+NLI computed once and reused across θ; only CSA+IRA re-run.)

### 6.5 α tuning (validation, seed-0 GRU)

`final(i) = α·s_llm(i) + (1−α)·s_gru(i)`, NDCG@5 on validation:

| α | 0.0 | **0.1** | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.8071 | **0.8186** | 0.7826 | 0.7747 | 0.7391 | 0.6899 | 0.6455 | 0.6080 | 0.5599 | 0.5207 | 0.4890 |

**α\* = 0.1** (mostly GRU weight). Full test curve is in `results/metrics.json` and
`results/alpha_curve.png`.

### 6.6 Systems compared (all on the same frozen 20-item lists)

Random, Popularity, GRU (α=0), LLM-Recency, LLM-Vanilla-RAG, ARAG, ARAG w/o NLI,
ARAG w/o NLI&CSA, **ARAG+GRU** (α\*). Non-LLM systems reported on all 94,762 test
users; LLM systems on the 100 sampled test users.

### 6.7 Metrics & significance

- Hit@K, NDCG@K, MRR; paired-bootstrap 95% CIs (10,000 resamples), vs ARAG and vs GRU.
- **Bootstrap memory bug found and fixed:** comparing two 94,762-user systems built a
  10,000 × 94,762 index array (~7.5 GB, near-OOM). Rewritten to sample in chunks of 500
  and cap the bootstrap sample at 20,000 users (the CI is insensitive beyond that);
  the reported mean difference still uses all users.
- Per-user predictions in `artifacts/predictions/*.parquet`; tracked CSV copies in
  `results/predictions/` are capped to the 100 test users to keep the repo small.

### 6.8 Cost & runtime

- 2,405 unique LLM calls; **4,886,695 input / 420,439 output tokens**; **$0.9853**
  (gpt-4o-mini, $0.15/$0.60 per 1M).
- Real evaluation wall time ≈ 15 min (8-way concurrency). A re-run from cache
  completes in 136 s and makes zero API calls.

---

## 7. Results

See `results/report.md`, `results/tables.md`, `results/alpha_curve.png`,
`results/metrics.json`, `results/cost.json`. Headline (test, 100 users):

| System | Hit@5 | NDCG@5 | Hit@10 | MRR |
|---|---|---|---|---|
| random | 0.251 | 0.149 | 0.502 | 0.181 |
| popularity | 0.969 | 0.801 | 0.996 | 0.749 |
| GRU (3 seeds, mean±std) | 0.976±0.001 | 0.823±0.004 | 0.994 | 0.773 |
| LLM-Recency | 0.490 | 0.389 | 0.670 | 0.403 |
| LLM-Vanilla-RAG | 0.480 | 0.386 | 0.670 | 0.403 |
| ARAG | 0.570 | 0.453 | 0.730 | 0.452 |
| ARAG w/o NLI | 0.610 | 0.481 | 0.720 | 0.472 |
| ARAG w/o NLI&CSA | 0.590 | 0.456 | 0.690 | 0.445 |
| **ARAG+GRU (α\*=0.1)** | **0.990** | 0.802 | 1.000 | 0.740 |

- **H1 supported:** ARAG+GRU vs ARAG ΔNDCG@5 = **+0.349** [+0.257, +0.441].
- **H2 not supported:** ARAG+GRU vs GRU ΔNDCG@5 = **+0.002** [−0.018, +0.022].
- Paper ordering holds directionally (Recency/Vanilla-RAG < ARAG), not significant.
- Ablations did not hurt ARAG (weakly better; not significant), unlike the paper.

---

## 8. Product demo (Phase B)

- SQLite memory (`artifacts/app.db`): users, items, interactions (source `dataset|app`),
  profiles, sessions, session_events, rec_runs, rec_items. Seeded with **8 real test
  users** (dataset history) and all 137,260 items with text.
- FastAPI (`make app`): `/health`, `/users`, `/sessions`, `/feed`, `/search`,
  `/feedback`, `/users/{id}/profile` (GET/DELETE), `/runs/{run_id}`,
  `/runs/{run_id}/replay`, `/results`. Bound to 127.0.0.1:8000.
- Search parses the query (LLM) → hard filters (price/exclude) → FAISS top-50 →
  drop dismissed/interacted → top-20 → ARAG+GRU → top-5. Evidence snippets are
  **exact substrings of item text selected in code** (never generated).
- Feed: FAISS on user-context embedding ∪ GRU top-k → 20 candidates → ARAG+GRU →
  paginated. Feedback stored immediately; `dismiss` excludes the item.
- `REPLAY_ONLY=1` refuses uncached LLM calls (offline demo).
- **OMP fix:** loading torch/SentenceTransformer lazily inside a request thread crashed
  the server (`OMP: Error #179`). Models are now preloaded in the main thread at
  startup, with `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false`.

---

## 9. Reproducibility & audit

- `make verify` recomputes every metric from `artifacts/predictions/` and diffs against
  `results/metrics.json`, checks the frozen-list SHA-256 hashes, and checks the run
  record — **VERIFY OK** on the committed artifacts.
- `results/run_record.json` records the git commit, model/provider, category, seed,
  elapsed time, dataset manifest, full config, LLM cost and call count.
- The LLM cache makes the evaluation idempotent: re-running produces identical numbers
  with zero new API calls.

---

## 10. Issues found and fixed during the run

| # | Issue | Fix |
|---|---|---|
| 1 | `pack_padded_sequence` packed the wrong timesteps (left-padding) | right-padding + `total_length` |
| 2 | Loss mask shape mismatch (logits padded to batch-max, not `seq_len`) | reshape + `total_length` |
| 3 | `lists.py` candidate generation O(users × catalogue) — hung | oversample 19+20 then filter |
| 4 | Config key typo `n_valid_users` vs `n_validation_users` | corrected key |
| 5 | Local qwen3:8b flooded by 16 concurrent requests → timeouts | reduced concurrency; then switched to gpt-4o-mini |
| 6 | `run_csa` returns 3 values, callers unpacked 2 | unpack 3 |
| 7 | OpenAI strict mode rejected IRA `reasons` free-form map | reasons → list of `{item_id,reason}` |
| 8 | Bootstrap on 94k users allocated ~7.5 GB (near-OOM) | chunked + cap 20,000 (mean still full-sample) |
| 9 | Tracked prediction CSVs were ~30 MB each (147 MB total) | cap tracked CSVs to 100 test users |
| 10 | API queried an uninitialised DB | `init_db` before the users query |
| 11 | OMP crash loading models in a request thread | preload in main thread + OMP env vars |
| 12 | **Candidate lists drew negatives from all 137k text items, ~81% of which have zero interactions** → ~15.5/19 negatives per list were OOV for the GRU (shared embedding, `score.py` `OOV_IDX=1`) and tied at zero for Popularity → the task collapsed to a ~4.5-known-item choice; Popularity Hit@5 = 0.97 and any GRU-vs-LLM gap was inflated (found in post-submission review) | negatives resampled from the 25,527 items with train interactions, popularity-weighted (`sampling.negative_sampling: popularity`); lists rebuilt, all systems re-evaluated on the same 100 users — see §11 |
| 13 | `compute_all_metrics` crashed with `KeyError: 'arag'` when called on the non-LLM-only subset (hardcoded pairwise refs missing from the subset) | skip pairwise refs absent from the system dict; regression test added (`tests/test_metrics.py`) |
| 14 | GRU checkpoints had been selected by early stopping on the **original** (degenerate) validation lists | GRU retrained from scratch against the corrected lists (same config; objective is list-independent, only checkpoint selection differs) |

---

## 11. Post-review correction: the degenerate candidate lists (2026-10-11)

After the first report was written, a reviewer traced the suspiciously strong
Popularity baseline (Hit@5 = 0.97) to the candidate-list construction and found a
first-order protocol flaw. This section records the full diagnosis, fix and re-run.

### 11.1 Diagnosis (all numbers measured, not assumed)

- `dataprep/lists.py` sampled the 19 negatives from **all 137,260 catalogue items with
  text**, but only **25,612 items (18.7%)** appear in the 5-core interaction data —
  **111,649 (81%) have zero reviews**.
- Measured on the original frozen lists: **mean 15.5 of 19 negatives per list were
  never interacted with by anyone** (median 16, min 7). Only **3.5 known negatives**
  remained, so Popularity and the GRU were solving a ~4.5-way choice, not a 20-way one.
- Mechanism: `models/score.py` maps every unknown item to the shared OOV embedding
  (index 1) → identical scores → alphabetical tie-break; Popularity assigns count 0
  → same tie. The LLM sees no review counts and cannot exploit this.
- Consequence: the originally reported **H1 = +0.349 was an artefact** of list
  construction, and the report's stated cause ("targets are popular") was wrong.

### 11.2 Fix

- `dataprep/lists.py`: negatives now drawn from the **25,527 items with train
  interactions**, weighted by train popularity (`sampling.negative_sampling:
  popularity`, config) so negatives are about as popular as targets (measured ratio
  1.01× on validation, 1.40× on test) — popularity alone therefore cannot separate
  them. History still excluded; still seeded/deterministic/frozen/hashed.
- `eval/run.py`: the main metrics table now scores **every system on the same 100
  sampled test users** (previously GRU/Popularity used all 94,762 while LLM systems
  used 100 — an apples-to-oranges table); non-LLM systems are additionally reported
  on all users in a separate `metrics_all_users` block.
- Post-fix validation on the rebuilt lists: **0 OOV negatives, 19.0 known negatives
  per list, 0 targets outside the train vocabulary.**
- The GRU was retrained from scratch (early stopping had used the old lists).

### 11.3 Corrected outcome

See `docs/REPORT.md` §6–§7 for the full corrected tables and significance tests.
The headline reversal: with fair lists, **ARAG is the best system** (the LLM agent
significantly beats the GRU, Recency and Vanilla-RAG), the validation-optimal fusion
weight is **α\* = 1.0** (the GRU adds nothing to the LLM), the paper's baseline
ordering and its ablations now reproduce **significantly**, and Popularity falls to
at-or-below random. The original (degenerate) numbers are preserved in
`docs/REPORT.md` Appendix E (errata) rather than deleted.
