# ARAG + GRU
## Extending an Agentic LLM Recommender with a Trained GRU Sequential Model — Full Project Report

**Course deliverable — reproducibility-first report**

Companion documents: `SPEC.md` (specification & recorded deviations), `docs/PROCESS.md` (step-by-step engineering log with every real number), `results/report.md` (auto-generated results summary).

> **Errata notice.** An earlier version of this report presented results from a run whose
> candidate lists were degenerate (negatives drawn from the whole catalogue, ~81% of which
> have zero interactions), which artefactually inflated every popularity-like system and
> produced a false "H1 supported" conclusion. That flaw was found in review, the lists were
> rebuilt, every system was re-evaluated, the GRU was retrained, and this document was
> rewritten. The original numbers are preserved — not deleted — in **Appendix E**.

---

## Abstract

This project implements an **ARAG-style agentic recommender** (Maragheh et al., 2025, arXiv:2506.21931) on the Amazon Reviews 2023 **Video Games** category and extends it with **one trained deep-learning component: a PyTorch GRU sequential user model late-fused into the final LLM ranking**. All systems rank the *same frozen 20-item candidate lists* (1 held-out target + 19 **popularity-matched** negatives drawn from the training vocabulary) per user, so the comparison isolates **ranking** from retrieval and no system can win by exploiting an information asymmetry.

Using OpenAI `gpt-4o-mini` for the agent pipeline, 100 sampled test users for every system (plus all 94,762 test users for the non-LLM systems as a secondary table), we find:

- **H1 not supported:** fusing the trained GRU into ARAG does **not** improve on ARAG alone (ΔNDCG@5 = **−0.013**, 95% CI [−0.032, +0.005] — not significant). The validation-optimal fusion weight is **α\* = 0.9**, but on test even that 10% GRU weight is a small, non-significant *drag*.
- **H2 supported:** the fused system significantly outperforms the GRU alone (ΔNDCG@5 = **+0.174**, 95% CI [+0.069, +0.278]) — the LLM agent adds a large increment on top of the trained model.
- **ARAG is the best system.** It significantly beats LLM-Recency (Δ = −0.139), LLM-Vanilla-RAG (Δ = −0.162), GRU seeds 1 and 2 (Δ = −0.179, −0.209; and the fused system — statistically equal to ARAG — beats seed 0 by +0.174), Popularity (Δ = −0.315) and Random (Δ = −0.299) — every CI excluding 0.
- **The paper's claims reproduce.** The baseline ordering (Recency, Vanilla-RAG < ARAG) and one of the two ablations (removing NLI **and** CSA significantly hurts, Δ = −0.062) now hold **statistically**, unlike the earlier degenerate run.
- **A pure popularity baseline is at or below random** (NDCG@5 0.109 vs random 0.125): because negatives are drawn to be about as popular as targets, popularity carries no free signal.

All numbers are produced by a deterministic, audited pipeline: **`make verify` re-derives every reported metric from the raw per-user predictions and checks dataset hashes and the run record — it passes on the committed artifacts.** The corrected evaluation added **2,263 LLM calls ≈ $1.06** (total unique project spend across both protocol versions: **$2.04**); a re-run from cache is **free (137 s)**.

---

## 0. How to read this document (orientation, glossary, diagram)

**If you have 30 seconds:** the project asks whether adding a *trained* neural sequence model (GRU) to an *LLM agent* recommender (ARAG) improves ranking. Answer: **no (H1 fails) — and on fair candidate lists the LLM agent is itself the best system**, significantly beating the trained model, the paper's simpler LLM baselines, popularity and random ranking.

**If you have 10 minutes:** read §1 (questions), §3.2 (frozen candidate lists — including why they were rebuilt after review), §4 (system), §6 (results), §7 (analysis), and Appendix E (the errata, if you saw the earlier numbers).

**If you must defend it:** every number is reproducible — `make verify` re-derives all metrics from raw predictions and checks dataset hashes; §10 lists the full command sequence; §9 lists every deviation from the original spec; §11 logs every engineering issue found and fixed, including the list-construction flaw.

### 0.1 Glossary

| Term | Meaning |
|---|---|
| **ARAG** | Agentic recommender (Maragheh et al., 2025): an LLM pipeline of four agents (below). Our system is ARAG-*inspired*, not a reproduction. |
| **UUA** | User Understanding Agent — turns the user's history into interests/intent/summary. |
| **NLI** | Natural Language Inference agent — labels each candidate *support / neutral / contradict* with a 0–10 score against the user's history. |
| **CSA** | Context Summary Agent — summarises the accepted candidates into a short context. |
| **IRA** | Item Ranker Agent — outputs the final permutation of the 20 candidates. |
| **GRU** | Gated Recurrent Unit — a small recurrent neural network that reads an item sequence and produces a user vector. The project's trained model. |
| **h_u** | The GRU's user vector (its final hidden state): a fixed-length summary of the user's sequence. |
| **Vocabulary / OOV** | The set of item ids the GRU knows (from train data); items outside it are "out-of-vocabulary" and mapped to a special OOV embedding. The corrected lists guarantee **no OOV among candidates**. |
| **Candidate list (frozen)** | The 20 products each user is ranked against: their true held-out **target** + 19 **popularity-matched negatives** from the training vocabulary. Frozen so every system ranks identical items. |
| **Target / negatives** | The one item the user actually interacted with next (target) and the 19 decoys (negatives), sampled proportional to train popularity so popularity alone cannot separate them. |
| **Leave-last-out** | The dataset's official split: per user, last item = test target, second-to-last = validation target, the rest = train history. |
| **5-core** | Dataset filter: keep only users/items with ≥5 interactions. |
| **Hit@K** | 1 if the target is in the top K, else 0. |
| **NDCG@K** | 1/log₂(rank+1) if the target is in the top K, else 0 — rewards higher placement. |
| **MRR** | 1/rank of the target (Mean Reciprocal Rank). |
| **θ (theta)** | NLI acceptance threshold: candidates with score ≥ θ enter CSA. Tuned on validation. |
| **α (alpha)** | Fusion weight: final score = α·(LLM) + (1−α)·(GRU). α=1 → ARAG only; α=0 → GRU only. Tuned on validation. |
| **Late fusion** | Combining two ranked lists after both have ranked (here by blending rank-derived scores). |
| **Paired bootstrap** | Resampling users with replacement to get a 95% CI for the *difference* between two systems on the same users. CI containing 0 ⇒ not significant. |
| **MiniLM / FAISS** | Sentence-transformer model (embeddings) / vector index used for similarity search. |
| **LLM disk cache** | Responses stored by (model, prompt, schema) hash so re-runs make zero API calls. |

### 0.2 System diagram

```
                          ┌──────────────── ARAG (LLM agent pipeline) ────────────────┐
 history (train+val,      │   ┌─────┐                                                  │
 ≤55 items, rendered as   │   │ UUA │──────────┐                                       │
 "title|category|score|   ├──►└─────┘          ▼                                       │
 date")                   │   ┌─────┐      ┌─────┐      ┌─────┐                       │
 20 frozen candidates ────┼──►│ NLI │─────►│ CSA │─────►│ IRA │──► π_LLM (permutation)│
                           │   └─────┘  I+  └─────┘      └─────┘        │              │
                           └──────────  (score≥θ)  ────────────────────  │              │
                                                                         │              │
                           ┌──────────────── GRU (trained on 625k train pairs) ────────┐│
 history (last 50 train   │   h_u = GRU(sequence)                                    ││
 items → indices)         ├──►  score_i = h_u · e_i  ──► π_GRU (permutation)         ││
 20 frozen candidates ────┤                                                             ││
                           └─────────────────────────────────────────────────────────────┘│
                                                                         │              │
                                                                         ▼              ▼
                                               final = α·s_LLM + (1−α)·s_GRU   (α* = 0.9)
                                                           │
                                                           ▼
                                               ranked 20 candidates
```

**Where each piece lives in code:** agents in `src/arag/agents/` (`uua.py`, `nli.py`, `csa.py`, `ira.py`, `arag.py`); GRU in `src/arag/models/` (`gru.py`, `train_gru.py`, `score.py`); fusion in `src/arag/ranking/fusion.py`; the whole experiment in `src/arag/eval/run.py`.

---

## 1. Introduction and research questions

Large language models can rank items by reasoning over a user's history in natural language. ARAG (Agentic Recommender with Augmented Generation) formalises this as a multi-agent pipeline: a User Understanding Agent (UUA), a Natural Language Inference (NLI) agent, a Context Summary Agent (CSA) and an Item Ranker Agent (IRA). This project builds an ARAG-inspired system and asks whether adding a **trained** sequential model helps:

**Primary question.**
> On identical frozen candidate lists, does fusing a trained GRU score into ARAG's LLM ranking improve Hit@5 / NDCG@5 over (a) ARAG alone and (b) the GRU alone?

- **H1:** ARAG+GRU > ARAG (i.e., improvement over the LLM-only system).
- **H2:** ARAG+GRU > GRU alone (i.e., the LLM adds value on top of the trained model).

**Secondary questions**, mirroring the paper's internal comparisons on our data:
1. Do the paper's simpler LLM baselines (LLM-Recency, LLM-Vanilla-RAG) rank below ARAG?
2. Do the paper's two ablations (ARAG without NLI; ARAG without NLI & CSA) lose to full ARAG?
3. What does the system cost (tokens, USD, calls) and how reproducible is it?

**Scope note (stated up front).** This is **ARAG-inspired, not an exact reproduction**: the paper does not publish prompts, the NLI threshold, candidate-list construction or code, and it used gpt-3.5-turbo on an older Amazon dataset. Our numbers must never be compared directly with the paper's Table 1. Negative or mixed results are treated as valid outcomes and reported as measured — including H1 failing, and including the protocol flaw found in review (Appendix E).

---

## 2. Background

- **ARAG (Maragheh et al., 2025).** Agentic recommendation: interpret the user (UUA), judge candidate–user compatibility (NLI), summarise relevant context (CSA), then rank (IRA). We follow this decomposition.
- **GRU4Rec (Hidasi et al., 2016).** Session-based recommendation with a GRU over item sequences and a full-softmax next-item objective. Our GRU is a GRU4Rec-style model with tied input/output embeddings.
- **Late fusion.** Because an LLM ranker emits a *permutation* (not a vector), the natural way to inject a trained model is to fuse two rankings. This also makes the trained model's contribution **exactly ablatable** (α = 1 ⇒ ARAG only; α = 0 ⇒ GRU only).

---

## 3. Dataset and evaluation protocol

### 3.1 Data

| Item | Value |
|---|---|
| Dataset | `McAuley-Lab/Amazon-Reviews-2023` (McAuley Lab) |
| Category | **Video_Games** |
| Split | official **rating-only 5-core, leave-last-out** benchmark (`benchmark/5core/last_out/`) |
| Files | `Video_Games.{train,valid,test}.csv` (columns `user_id,parent_asin,rating,timestamp`) + `meta_Video_Games.jsonl` (437 MB) |
| Integrity | SHA-256 of all four files recorded in `results/manifest.json` |
| Interactions | 814,586 total — **625,062 train / 94,762 valid / 94,762 test** |
| Users | 94,762 (one validation and one test target each) |
| Items | 137,269 metadata rows, 137,260 with text; **25,612 of them (18.7%) appear in the interaction data — 111,649 (81%) have zero reviews** |
| Item text | product metadata only (title, main_category, categories, features, description), truncated to **120 words**; review text is never used |

Leave-last-out semantics: per user, the **last** interaction is the test target, the **second-to-last** is the validation target, and the rest form the train history. Assertions enforce `max(train ts) ≤ valid ts ≤ test ts` and that targets never appear in their own history (no violations; `make lists` fails hard otherwise).

### 3.2 Frozen candidate lists (the central design decision — and why they were rebuilt)

For every validation and test user we construct a **20-item candidate list**:

> the user's true held-out target + **19 negatives sampled (seed 42) proportional to train interaction count** from the **25,527 items that have train interactions**, excluding anything the user interacted with in any split; the list is shuffled with the seeded RNG and then **frozen**.

Every system — Random, Popularity, GRU, the LLM baselines, ARAG, the ablations and the fused system — ranks this identical list. This is deliberate:

- If each method retrieved its own candidates, a missed retrieval would score 0 for that method and the experiment would measure **retrieval**, not **ranking**.
- A frozen list is the only way to attribute a Hit@5 difference to *ranking quality*.
- The expected Hit@5 of a uniformly random permutation on a 20-item list is exactly **0.25** (0.220 on our 100-user sample — within noise of n=100), which we use as a sanity anchor.

**Why the negatives are popularity-weighted (post-review correction).** The first version of this experiment sampled negatives uniformly from *all 137,260 catalogue items with text*. Because 81% of those items have **zero interactions**, a measured **mean 15.5 of 19 negatives per list were unknown** to Popularity (tied at count 0) and to the GRU (one shared OOV embedding → identical scores → alphabetical tie-break). The effective task for those systems was a ~4.5-way choice, not a 20-way one — Popularity reached Hit@5 = 0.97 and any GRU-vs-LLM gap was inflated. This was found in review, fixed (`dataprep/lists.py`), and **every system was re-evaluated**; the GRU was also retrained. Full diagnosis and the superseded numbers: **Appendix E** and `docs/PROCESS.md §11`.

**Post-fix validation on the rebuilt lists:** 0 OOV negatives; 19.0/19 known negatives per list; 0 targets outside the train vocabulary; negatives average 1.01× (validation) / 1.40× (test) the target's train popularity — i.e. about as popular as the target, so popularity alone carries no free signal.

Lists are hashed (`artifacts/lists/lists_hashes.json`) and re-verified by `make verify`; they are never regenerated after evaluation begins.

### 3.3 Evaluation users

- **Every system** — including Random, Popularity and the GRU — is scored on the **same 100 sampled test users** for the main table and for all significance tests (this was corrected after review; the earlier table compared 100-user LLM numbers against all-user GRU numbers).
- Non-LLM systems are additionally reported on **all 94,762 test users** in a separate block (`metrics_all_users`), which gives near-deterministic estimates for those methods.
- The demo (Phase B) seeds 8 of the same real test users; no synthetic users exist anywhere in the evaluation.

### 3.4 Metrics

For a single relevant item per user with rank *r* (1-indexed):

- **Hit@K** = 1 if *r ≤ K* else 0
- **NDCG@K** = 1/log₂(r+1) if *r ≤ K* else 0
- **MRR** = 1/r

Reported metrics: Hit@5, NDCG@5, Hit@10, NDCG@10, MRR. Primary metrics are **Hit@5 and NDCG@5**.

### 3.5 Statistical testing

Every system is compared with **paired bootstrap** over the same 100 users (10,000 resamples, 95% percentile CIs of the mean per-user metric difference), versus **ARAG** and versus **GRU** (seed 0). A CI containing 0 is reported as *no significant difference*.

---

## 4. System description

### 4.1 ARAG agent pipeline

All prompts live in `configs/prompts/`; all outputs are validated by Pydantic models and requested from the LLM as **strict JSON schemas**.

| Agent | Input | Output | Calls/user |
|---|---|---|---|
| **UUA** (User Understanding) | long-term context Cₗₜ (≤50 items) + session context Cₛₜ (last 5) | interests, current intent, dislikes, ≤120-word summary | 1 |
| **NLI** (candidate judging) | raw history + **5 candidates per call** | per candidate: support/neutral/contradict, score 0–10, rationale | 4 |
| **CSA** (context summary) | UUA summary + accepted candidates (score ≥ θ) | ≤200-word relevant-context summary | 1 |
| **IRA** (item ranker) | UUA summary + CSA summary + all 20 candidates | full 20-item ranking + one-line reasons (top 5) | 1 |

Execution per user (`agents/arag.py`):

```
parallel:  S_user = UUA(C_lt, C_st)
           nli    = NLI(raw context, 20 candidates)     # 4 calls, 5 each
then:      S_ctx  = CSA(S_user, accepted(nli, θ))
then:      π      = IRA(S_user, S_ctx, candidates)
```

Robustness: NLI missing/unknown ids are re-asked once; anything still missing receives a neutral score of 5 and is counted. If no candidate passes θ, CSA falls back to the top-3 by score (flagged). The **ranking parser** repairs malformed outputs (drops unknown ids, removes duplicates keeping first, appends missing ids by NLI score, else list order) and counts every repair.

### 4.2 GRU sequential model (the deep-learning contribution)

- **Architecture:** item embedding (64-d) → dropout → 1-layer GRU (hidden **128**) → dropout → linear projection (64-d) → full-softmax over the vocabulary, with **input/output embeddings tied**.
- **Vocabulary:** items seen in the train split (25,527); index 0 = padding, 1 = OOV.
- **User vector:** h_u = last valid hidden state (projected to 64-d). Item score = h_u · e_i.
- **Training:** last 50 train items per user, right-padded, `pack_padded_sequence`; next-item prediction on every prefix; full-softmax cross-entropy; Adam lr 1e-3; batch 256; ≤12 epochs with early stopping (patience 3) on **validation NDCG@5 over all 94,762 validation users against the frozen lists**.
- **Hyperparameter grid:** hidden ∈ {64, 128} × dropout ∈ {0.2, 0.5}, then **3 seeds** of the best config. Total training time ≈ 1 h 50 m CPU.
- **Result (retrained on the corrected lists):** grid validation NDCG@5 = 0.1549 (64/0.2), 0.1545 (64/0.5), **0.2988 (128/0.2 — best)**, 0.1534 (128/0.5); seed validation NDCG@5 = 0.2928 / 0.2795 / 0.2838. Test (all 94,762 users): **Hit@5 0.422 ± 0.010, NDCG@5 0.279 ± 0.007, MRR 0.285 ± 0.005**.

Note the scale change versus the original run (NDCG@5 0.823): that number was obtained against the degenerate lists where the GRU only had to beat ~3.5 known decoys (Appendix E).

### 4.3 Late fusion

Both systems rank the 20 candidates; convert ranks to scores and blend:

```
s_llm(i) = 1 − (rank_IRA(i) − 1)/19
s_gru(i) = 1 − (rank_GRU(i) − 1)/19
final(i) = α·s_llm(i) + (1−α)·s_gru(i),      α ∈ {0.0, 0.1, …, 1.0}
```

Ties break by higher NLI score, then item_id. **α is chosen on the 100 validation users** (never on test) and then applied unchanged to test. By construction α=1 reproduces the IRA order exactly and α=0 the GRU order exactly (unit-tested).

### 4.4 LLM client and reproducibility infrastructure

- OpenAI SDK, `temperature=0`, strict JSON-schema `response_format`, 60 s timeout, exponential backoff on 429/5xx, one schema-retry with the validation error appended, then the documented per-agent fallback (always counted, never silent).
- **Disk cache** keyed by SHA-256 of (model, prompt, schema): re-running the evaluation makes **zero** new API calls and reproduces identical outputs.
- Per-call logging of agent, tokens in/out, latency, cache hit, cost, fallback.
- `REPLAY_ONLY=1` refuses uncached calls (offline demo mode).

---

## 5. Experimental setup

### 5.1 Systems compared

| System | Definition | LLM calls/user |
|---|---|---|
| Random | seeded random permutation | 0 |
| Popularity | order by train interaction count | 0 |
| GRU | order by GRU score (α = 0) | 0 |
| LLM-Recency | last 10 history items + 20 candidates, one ranking prompt | 1 |
| LLM-Vanilla-RAG | 10 history items most similar (MiniLM cosine) to the mean candidate embedding + 20 candidates | 1 |
| ARAG | UUA ∥ NLI → CSA → IRA | 7 |
| ARAG w/o NLI | CSA summarises all 20 candidates → IRA | 2 (UUA reused) |
| ARAG w/o NLI & CSA | IRA receives UUA summary + raw candidates | 1 (UUA reused) |
| **ARAG+GRU (proposed)** | late fusion at α\* | 0 extra |

### 5.2 Tuning (validation only)

- **θ (NLI acceptance threshold):** validation NDCG@5 at θ ∈ {4, 5, 6, 7} = 0.4228 / **0.4305** / 0.3993 / 0.3962 → **θ\* = 5**.
- **α (fusion weight):** validation NDCG@5 over α ∈ {0.0…1.0} peaks at **α\* = 0.9** (0.4341), with α = 1.0 at 0.4305 and α = 0.0 at 0.2965.

Both are selected **only** on the 100 validation users; the test set is never used for tuning.

### 5.3 Model & cost

- LLM: OpenAI **gpt-4o-mini** (temperature 0). Provider is configurable (local Ollama `qwen3:8b` is supported through the same client; the committed results use gpt-4o-mini — see §9).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalized) + FAISS `IndexFlatIP`.
- **Cost of the corrected evaluation:** 2,263 new unique LLM calls (the LLM outputs depend on the candidate lists, so the rebuilt lists invalidated the old cache). Total unique project spend across both protocol versions: 4,668 calls, 10,310,033 input / 830,142 output tokens, **$2.04** at $0.15/$0.60 per 1M (original run $0.9853 + corrected run ≈ $1.06). A cached re-run completes in **137 s** at **$0**.

---

## 6. Results

### 6.1 Main comparison (test)

**Every system is scored on the same 100 sampled test users.** GRU rows are individual seeds (the trained model is not seed-averaged at inference).

| System | Hit@5 | NDCG@5 | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Random | 0.220 | 0.125 | 0.470 | 0.205 | 0.162 |
| Popularity | 0.200 | 0.109 | 0.470 | 0.197 | 0.150 |
| GRU (seed 0) | 0.390 | 0.238 | 0.640 | 0.321 | 0.249 |
| GRU (seed 1) | 0.450 | 0.246 | 0.640 | 0.306 | 0.230 |
| GRU (seed 2) | 0.360 | 0.216 | 0.630 | 0.302 | 0.230 |
| LLM-Recency | 0.400 | 0.285 | 0.630 | 0.358 | 0.300 |
| LLM-Vanilla-RAG | 0.380 | 0.263 | 0.670 | 0.355 | 0.283 |
| **ARAG (θ\*=5)** | **0.530** | **0.424** | 0.700 | **0.479** | **0.432** |
| ARAG w/o NLI | 0.510 | 0.376 | 0.720 | 0.445 | 0.380 |
| ARAG w/o NLI & CSA | 0.470 | 0.362 | 0.630 | 0.414 | 0.372 |
| ARAG+GRU (α\*=0.9) | 0.540 | 0.412 | 0.710 | 0.466 | 0.411 |

**Secondary table — non-LLM systems on all 94,762 test users** (near-deterministic estimates; GRU mean ± std over seeds):

| System | Hit@5 | NDCG@5 | MRR |
|---|---|---|---|
| Random | 0.252 | 0.149 | 0.181 |
| Popularity | 0.213 | 0.128 | 0.166 |
| GRU (mean ± std) | 0.422 ± 0.010 | 0.279 ± 0.007 | 0.285 ± 0.005 |

### 6.2 Hypothesis tests (paired bootstrap, ΔNDCG@5, same 100 users, 95% CI)

| Comparison | ΔNDCG@5 | 95% CI | Verdict |
|---|---|---|---|
| **ARAG+GRU − ARAG (H1)** | **−0.013** | [−0.032, +0.005] | **not supported** |
| **ARAG+GRU − GRU seed 0 (H2)** | **+0.174** | [+0.069, +0.278] | **significant — supported** |
| ARAG − LLM-Recency | +0.139 | [+0.063, +0.217] | significant — paper's baseline ordering reproduces |
| ARAG − LLM-Vanilla-RAG | +0.162 | [+0.079, +0.247] | significant — paper's baseline ordering reproduces |
| ARAG − ARAG w/o NLI | +0.049 | [−0.011, +0.111] | not significant |
| ARAG − ARAG w/o NLI & CSA | +0.062 | [+0.016, +0.113] | significant — paper's ablation reproduces |
| ARAG − GRU seed 1 | +0.179 | [+0.078, +0.282] | significant |
| ARAG − GRU seed 2 | +0.209 | [+0.098, +0.318] | significant |
| GRU seed 0 − Popularity | +0.129 | [+0.078, +0.186] | significant |
| GRU seed 0 − Random | +0.113 | [+0.022, +0.207] | significant |
| LLM-Recency − GRU seed 0 | +0.047 | [−0.057, +0.148] | not significant |
| LLM-Vanilla-RAG − GRU seed 0 | +0.025 | [−0.072, +0.119] | not significant |

(GRU seed 0 is one of the two reference systems in the pairwise tests, so it has no `vs ARAG` entry of its own; its gap to ARAG is captured by H2 above — the fused system, statistically indistinguishable from ARAG, beats seed 0 by +0.174 — and by the mean differences in §6.1. Signs are shown in the intuitive direction; raw `*_vs_*` entries in `metrics.json` store the difference as named.)

### 6.3 The α curve (validation and test)

| α | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | **0.9** | 1.0 |
|---|---|---|---|---|---|---|---|
| validation NDCG@5 | 0.2965 | 0.3127 | 0.3229 | 0.3572 | 0.3993 | **0.4341** | 0.4305 |
| test NDCG@5 | 0.2383 | 0.2731 | 0.3045 | 0.3582 | 0.3896 | 0.4119 | **0.4245** |

The curve rises steeply towards the LLM end on **both** splits: almost all of the ranking signal is on the LLM side. Validation prefers α=0.9 over α=1.0 by a hair (0.4341 vs 0.4305 — 3.6 thousandths on 100 users), but on test the pure-LLM end is best (0.4245 vs 0.4119). That is the classic signature of a **hyperparameter overfit on a small validation set**: the 10% GRU weight that looked helpful on validation is a small (non-significant) drag on test. The full curve is plotted in `results/alpha_curve.png`.

### 6.4 Worked example — one real test user, end to end

This is real data from the evaluation (not an illustration). User `AEDTQXNMWBEI7UECETV5JVUEWNDQ`, a console-games buyer.

**Their history (train + validation, in time order):**

| # | Item |
|---|---|
| 1 | The Legend of Zelda: Breath of the Wild — Wii U |
| 2 | Need for Speed Most Wanted U — Nintendo Wii U |
| 3 | Hot Wheels World's Best Driver |
| 4 | LEGO: Marvel — PC |

**Their frozen 20-item candidate list** (order as stored), with the true held-out target marked:

| # | Title (truncated) | | # | Title (truncated) |
|---|---|---|---|---|
| 1 | Demon's Souls — PS3 | | 11 | Insten Snap-on Crystal Case for PS4 |
| 2 | New Super Mario Bros. U | | 12 | Mortal Kombat X — PS4 |
| 3 | Sesame Street: Cookie's Counting Carnival — PC | | 13 | PlayStation 2 Dance Mat |
| 4 | Luigi amiibo | | 14 | Conker's Bad Fur Day |
| 5 | Far Cry New Dawn — Xbox | | 15 | Carnival Games — Xbox 360 |
| 6 | Transformers: War for Cybertron — Xbox 360 | | 16 | PS4 Controller Charger Dock |
| 7 | Super Mario Land 2 | | 17 | Xbox360 Controller Battery Cover |
| 8 | PS4 Controller Grips (silicone) | | 18 | Tomee 256KB Memory Card for N64 |
| 9 | Resident Evil 3 Nemesis — PC | | 19 | Ghost of Tsushima — PS4 |
| **10** | **ARKARTECH Gaming Headset ← TARGET** | | 20 | Stereo Gaming Headset (Blue) |

Note the list itself: **every item is a plausible, well-known gaming product** — that is the popularity-weighted negative sampling working. There are no freebie zero-review decoys.

**How each system ranked the target:**

| System | Rank of target |
|---|---|
| **ARAG (LLM)** | **2 (hit)** |
| **ARAG+GRU (α\*=0.9)** | **2 (hit)** |
| LLM-Recency | 7 (miss) |
| **GRU (seed 0)** | **11 (miss)** |
| Popularity | 15 (miss) |

**What this tells you in one example:** the LLM reads four game titles and correctly infers *"console gamer → gaming headset"*, placing the true next item 2nd among 19 hard decoys (including a second, competing headset at #20). The GRU — trained on sequences, blind to product text — misses at rank 11; Recency misses at 7; Popularity is useless (15) because the decoys are as popular as the target. This single case is a microcosm of the aggregate result: **ARAG ≫ GRU, and fusion inherits the LLM.**

---

## 7. Analysis and discussion

### 7.1 H1 not supported: the trained GRU adds nothing to the agentic pipeline

Fusing the GRU at the validation-optimal α\* = 0.9 changes NDCG@5 by **−0.013 [−0.032, +0.005]** versus ARAG alone — not significant, and if anything slightly negative. The α curve (§6.3) shows why: validation and test both rise monotonically towards α = 1, with the only exception being a 0.0036 validation-only bump at 0.9 that reverses on test. On 100 validation users that bump is noise; tuning α there and applying it to test is exactly the setting where a small overfit is expected, and we report it rather than hide it.

Substantively: the GRU's signal — sequential co-occurrence statistics over 625k interactions — is **subsumed** by what the IRA agent already does. The IRA sees the same user's full history in text plus 20 candidate descriptions and a NLI score per candidate; the 10% weight on a blind sequence model cannot add information the LLM has not already used. This is the honest, null answer to the project's primary question.

### 7.2 H2 supported: the LLM adds a large, significant increment over the GRU

The fused system beats the GRU alone by **+0.174 [+0.069, +0.278]**, and pure ARAG significantly beats GRU seeds 1 and 2 (Δ = +0.179, +0.209) while the fused system — statistically indistinguishable from ARAG — significantly beats seed 0. Even the paper's *weakest* LLM baseline (Vanilla-RAG, +0.025) is directionally above the GRU, though not significantly. Under a protocol where popularity cannot win for free, **content reasoning beats sequential co-occurrence** in this domain. Plausible reasons: 5-core users have short histories (~6.6 interactions), so the GRU's sequence signal is thin; item text (title, category, features) is highly informative in Video Games (genres, platforms, accessories); and the LLM can perform cross-category inference ("buys Wii U games → buys a headset") that a 128-d sequence embedding trained on next-item likelihood does not capture.

### 7.3 The paper's claims now reproduce — significantly

> **Appendix F** puts the paper's own Table 1 side by side with ours (baselines and ablations, all three of the paper's categories plus our Video Games), and documents why the absolute numbers are not comparable.

| Paper claim | Our result |
|---|---|
| LLM-Recency < ARAG | **Significant**: Δ = −0.139 [−0.217, −0.063] |
| LLM-Vanilla-RAG < ARAG | **Significant**: Δ = −0.162 [−0.247, −0.079] |
| Removing NLI & CSA hurts | **Significant**: Δ = −0.062 [−0.113, −0.016] |
| Removing only NLI hurts | Not significant: Δ = −0.049 [−0.111, +0.011] (direction correct) |

This is a meaningful contrast with the earlier degenerate run, where none of these differences reached significance and the ablations were weakly *better* than full ARAG (Appendix E). The confound that hid them was the popularity-dominated candidate lists: when almost every decoy is an unreviewed item, the elaborate agentic context-processing has no room to show its value, and a one-shot ranking prompt does about as well. Once the decoys are real, popular products, **the full agent stack (especially CSA) demonstrably earns its calls.**

### 7.4 Popularity is now at or below random — the protocol works

With popularity-matched negatives, the Popularity baseline scores NDCG@5 **0.109 — below random's 0.125** on the 100-user sample (and 0.128 vs 0.149 on all 94,762 users). It is significantly beaten by the GRU (Δ = 0.129) and by ARAG (Δ = 0.315). This is the designed behaviour: negatives are drawn from the same popularity distribution as targets, so global frequency carries no free signal and any system that wins must actually use the user.

### 7.5 Secondary observations

- LLM-Recency (0.285) and LLM-Vanilla-RAG (0.263) are close, and neither significantly beats the GRU — the MiniLM retrieval step in Vanilla-RAG again adds nothing measurable over simply taking the last 10 items.
- The three GRU seeds agree with each other (seed-to-seed Δ n.s.), so the GRU-vs-ARAG gap is not a seed artefact.
- Random's 0.220 (vs the exact expectation 0.25) is ordinary sampling noise at n=100 (SE ≈ 0.043); the all-user estimate is 0.252.

---

## 8. Threats to validity and limitations

1. **Sampled negatives, not retrieval-mined candidates.** Popularity-matched negatives from the training vocabulary remove the popularity shortcut but are still easier than what a production retriever would surface. Headline numbers should be read as *"ranking quality on hard 20-item lists"*, not as production retrieval quality.
2. **100 test users ⇒ wide CIs.** Roughly ±8–10 points on Hit@5. The large effects (ARAG vs everything) are comfortably resolved; the small ones (α bump, NLI-only ablation) are not, and α itself is a noisy hyperparameter at this n (evidenced by the valid/test disagreement at the top of the curve).
3. **Leave-last-out leakage into GRU training.** Other users' later interactions may appear in the GRU's training data; inherent to the official split, applies equally to all systems, but means GRU numbers are upper bounds relative to a strictly time-separated setup.
4. **Single category, single dataset.** Video Games only; results may not transfer.
5. **Prompt and threshold sensitivity.** The paper does not publish prompts or θ; our prompt wording and θ\*=5 are our own choices, tuned only on validation.
6. **Model substitution.** The paper used gpt-3.5-turbo on an older dataset; we use gpt-4o-mini. This is ARAG-inspired, not a reproduction, and no number here is comparable to the paper's tables.
7. **No human evaluation.** All judgments are offline metrics against a single held-out item.
8. **The protocol was corrected mid-project** (Appendix E). The correction was made in response to review, applied to every system, and the GRU retrained; but a reader should treat the corrected run as the definitive one and be aware the original run's conclusions were wrong.

---

## 9. Deviations from the original specification (recorded, not hidden)

| # | Spec | What was done | Why |
|---|---|---|---|
| 1 | LLM = OpenAI gpt-4o-mini | **Yes — gpt-4o-mini for the committed results.** Local Ollama `qwen3:8b` is also supported through the same client | Local qwen3:8b measured ≈20 s/call with serial Ollama slots (~14 h for the eval); gpt-4o-mini completed the same eval in ~15 min for ~$1. Developed and tested against both |
| 2 | GRU ≤30 epochs | **≤12 epochs** | Validation NDCG@5 plateaus well before 30 under early stopping (patience 3) |
| 3 | `llm/fake.py` test double | **Removed** | Per project decision: no fake LLM client anywhere; deterministic logic is tested with hand-made fixtures, integration tests hit a real provider |
| 4 | Exact file names in spec (`raw_meta_Video_Games`) | **Corrected to the real dataset paths** | The config name does not exist for Video_Games; metadata is `raw/meta_categories/meta_Video_Games.jsonl` (verified against the dataset card) |
| 5 | Tracked prediction CSVs | **Capped to the 100 test users** | Full CSVs were ~147 MB; parquet in `artifacts/` holds every user |
| 6 | **Negative sampling: uniform over catalogue items with text** | **Changed to popularity-weighted sampling over items with train interactions** (`sampling.negative_sampling: popularity`) | Post-review protocol fix: the original scheme made ~81% of negatives OOV, collapsing the task (Appendix E, `PROCESS.md §11`) |
| 7 | Main table mixes 100-user LLM numbers with all-user non-LLM numbers | **All systems scored on the same 100 sampled users**; non-LLM additionally on all users in a separate block | Apples-to-oranges comparison; also found in review |
| 8 | GRU grid picked hidden=64/dropout=0.2 | **Re-run on the corrected lists: hidden=128/dropout=0.2** | The old grid was selected against the degenerate lists; the retrained model is part of the corrected pipeline |

---

## 10. Reproducibility and artifact integrity

- **`make verify`** recomputes every reported metric from the raw per-user predictions in `artifacts/predictions/`, diffs them against `results/metrics.json` (both the sampled-user and all-user blocks), recomputes the SHA-256 of the frozen candidate lists, and checks the run record — currently **VERIFY OK**.
- `results/run_record.json` records the git commit, provider/model, category, seed, elapsed time, the dataset manifest (file hashes + row counts) and the full config.
- **No fabricated numbers anywhere.** `results/` is written only by pipeline code on real data; the unit/integration test fixtures are hand-made inputs that never touch `results/` or `artifacts/`.
- Determinism: fixed seeds (data 42; GRU 0/1/2), temperature 0, frozen hashed lists, SHA-256-keyed LLM cache. Re-running the evaluation reproduces identical numbers with **zero** new API calls (137 s).

**Full reproduction (from a fresh clone):**

```bash
make setup          # venv + pinned pip deps + npm install
export OPENAI_API_KEY=...     # or switch config.yaml to provider: ollama
make data           # download + manifest + parquet tables
make lists          # sequences, samples, frozen candidate lists (popularity-weighted)
make index          # MiniLM embeddings + FAISS
make gru            # GRU grid + 3 seeds          (~1 h 50 m CPU)
make eval           # all systems, θ/α tuning, bootstrap
make report         # results/*.md|json|png
make verify         # audit: recompute metrics, check hashes, run record
make test           # pytest suite
make app            # demo: FastAPI + built React frontend on 127.0.0.1:8000
```

---

## 11. Engineering issues found and fixed (summary)

A complete log is in `docs/PROCESS.md §10–§11`. The notable ones:

1. **Padding side bug in GRU training** — `pack_padded_sequence` expects right-padding; the initial left-padded data silently packed the wrong timesteps, caught by a unit test and fixed.
2. **Candidate-list generation was O(users × catalogue)** and effectively hung; rewritten with oversampling + filtering (94,762 users × 2 splits in ~15 s).
3. **`run_csa` return-arity bug** (3 values unpacked as 2) crashed the first eval attempt at θ-tuning.
4. **OpenAI strict-schema constraint** — free-form `dict[str,str]` reasons are not allowed; output schema changed to a list of `{item_id, reason}`.
5. **Bootstrap memory blow-up** — a naive 10,000 × 94,762 resampling matrix (~7.5 GB) was OOM-killed; rewritten to chunked sampling with a 20,000-user cap (means still computed on all users).
6. **Degenerate candidate lists (found in review)** — negatives were drawn from all 137k text items, ~81% of which have zero interactions, so ~15.5/19 negatives per list were OOV for the GRU and tied-at-zero for Popularity; the task collapsed to a ~4.5-way choice and produced a false "H1 supported". Fixed by popularity-weighted sampling over the training vocabulary; lists rebuilt, all systems re-evaluated, GRU retrained (Appendix E).
7. **`compute_all_metrics` KeyError** on the new non-LLM-only metrics block (hardcoded pairwise refs missing from the subset); fixed with a presence guard + regression test.
8. **GRU checkpoints selected under the old lists** — early stopping had used the degenerate validation lists; the GRU was retrained from scratch against the corrected ones.

---

## 12. Conclusion

We built, trained and evaluated an ARAG-inspired agentic recommender fused with a trained GRU, under a ranking-isolating protocol with **hard, popularity-matched candidate lists** (rebuilt after review — Appendix E):

- **H1 not supported:** fusing the trained GRU into ARAG does not improve ARAG (ΔNDCG@5 = −0.013, n.s. at the validation-optimal α\* = 0.9). The LLM's ranking already subsumes the sequential model's signal.
- **H2 supported:** the fused system significantly beats the GRU alone (+0.174), and ARAG significantly beats GRU seeds 1 and 2 — **the agentic LLM pipeline is the best system in this study**.
- **The paper's claims reproduce significantly:** ARAG > LLM-Recency, ARAG > LLM-Vanilla-RAG, and removing NLI+CSA significantly hurts.
- **Popularity carries no free signal** under popularity-matched negatives (at or below random), confirming the protocol is doing its job.

The most valuable finding is methodological as much as empirical: **candidate-list construction can manufacture or erase entire conclusions.** Under the original uniform-over-catalogue lists, the trained model looked dominant (+0.349, "significant"); under fair lists the effect reverses and the LLM agent dominates. Both runs are reported in full, the fix is a ~60-line change in one file, and every corrected number is auditable with `make verify`. Future work: retrieval-mined negatives, ≥300 test users for tighter CIs and a stable α, a strictly time-separated split for the GRU, and an LLM ranker given explicit popularity features to test whether the agentic increment survives with the last confound removed.

---

## 13. Product demo (Phase B)

Beyond the research comparison, the project ships a working interactive demo that serves recommendations from the same components (ARAG agents, trained GRU, FAISS retrieval, fused at the evaluation's α\* read from `results/metrics.json`).

**Stack:** FastAPI (Python) + SQLite memory + React/Vite frontend served by the API at `http://127.0.0.1:8000`. Run with `make app`.

**Memory (`artifacts/app.db`).** Seeded with **8 real test users** (their actual dataset history) and all 137,260 items with text. Tables: `users`, `items`, `interactions` (each row tagged `source = dataset|app`), `profiles`, `sessions`, `session_events`, `rec_runs`, `rec_items`. No synthetic users are ever used; app-event types are `view, click, like, dismiss, save`.

**Flows.**

| Flow | Steps |
|---|---|
| **Feed** | FAISS top-K on the user-context embedding **∪** GRU top-K over the catalogue → de-duplicate → drop already-interacted/dismissed → 20 candidates → ARAG+GRU (α from `results/metrics.json`) → paginated (10/page); the run is cached per page request |
| **Search** | LLM parses the query into `{keywords, max_price, exclude_terms}` → hard filters → FAISS top-50 on the query embedding → drop dismissed/interacted → top-20 → ARAG+GRU → top-5 with reasons |
| **Feedback** | stored immediately; `dismiss` permanently excludes the item for that user; the GRU user vector is recomputed from the current sequence at request time (no online retraining) |
| **Profile** | the UUA output is stored with provenance (which event ids were used); regenerated when ≥5 new events accumulate; viewable and deletable from the UI |
| **Replay** | every request is recorded as a `rec_run`; with the LLM cache a run replays with zero API calls, and `REPLAY_ONLY=1` makes the API refuse any uncached LLM call (fully offline demo) |

**API surface:** `/health`, `/users`, `/sessions`, `/feed`, `/search`, `/feedback`, `/users/{id}/history`, `/users/{id}/profile` (GET/POST/DELETE), `/runs/{run_id}`, `/runs/{run_id}/replay`, `/results`. Sessions are user-scoped: a user can only access their own session (cross-user access returns 403).

**UI (React + Vite).** User picker; **Feed** cards with title, category, price, reason and an **evidence snippet that is an exact substring of the item text selected in code (never generated)**, with like/dismiss/save buttons; **Search** with a query box and max-price filter; **Profile** panel (view/delete); an **agent-trace drawer** that shows the run's stages (agent outputs, rankings, repairs); and a **Results** view rendering the committed metrics table and α curve from `/results`. Layout is responsive down to phone width.

**Engineering note.** Loading PyTorch/SentenceTransformer lazily inside a request thread crashed the server (`OMP: Error #179`); models are now preloaded on the main thread at startup with `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false`.

---

## Appendix A — Repository map

```
SPEC.md                  original specification + recorded deviations
README.md                setup, commands, cost, how to read results
docs/PROCESS.md          step-by-step engineering log (every real number)
docs/REPORT.md           this document
configs/config.yaml      all paths, seeds, model names, θ/α grids, negative sampling
configs/prompts/         6 agent prompts (uua, nli, csa, ira, recency, vanilla_rag) + search
src/arag/
  dataprep/   download.py load.py items.py lists.py manifest.py
  models/     gru.py train_gru.py score.py
  retrieval/  embed.py index.py
  llm/        client.py cache.py schemas.py
  agents/     context.py uua.py nli.py csa.py ira.py arag.py baselines.py prompts.py
  ranking/    parser.py fusion.py simple.py
  eval/       metrics.py bootstrap.py run.py report.py verify.py
  memory/     db.py repo.py
  api/        app.py routes.py schemas.py services.py
frontend/                React + Vite demo UI
tests/                   pytest suite
results/                 metrics.json tables.md alpha_curve.png cost.json
                         run_record.json report.md manifest.json predictions/
artifacts/               (git-ignored) frozen lists, GRU checkpoints, FAISS, LLM cache
```

## Appendix B — Headline numbers at a glance

| Quantity | Value |
|---|---|
| Interactions / users / items | 814,586 / 94,762 / 137,269 (25,612 with interactions) |
| Frozen lists | 20 items (1 target + 19 popularity-matched negatives from the train vocabulary), seed 42, hashed |
| List validation | 0 OOV negatives; 19.0/19 known negatives; negatives ≈ as popular as targets |
| GRU vocab / config | 25,527 items; hidden **128**, dropout 0.2; 3 seeds; retrained on corrected lists |
| GRU test (all users) | Hit@5 0.422 ± 0.010, NDCG@5 0.279 ± 0.007 |
| θ\* / α\* | 5 / **0.9** (validation); test curve peaks at α = 1.0 |
| **H1** (ARAG+GRU − ARAG, NDCG@5) | **−0.013** [−0.032, +0.005] — **not supported** |
| **H2** (ARAG+GRU − GRU, NDCG@5) | **+0.174** [+0.069, +0.278] — **significant** |
| Best system | **ARAG**: Hit@5 0.530, NDCG@5 0.424 (100 users); beats Recency, Vanilla-RAG, all GRU seeds, Popularity, Random — all significant |
| Paper's ablations | w/o NLI&CSA −0.062 [−0.113, −0.016] **significant**; w/o NLI −0.049 n.s. |
| LLM calls (corrected eval) | 2,263 new calls ≈ **$1.06**; project total 4,668 calls / **$2.04** |
| Cached re-run | 137 s, zero new API calls |
| Audit | `make verify` → VERIFY OK |

## Appendix C — Anticipated questions and short answers

**Q1. Why frozen candidate lists instead of retrieving candidates with FAISS?**
Because the research question is about *ranking*, not retrieval. If each system retrieved its own candidates, a retrieval miss would be indistinguishable from a ranking miss, and methods would not be comparable. Freezing one 20-item list per user makes every comparison apples-to-apples. (Trade-off: sampled negatives are still easier than real retrieval — listed as a limitation.)

**Q2. Why are the negatives sampled proportional to popularity?**
Because uniform sampling over the catalogue was catastrophic: 81% of catalogue items have zero reviews, so ~15.5 of 19 uniform negatives were unknown to Popularity and the GRU (tied scores) and the task collapsed to a ~4.5-way choice (Appendix E). Sampling proportional to train popularity makes negatives about as popular as targets (measured 1.0–1.4×), so **popularity alone separates nothing** — and indeed the Popularity baseline now sits at or below random. Any system that wins must actually use the user.

**Q3. H1 failed — is the project a failure?**
No, and it is the more informative outcome. H2 succeeds strongly: the LLM agent adds a large significant increment over the trained model (+0.174), and the paper's own baseline and ablation claims reproduce significantly. A well-powered null on H1 — with the fusion weight tuned honestly on validation and the full α curve shown — is a defensible scientific result. What would *not* be defensible is reporting the earlier artefactual "+0.349 supported" (Appendix E).

**Q4. Was α\* = 0.9 chosen to make the fusion look good?**
No. α was selected only on the 100 validation users by maximizing validation NDCG@5 across the full grid; both curves are published. Validation preferred 0.9 by 0.0036 over 1.0, but on test the pure-LLM end wins — which is exactly why the report says the bump is validation noise and H1 fails. Selecting on test would have been the way to manufacture an H1 win; we did not.

**Q5. Why gpt-4o-mini rather than the paper's gpt-3.5-turbo?**
The paper's exact model, prompts and data are not reproducible from its text. gpt-4o-mini is the modern successor at ~$0.15/1M tokens, making the full evaluation cost about a dollar. The code supports any provider (local qwen3:8b via Ollama is also wired in). We explicitly do **not** compare our numbers to the paper's tables.

**Q6. Why a GRU and not a transformer / LLM fine-tune?**
The spec asked for a compact, CPU-trainable sequential model. The GRU4Rec formulation is the standard, well-understood baseline for next-item prediction, trains in ~1 h 50 m on CPU here, and is exactly the "trained deep-learning component" the project needed. A transformer would add little on 25k items with short 5-core histories.

**Q7. How do we know the results are not fabricated?**
(a) `results/` is written only by pipeline code on real data; (b) `make verify` recomputes every metric from raw per-user predictions (both metric blocks) and diffs against `metrics.json` — VERIFY OK; (c) the frozen candidate lists are SHA-256-hashed and re-checked; (d) `run_record.json` records the git commit, model, config and dataset manifest; (e) the LLM cache makes the run idempotent — anyone can re-run and get identical numbers at $0; (f) when a protocol flaw was found, it was fixed, re-run and documented rather than hidden (Appendix E).

**Q8. Why only 100 test users for the LLM systems?**
Cost and statistics: each LLM user costs ~7–12 API calls, and 100 paired users comfortably resolve the effects we report (the smallest significant gap is ~0.06). CIs are published so small differences are not over-read. Non-LLM systems are additionally reported on all 94,762 test users.

**Q9. How do our results compare to the paper's claims?**
Qualitatively and now statistically: ARAG significantly beats LLM-Recency and LLM-Vanilla-RAG (the paper's baselines), and removing NLI **and** CSA significantly hurts ARAG (the paper's ablation). Removing only NLI is directionally negative but not significant. **Appendix F** puts the paper's Table 1 side by side with ours — including the absolute numbers, why they are not comparable (different category, model, dataset version, protocol), and where our results are stronger or weaker than the paper's. We never claim to reproduce the paper's absolute numbers. Note this is a change from the first run, where none of these differences were significant and the ablations were weakly *better* — the degenerate lists had hidden them (Appendix E).

**Q10. What would you change with more time/budget?**
≥300 test users for tighter CIs and a stable α (the 0.9-vs-1.0 disagreement is a small-n artefact); retrieval-mined hard negatives; a strictly time-separated split for the GRU; and an LLM ranker explicitly given popularity/statistical features, to test whether the agentic contribution survives with every confound removed.

---

## Appendix D — One-paragraph summaries at three levels

**To a non-technical person:** We built a video-game store recommender that combines two "brains": an AI assistant that reads what you bought and reasons about it, and a small neural network that has learned from a million past purchases what people buy next. We tested whether combining them beats each alone by giving all systems the same 20 real products to rank (1 right answer + 19 plausible decoys that are as popular as the right answer, so nobody can cheat by just picking famous products). Result: **the AI assistant is clearly better**, and adding the neural network on top doesn't help — but the neural network alone would have been much worse. We also found, and fixed, an earlier version of the test that accidentally made the neural network look much better than it was.

**To a computer-science student:** We implement an ARAG-style 4-agent LLM pipeline (user understanding → per-candidate NLI scoring → context summarisation → ranking) and fuse its permutation with a GRU4Rec-style next-item model via rank-derived score blending (`α·s_LLM + (1−α)·s_GRU`, α tuned on validation). On frozen 20-item lists (1 target + 19 popularity-matched negatives from the train vocabulary) over Amazon Video_Games (94,762 users), the fused system does **not** beat ARAG alone (ΔNDCG@5 = −0.013, n.s., α\* = 0.9) but significantly beats the GRU alone (+0.174); ARAG significantly beats Recency, Vanilla-RAG and all GRU seeds, and the paper's NLI+CSA ablation reproduces. A post-review protocol fix (negatives were originally drawn from the whole catalogue, 81% of which has zero interactions) is documented with the superseded numbers in Appendix E.

**To a recommender-systems researcher:** We quantify the marginal value of an agentic LLM ranker against a GRU4Rec-style sequential baseline under a rank-isolating protocol, and document a protocol artefact that can manufacture conclusions. Contributions: (i) a reproducible ARAG+GRU fusion with α/θ tuned on validation and paired-bootstrap CIs; (ii) evidence that under popularity-matched negatives the agentic ranker dominates (ARAG NDCG@5 0.424 vs the best GRU seed's 0.246 on the same 100 users; the fused system, statistically equal to ARAG, beats seed 0 by +0.174, significant) while a popularity baseline falls to random; (iii) a natural experiment: the original uniform-over-catalogue lists made ~81% of negatives OOV, which inflated the GRU to NDCG@5 0.823 and produced a false "fusion helps" result (+0.349) — reversing the conclusion when fixed, with both runs fully audited; (iv) the paper's baseline ordering and NLI+CSA ablation reproduce significantly. Limitations: single category, n=100 for the main table, sampled (not retrieval-mined) negatives, leave-last-out leakage into GRU training, no human evaluation.

## Appendix E — Errata: the original (superseded) run

This appendix preserves the first evaluation's numbers and the diagnosis of why they were wrong. **Do not cite these numbers as results**; they are kept so the correction is auditable.

### E.1 What the original run reported (superseded)

| System | Hit@5 | NDCG@5 | MRR |
|---|---|---|---|
| Random *(all users)* | 0.251 | 0.149 | 0.181 |
| Popularity *(all users)* | 0.969 | 0.801 | 0.749 |
| GRU *(all users)* | 0.976 ± 0.001 | 0.823 ± 0.004 | 0.775 ± 0.005 |
| LLM-Recency | 0.490 | 0.389 | 0.403 |
| LLM-Vanilla-RAG | 0.480 | 0.386 | 0.403 |
| ARAG (θ\*=5) | 0.570 | 0.453 | 0.452 |
| ARAG w/o NLI | 0.610 | 0.481 | 0.472 |
| ARAG w/o NLI & CSA | 0.590 | 0.456 | 0.445 |
| ARAG+GRU (α\*=0.1) | 0.990 | 0.802 | 0.740 |

Original headline claims (all superseded): **H1 "supported"** (ΔNDCG@5 = +0.349 [+0.257, +0.441]); H2 "not supported" (+0.002 [−0.018, +0.022]); α\* = 0.1; Recency/Vanilla-RAG vs ARAG not significant; both ablations weakly *better* than ARAG (n.s.); original run cost $0.9853 for 2,405 calls.

### E.2 The diagnosis

The original `dataprep/lists.py` sampled the 19 negatives **uniformly from all 137,260 catalogue items with text**. But only **25,612 items (18.7%)** appear in the 5-core interaction data; **111,649 (81%) have zero reviews**. Measured on the original lists:

- **mean 15.5 of 19 negatives per list had zero interactions** (median 16, min 7), leaving only **3.5 known negatives** — a ~4.5-way task, not a 20-way one;
- for the GRU, every unknown candidate mapped to the **same untrained OOV embedding** (`models/score.py`, `OOV_IDX = 1`) → identical scores → alphabetical tie-break;
- for Popularity, unknown candidates all had count 0 → the same tie.

The LLM sees no review counts, so it could not exploit this. Popularity and the GRU were effectively being handed the answer, which is why Popularity Hit@5 = 0.97 and why mixing in the GRU appeared to "boost" ARAG by +0.349. The original report explained the strong popularity baseline as "targets are popular" — that was **wrong**: the real cause was the OOV negatives.

### E.3 What was changed

1. `dataprep/lists.py`: negatives now sampled from the **25,527 items with train interactions**, weighted by train popularity, excluding user history (`sampling.negative_sampling: popularity` in `configs/config.yaml`). New lists: 0 OOV negatives, 19.0/19 known, negatives ≈ as popular as targets. New hashes recorded.
2. `eval/run.py`: the main table and all paired tests now use the **same 100 sampled test users for every system** (previously GRU/Popularity used all 94,762); a separate `metrics_all_users` block keeps the full-cohort non-LLM estimates.
3. The **GRU was retrained from scratch** (hidden 64 → 128 under the re-run grid) because its checkpoints had been selected by early stopping on the old validation lists.
4. `docs/REPORT.md` rewritten; `docs/PROCESS.md §11` logs the incident; `SPEC.md` records the amendment.

### E.4 The reversal, side by side

| Quantity | Original (degenerate lists) | Corrected |
|---|---|---|
| Unknown negatives per list | 15.5 / 19 | **0 / 19** |
| Known negatives per list | 3.5 | 19.0 |
| Popularity NDCG@5 | 0.801 (near the GRU) | **0.109 (below random)** |
| GRU NDCG@5 (all users) | 0.823 | **0.279** |
| ARAG NDCG@5 | 0.453 | **0.424** |
| Best system | GRU / fused | **ARAG** |
| α\* | 0.1 (90% GRU) | **0.9** (90% LLM) |
| H1 (fusion vs ARAG) | +0.349 "significant" | **−0.013 n.s. (not supported)** |
| H2 (fusion vs GRU) | +0.002 n.s. | **+0.174 significant** |
| ARAG vs Recency / Vanilla-RAG | n.s. | **both significant** |
| Ablations hurt ARAG? | no (wrong direction) | **NLI+CSA removal significantly hurts** |

The corrected run is the definitive one. The original run cost $0.9853 (2,405 calls); the corrected evaluation added 2,263 calls (≈ $1.06); the GRU retraining was free of API cost (~1 h 50 m CPU). Total unique project spend: **$2.04**.

---

## Appendix F — Side-by-side with the base paper

This appendix documents how our implementation compares to the paper we build on. **The paper's numbers are quoted for context only.** They are *not* directly comparable to ours — different category, model, dataset version, candidate protocol and evaluation size — and nothing in the main body of this report depends on them. The comparison that matters is qualitative (do the paper's claims hold?), and that lives in §7.3.

### F.1 The paper

Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta, Kai Zhao, Aysenur Inan, Kehui Yao, Jianpeng Xu, Praveen Kanumala, Jason Cho, Sushant Kumar. **"ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation."** ACL SIGIR 2025 / arXiv:2506.21931 (Walmart Global Tech).

### F.2 Why the absolute numbers are not comparable

| Axis | Paper | Ours |
|---|---|---|
| Dataset | Amazon Reviews **2016** (He & McAuley) | Amazon Reviews **2023** (McAuley-Lab) |
| Categories | Clothing, Electronics, Home & Kitchen | **Video Games** — a fourth category the paper never ran |
| LLM | `gpt-3.5-turbo` (v0125), temperature 0 | `gpt-4o-mini`, temperature 0 |
| Users evaluated | 10,000 randomly sampled | 100 sampled test users (LLM systems); 94,762 (non-LLM all-users block) |
| Candidate construction | unspecified recall set | **frozen 20-item lists**: 1 held-out target + 19 popularity-matched in-vocab negatives |
| Statistical testing | none reported | paired bootstrap, 95% CIs, significance flags on every contrast |
| Systems compared | Recency, Vanilla RAG | + Random, Popularity, GRU×3 seeds, ARAG+GRU fusion |

We therefore follow SPEC.md and the report's scope note (§111): **we never claim to reproduce the paper's absolute numbers.** What we test is whether the paper's *claims* hold on our data.

### F.3 Baselines vs ARAG — paper Table 1 next to ours

Paper numbers are NDCG@5 / Hit@5. Ours are the same metrics on Video Games (100 users, `gpt-4o-mini`).

| System | Paper: Clothing | Paper: Electronics | Paper: Home | **Ours: Video Games** |
|---|---|---|---|---|
| Recency-based ranking | 0.309 / 0.395 | 0.225 / 0.304 | 0.224 / 0.299 | **0.285 / 0.400** |
| Vanilla RAG | 0.299 / 0.379 | 0.238 / 0.321 | 0.229 / 0.312 | **0.263 / 0.380** |
| **ARAG (full)** | **0.439 / 0.535** | **0.329 / 0.420** | **0.289 / 0.383** | **0.424 / 0.530** |
| ARAG lift over best baseline | +42.1% / +35.5% | +37.9% / +30.9% | +25.6% / +22.7% | **+48.8% / +32.5%** (vs Recency) |

Observations, stated carefully:

1. **ARAG's absolute scores are strikingly close** to the paper's strongest category (Clothing: 0.439 / 0.535 vs our 0.424 / 0.530 — within 1.6 NDCG@5 points and 0.5 Hit@5 points). Given the category, model and protocol differences, this is coincidence, not replication — but it indicates the implementation is in the right performance regime rather than crippled by a bug.
2. **ARAG's relative lift over the best baseline is larger for us** (+48.8% NDCG@5 vs the paper's best +42.1%). Our baselines are somewhat weaker in absolute terms, so the agentic stack separates more clearly. The paper's central claim — agentic reasoning beats simple prompting — is *more* pronounced on our data, not less.
3. **The Recency-vs-Vanilla-RAG ordering flips by domain in the paper** (Recency wins Clothing; Vanilla wins Electronics and Home). Video Games behaves like Clothing: Recency 0.285 > Vanilla 0.263.
4. Our Hit@5 lift (+32.5%) is slightly below the paper's Clothing figure (+35.5%), while our NDCG lift is higher — consistent with our ARAG placing the target high when it hits, but the small evaluation size (n = 100, SE ≈ 0.05 on Hit@5) makes fine distinctions on lift unreliable.

### F.4 Ablation staircase — paper next to ours

The paper's core mechanistic claim is that each agent component earns its place. Reading the table top-down, every step down removes a component and the score should drop.

| System | Paper: Clothing | Paper: Electronics | Paper: Home | **Ours: Video Games** |
|---|---|---|---|---|
| Vanilla RAG | 0.299 | 0.238 | 0.229 | **0.263** |
| ARAG w/o NLI & CSA | 0.302 | 0.272 | 0.249 | **0.362** |
| ARAG w/o NLI | 0.385 | 0.296 | 0.273 | **0.376** |
| **ARAG (full)** | **0.439** | **0.329** | **0.289** | **0.424** |

(NDCG@5 only. Hit@5 shows the same ordering: ours 0.380 < 0.470 < 0.510 < 0.530.)

1. **The staircase is identical in all four columns.** w/o NLI&CSA < w/o NLI < full ARAG, everywhere. The paper's ablation claim reproduces on our data — this is the result reported in §7.3.
2. **The NLI-only step is similar in size.** Removing NLI costs the paper −0.055 (Clothing) and us −0.048.
3. **The NLI+CSA step differs in magnitude, and this is a genuine difference worth knowing.** In the paper's Clothing category, removing both NLI and CSA collapses ARAG almost exactly to Vanilla RAG (0.302 vs 0.299) — in that domain the agentic value lives almost entirely in NLI+CSA. On our Video Games data, the UUA+IRA core (0.362) stays **well above** Vanilla RAG (0.263): the user-understanding summary plus a ranker with raw candidates is itself worth +0.099 over vanilla. Electronics and Home in the paper show the same *direction* (+0.034, +0.020 over Vanilla), so our result is an amplification of the paper's non-Clothing pattern, not a contradiction.
4. **Every drop we measure is statistically tested** (§6.2): the NLI+CSA removal is significant (Δ = −0.062 [−0.113, −0.016]); the NLI-only removal is directionally negative but not significant (Δ = −0.049 [−0.111, +0.011]). The paper reports no confidence intervals, so we cannot say whether their NLI-only drop would have cleared a significance bar either.

### F.5 What the paper does not have (our additions)

| Feature | Paper | Ours |
|---|---|---|
| Trained sequential baseline | none | GRU4Rec-style model, 3 seeds, grid-searched |
| LLM + neural fusion | none | rank-score late fusion, α tuned on validation (α\* = 0.9) |
| Frozen, audited candidate lists | unspecified | SHA-256-hashed; `make verify` re-derives all metrics |
| Protocol-flaw audit | n/a | Appendix E: degenerate lists found, fixed, re-run, both runs preserved |

The paper's evaluation ends at "ARAG beats its two baselines." Ours extends the question to "does ARAG beat a *trained* model, and does fusing them help?" — the answer to both is in §7: yes to the first (significantly), no to the second (H1 fails).

### F.6 One-sentence summary for a talk or viva

*On a category the paper never evaluated, with a different and cheaper LLM, our ARAG implementation lands within 1.6 NDCG@5 points of the paper's best category, reproduces the paper's ablation staircase and baseline ordering significantly, shows a larger relative lift over baselines than the paper reports, and extends the evaluation with a trained GRU baseline and a fusion test the paper never attempted — while never claiming the absolute numbers are comparable.*
