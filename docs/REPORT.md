# ARAG + GRU
## Extending an Agentic LLM Recommender with a Trained GRU Sequential Model — Full Project Report

**Course deliverable — reproducibility-first report**

> **Outdated — do not cite these numbers.** This report describes a run whose candidate
> lists drew negatives from all 137k catalogue items. About 14 of the 19 negatives per list
> had no train interactions, so Popularity and the GRU only had to pick the target out of
> ~6 known items (Popularity Hit@5 = 0.97), and the "H1 supported" result is an artefact of
> that. Negatives are now sampled from items with train interactions, proportional to
> popularity (`sampling.negative_sampling` in `configs/config.yaml`). Re-run
> `make lists gru eval report` and rewrite this report from the new `results/`.
Companion documents: `SPEC.md` (specification & recorded deviations), `docs/PROCESS.md` (step-by-step engineering log with every real number), `results/report.md` (auto-generated results summary).

---

## Abstract

This project implements an **ARAG-style agentic recommender** (Maragheh et al., 2025, arXiv:2506.21931) on the Amazon Reviews 2023 **Video Games** category and extends it with **one trained deep-learning component: a PyTorch GRU sequential user model late-fused into the final LLM ranking**. All systems rank the *same frozen 20-item candidate lists* (1 held-out target + 19 uniform negatives) per user, so the comparison isolates **ranking** from retrieval.

Using OpenAI `gpt-4o-mini` for the agent pipeline and 100 sampled test users (plus all 94,762 test users for non-LLM systems), we find:

- **H1 supported:** ARAG+GRU significantly outperforms ARAG alone (ΔNDCG@5 = **+0.349**, 95% CI [+0.257, +0.441]).
- **H2 not supported:** ARAG+GRU is statistically indistinguishable from the GRU alone (ΔNDCG@5 = **+0.002**, 95% CI [−0.018, +0.022]). The validation-optimal fusion weight is **α\* = 0.1** (90% GRU / 10% ARAG), i.e. the trained sequential model carries almost all of the ranking signal under this protocol.
- **Protocol insight:** a *pure popularity* baseline (NDCG@5 = 0.801) is nearly as strong as the trained GRU (0.823) and far stronger than every LLM system (≤ 0.481). With uniform random negatives, global item popularity is a dominant signal, which strongly compresses the headroom for LLM-side improvements and is the main explanation for the H2 outcome.

All numbers are produced by a deterministic, audited pipeline: **`make verify` re-derives every reported metric from the raw per-user predictions and checks dataset hashes and the run record — it passes on the committed artifacts.** The total LLM cost of the full evaluation was **$0.9853**.

---

## 0. How to read this document (orientation, glossary, diagram)

**If you have 30 seconds:** the project asks whether adding a *trained* neural sequence model (GRU) to an *LLM agent* recommender (ARAG) improves ranking. Answer: **yes vs the LLM alone (H1), no vs the trained model alone (H2)** — the GRU carries almost all the signal on this evaluation protocol, and a popularity baseline is nearly as strong as the GRU because the candidate lists use uniform random negatives.

**If you have 10 minutes:** read §1 (questions), §3.2 (frozen candidate lists — the key design decision), §4 (system), §6.1–6.3 (results), §7.2 (why H2 fails), §8 (limitations).

**If you must defend it:** every number is reproducible — `make verify` re-derives all metrics from raw predictions and checks dataset hashes; §10 lists the full command sequence; §9 lists every deviation from the original spec.

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
| **Vocabulary / OOV** | The set of item ids the GRU knows (from train data); items outside it are "out-of-vocabulary" and mapped to a special OOV embedding. |
| **Candidate list (frozen)** | The 20 products each user is ranked against: their true held-out **target** + 19 random **negatives**. Frozen so every system ranks identical items. |
| **Target / negatives** | The one item the user actually interacted with next (target) and the 19 decoys the user never touched (negatives). |
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
                                              final = α·s_LLM + (1−α)·s_GRU   (α* = 0.1)
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

**Scope note (stated up front).** This is **ARAG-inspired, not an exact reproduction**: the paper does not publish prompts, the NLI threshold, candidate-list construction or code, and it used gpt-3.5-turbo on an older Amazon dataset. Our numbers must never be compared directly with the paper's Table 1. Negative or mixed results are treated as valid outcomes and reported as measured.

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
| Items | 137,269 metadata rows, 137,260 with text; 0 interacted items lacked metadata |
| Item text | product metadata only (title, main_category, categories, features, description), truncated to **120 words**; review text is never used |

Leave-last-out semantics: per user, the **last** interaction is the test target, the **second-to-last** is the validation target, and the rest form the train history. Assertions enforce `max(train ts) ≤ valid ts ≤ test ts` and that targets never appear in their own history (no violations; `make lists` fails hard otherwise).

### 3.2 Frozen candidate lists (the central design decision)

For every validation and test user we construct a **20-item candidate list**:

> the user's true held-out target + **19 negatives sampled uniformly** from items that (i) have text and (ii) the user never interacted with in **any** split; the list is shuffled with a seeded RNG (seed 42) and then **frozen**.

Every system — Random, Popularity, GRU, the LLM baselines, ARAG, the ablations and the fused system — ranks this identical list. This is deliberate:

- If each method retrieved its own candidates, a missed retrieval would score 0 for that method and the experiment would measure **retrieval**, not **ranking**.
- A frozen list is the only way to attribute a Hit@5 difference to *ranking quality*.
- The expected Hit@5 of a uniformly random permutation on a 20-item list is exactly **0.25** (empirically 0.251), which we use as a sanity anchor.

Lists are hashed (`artifacts/lists/lists_hashes.json`) and re-verified by `make verify`; they are never regenerated after evaluation begins.

### 3.3 Evaluation users

- Every **LLM-based** system is evaluated on the same **100 validation + 100 test users** (sampled with seed 42 from users whose target item has text).
- Non-LLM systems (Random, Popularity, GRU) are additionally reported on **all 94,762 test users**, which gives near-deterministic estimates for those methods.
- The demo (Phase B) seeds 8 of the same real test users; no synthetic users exist anywhere in the evaluation.

### 3.4 Metrics

For a single relevant item per user with rank *r* (1-indexed):

- **Hit@K** = 1 if *r ≤ K* else 0
- **NDCG@K** = 1/log₂(r+1) if *r ≤ K* else 0
- **MRR** = 1/r

Reported metrics: Hit@5, NDCG@5, Hit@10, NDCG@10, MRR. Primary metrics are **Hit@5 and NDCG@5**.

### 3.5 Statistical testing

Every system is compared with **paired bootstrap** over users (10,000 resamples, 95% percentile CIs of the mean per-user metric difference), versus **ARAG** and versus **GRU**. A CI containing 0 is reported as *no significant difference*. Comparisons use only the users shared by both systems.

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

- **Architecture:** item embedding (64-d) → dropout → 1-layer GRU (hidden 64) → dropout → linear projection (64-d) → full-softmax over the vocabulary, with **input/output embeddings tied**.
- **Vocabulary:** items seen in the train split (25,527); index 0 = padding, 1 = OOV.
- **User vector:** h_u = last valid hidden state (projected to 64-d). Item score = h_u · e_i.
- **Training:** last 50 train items per user, right-padded, `pack_padded_sequence`; next-item prediction on every prefix; full-softmax cross-entropy; Adam lr 1e-3; batch 256; ≤12 epochs with early stopping (patience 3) on **validation NDCG@5 over all 94,762 validation users** against the frozen lists.
- **Hyperparameter grid:** hidden ∈ {64, 128} × dropout ∈ {0.2, 0.5}, then **3 seeds** of the best config.
- **Result:** best config hidden=64, dropout=0.2; seed validation NDCG@5 = 0.8409 / 0.8455 / 0.8385. Test (all 94,762 users): **Hit@5 0.976 ± 0.001, NDCG@5 0.823 ± 0.004, MRR 0.775 ± 0.005**.

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

- **θ (NLI acceptance threshold):** validation NDCG@5 at θ ∈ {4, 5, 6, 7} = 0.4606 / **0.4890** / 0.4574 / 0.4772 → **θ\* = 5**.
- **α (fusion weight):** validation NDCG@5 peaks at **α\* = 0.1** (0.8186), with α = 0 at 0.8071 and α = 1 at 0.4890.

Both are selected **only** on the 100 validation users; the test set is never used for tuning.

### 5.3 Model & cost

- LLM: OpenAI **gpt-4o-mini** (temperature 0). Provider is configurable (local Ollama `qwen3:8b` is supported through the same client; the committed results use gpt-4o-mini — see §9).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalized) + FAISS `IndexFlatIP`.
- **Cost:** 2,405 unique LLM calls, 4,886,695 input / 420,439 output tokens, **$0.9853** at $0.15/$0.60 per 1M. A cached re-run completes in **136 s** at **$0**.

---

## 6. Results

### 6.1 Main comparison (test)

LLM systems on the **100 sampled test users**; non-LLM systems on **all 94,762 test users** (flagged). GRU = mean over seeds 0/1/2.

| System | Hit@5 | NDCG@5 | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Random *(all users)* | 0.251 | 0.149 | 0.502 | 0.229 | 0.181 |
| Popularity *(all users)* | 0.969 | 0.801 | 0.996 | 0.810 | 0.749 |
| GRU *(all users; mean ± std)* | 0.976 ± 0.001 | **0.823 ± 0.004** | 0.994 | 0.828 | 0.775 |
| LLM-Recency | 0.490 | 0.389 | 0.670 | 0.447 | 0.403 |
| LLM-Vanilla-RAG | 0.480 | 0.386 | 0.670 | 0.448 | 0.403 |
| ARAG (θ\*=5) | 0.570 | 0.453 | 0.730 | 0.504 | 0.452 |
| ARAG w/o NLI | 0.610 | 0.481 | 0.720 | 0.516 | 0.472 |
| ARAG w/o NLI & CSA | 0.590 | 0.456 | 0.690 | 0.488 | 0.445 |
| **ARAG+GRU (α\*=0.1)** | **0.990** | 0.802 | **1.000** | 0.805 | 0.740 |

### 6.2 Hypothesis tests (paired bootstrap, ΔNDCG@5, 95% CI)

| Comparison | ΔNDCG@5 | 95% CI | Verdict |
|---|---|---|---|
| **ARAG+GRU − ARAG (H1)** | **+0.349** | [+0.257, +0.441] | **significant** |
| **ARAG+GRU − GRU (H2)** | +0.002 | [−0.018, +0.022] | **not significant** |
| ARAG+GRU − ARAG, Hit@5 | +0.420 | [+0.320, +0.520] | significant |
| ARAG+GRU − GRU, Hit@5 | 0.000 | [0.000, 0.000] | not significant |
| LLM-Recency − ARAG | −0.064 | [−0.152, +0.022] | not significant |
| LLM-Vanilla-RAG − ARAG | −0.068 | [−0.151, +0.014] | not significant |
| ARAG w/o NLI − ARAG | +0.027 | [−0.025, +0.084] | not significant |
| ARAG w/o NLI & CSA − ARAG | +0.003 | [−0.045, +0.053] | not significant |
| Popularity − GRU *(all users)* | −0.021 | [−0.023, −0.017] | GRU significantly better (small) |
| Popularity − ARAG | +0.360 | [+0.262, +0.458] | significant |
| GRU − ARAG *(seed 1)* | +0.338 | [+0.241, +0.436] | significant |

### 6.3 The α curve (validation and test)

| α | 0.0 | **0.1** | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| validation NDCG@5 | 0.8071 | **0.8186** | 0.7826 | 0.7747 | 0.6899 | 0.6080 | 0.5207 | 0.4890 |
| test NDCG@5 | 0.7995 | **0.8017** | 0.7641 | 0.7482 | 0.6530 | 0.5677 | 0.4841 | 0.4532 |

The curve is monotone decreasing in α except for the small α\*=0.1 bump: **the more weight on the LLM ranking, the worse the result.** On test the optimum sits essentially at pure GRU. The full curve is plotted in `results/alpha_curve.png`.

---

### 6.4 Worked example — one real test user, end to end

This is real data from the evaluation (not an illustration). User `AE24RBZBFTNQ2ZTQIVY5GTEEOS2A`, a PC-gaming peripherals buyer.

**Their history (train + validation, in time order):**

| # | Item | Title (truncated) |
|---|---|---|
| 1 | B00HTK1NCS | Redragon M601 RGB Gaming Mouse |
| 2 | B0C2CYCX75 | EasySMX 2.4G Wireless Controller for PS3/PC |
| 3 | B00E4MQODC | Logitech G602 Wireless Gaming Mouse |
| 4 | B01N3ASPNV | amFilm Tempered Glass Screen Protector for Nintendo Switch |
| 5 | B07DDN59PJ | CORSAIR SABRE RGB Gaming Mouse |

**Their frozen 20-item candidate list** (order as stored), with the true held-out target marked:

| # | Item | Title (truncated) |
|---|---|---|
| 1 | B08DV4ZFVQ | INPHIC Wireless Mouse |
| 2 | B0BY8X7VVB | Marada G920 Racing Wheel Stand |
| 3 | B07ZQ7S28B | Mr Wonderful Vinyl Skin for Nintendo |
| **4** | **B07JH4R6ZT** | **Cooler Master MH-752 Gaming Headset ← TARGET** |
| 5 | B09V5428B3 | NexiGo VR Stand |
| 6 | B00B920R9C | PS3 Modded Controller (Rapid Fire) |
| 7 | B07VWNN9BZ | Controller Repair Screw Set |
| 8 | B09YDQRZF8 | Linux Basics for Hackers (book) |
| 9 | B07Z3FV3X9 | FASTSNAIL Charging Dock for Switch |
| 10–20 | … | (thumb grips, skins, arcade kit, keyboard, makeup, CD-Rs, a Wii game — mostly irrelevant decoys) |

**How each system ranked the target:**

| System | Rank of target | Its top-5 (in order) |
|---|---|---|
| **ARAG (LLM)** | **6 (miss)** | wireless mouse, racing wheel stand, PS3 controller, repair kit, Switch dock |
| **GRU** | **2 (hit)** | Wii game, **headset (target)**, PS2 pad, CD-Rs, PS3 controller |
| **ARAG+GRU (α\*=0.1)** | **2 (hit)** | Wii game, **headset (target)**, PS2 pad, PS3 controller, CD-Rs |
| Popularity | 2 (hit) | Wii game, **headset (target)**, PS2 pad, CD-Rs, PS3 controller |

**What this tells you in one example:**

1. **ARAG reasons plausibly but misses.** Its top-5 are all gaming peripherals (sensible from the history), but it puts the headset 6th — the same category it should predict. An LLM reasoning over five title lines does not reliably pin the exact next item.
2. **The GRU gets it.** Trained to predict the next item across 625k interactions, it places the true next item 2nd. It has learned that "mouse/controller buyer → gaming headset" is a common continuation.
3. **The fusion inherits the GRU.** With α\*=0.1 the blended list is essentially the GRU's list, so the target lands 2nd (Hit@5). This single case is a microcosm of the aggregate result: fusion ≈ GRU ≫ ARAG on this protocol.
4. **Popularity also gets it** (2nd) — the headset is a popular item, illustrating §7.2's point: with random negatives, popularity alone already captures much of the signal, which is why the LLM's residual contribution is hard to detect.

---

## 7. Analysis and discussion

### 7.1 H1: fusing the trained model into ARAG helps, massively

Adding the GRU lifts ARAG from NDCG@5 0.453 to 0.802 (+0.349, CI excludes 0 by a wide margin). This is unsurprising in hindsight: the LLM ranks by *reasoning over one user's ~55 items of text*, while the GRU has learned next-item statistics from **625k interactions across the whole population** — collaborative signal that is literally absent from the prompt. On a leave-last-out target, next-item prediction is exactly the task the GRU was trained for.

### 7.2 H2: the LLM adds nothing measurable on top of the GRU, under this protocol

The fused system is statistically indistinguishable from the GRU alone (ΔNDCG@5 = +0.002, CI [−0.018, +0.022]), and the validation-optimal weight puts only 10% on the LLM. Three observations explain this:

1. **The GRU is near the protocol ceiling.** It already places the target in the top-5 for ~97.6% of users (all-user test). With 19 uniform negatives, most of the remaining error is not fixable by better reasoning — it is label noise / genuinely unpredictable targets.
2. **Uniform random negatives strongly favour popularity-like signals.** A pure *popularity* baseline reaches NDCG@5 = 0.801 — nearly the GRU's 0.823 and far above any LLM system. The target item of a 5-core user is typically a popular item, and random negatives are mostly obscure items; ranking by global frequency therefore scores very well. LLM systems receive no global popularity statistics (only one user's history), so they are structurally handicapped on this list construction.
3. **The paper's ordering** (LLM-Recency, LLM-Vanilla-RAG < ARAG) holds **directionally** (0.389, 0.386 < 0.453) but **not significantly** at n=100. The ablations (w/o NLI, w/o NLI&CSA) did **not** hurt ARAG in our run — the opposite of the paper's ablation result (they were weakly better, not significant). Both are consistent with the explanation that, on uniform negatives, elaborate agentic context-processing buys little over a one-shot ranking prompt.

**Interpretation.** The honest reading is: *the trained sequential model dominates this task; the agentic LLM pipeline contributes a small, non-significant increment.* This is a legitimate and informative outcome — it quantifies exactly what the LLM contributes on top of a strong trained baseline, rather than assuming a gain.

### 7.3 Secondary baselines and cost

- Popularity is a *very* strong baseline here (NDCG@5 0.801 on all users), though the GRU is still significantly better (Δ = −0.021, CI excludes 0) — a tiny but reliable margin over 94,762 users.
- LLM-Recency and LLM-Vanilla-RAG behave nearly identically (~0.39), i.e. the retrieval step in Vanilla-RAG adds nothing measurable in this setup.
- Total evaluation cost is **$0.9853**, well inside the ≈$1 budget estimated in the spec; a cached re-run is free (136 s) because every call is keyed by (model, prompt, schema).

---

## 8. Threats to validity and limitations

1. **Uniform negatives inflate popularity-like methods.** The frozen-list protocol (chosen to isolate ranking) makes the task easier than real retrieval and structurally favours Popularity and the GRU. Any headline number should be read as *"ranking quality on uniform 20-item lists"*, not as production retrieval quality.
2. **100 test users ⇒ wide CIs.** Roughly ±10 points on Hit@5; small differences (including H2) cannot be resolved. Non-LLM systems use all 94,762 test users, so those estimates are tight.
3. **Leave-last-out leakage into GRU training.** Other users' later interactions may appear in the GRU's training data; this is inherent to the official split and applies equally to all systems, but it means the GRU's advantage is an upper bound relative to a strictly time-separated setup.
4. **Single category, single dataset.** Video Games only; results may not transfer.
5. **Prompt and threshold sensitivity.** The paper does not publish prompts or θ; our prompt wording and θ\*=5 are our own choices, tuned only on validation.
6. **Model substitution.** The paper used gpt-3.5-turbo on an older dataset; we use gpt-4o-mini. This is ARAG-inspired, not a reproduction, and no number here is comparable to the paper's tables.
7. **No human evaluation.** All judgments are offline metrics against a single held-out item.

---

## 9. Deviations from the original specification (recorded, not hidden)

| # | Spec | What was done | Why |
|---|---|---|---|
| 1 | LLM = OpenAI gpt-4o-mini | **Yes — gpt-4o-mini for the committed results.** Local Ollama `qwen3:8b` is also supported through the same client | Local qwen3:8b measured ≈20 s/call with serial Ollama slots (~14 h for the eval); gpt-4o-mini completed the same eval in ~15 min for $0.99. The pipeline was developed and tested against both |
| 2 | GRU ≤30 epochs | **12 epochs** | Validation NDCG@5 had plateaued (epoch 9 ≈ 0.829, epoch 11 = 0.840); 30 epochs added hours for ranking-irrelevant gains |
| 3 | `llm/fake.py` test double | **Removed** | Per project decision: no fake LLM client anywhere; deterministic logic is tested with hand-made fixtures, integration tests hit a real provider |
| 4 | Exact file names in spec (`raw_meta_Video_Games`) | **Corrected to the real dataset paths** | The config `raw_meta_Video_Games` does not exist for Video_Games; metadata is `raw/meta_categories/meta_Video_Games.jsonl` (verified against the dataset card before coding) |
| 5 | Tracked prediction CSVs | **Capped to the 100 test users** | Full CSVs were ~147 MB; parquet in `artifacts/` holds every user |

---

## 10. Reproducibility and artifact integrity

- **`make verify`** recomputes every reported metric from the raw per-user predictions in `artifacts/predictions/`, diffs them against `results/metrics.json`, recomputes the SHA-256 of the frozen candidate lists, and checks the run record — currently **VERIFY OK**.
- `results/run_record.json` records the git commit, provider/model, category, seed, elapsed time, the dataset manifest (file hashes + row counts) and the full config.
- **No fabricated numbers anywhere.** `results/` is written only by pipeline code on real data; the unit/integration test fixtures are hand-made inputs that never touch `results/` or `artifacts/`.
- Determinism: fixed seeds (data 42; GRU 0/1/2), temperature 0, frozen lists, SHA-256-keyed LLM cache. Re-running the evaluation reproduces identical numbers with **zero** new API calls.

**Full reproduction (from a fresh clone):**

```bash
make setup          # venv + pinned pip deps + npm install
export OPENAI_API_KEY=...     # or switch config.yaml to provider: ollama
make data           # download + manifest + parquet tables
make lists          # sequences, samples, frozen candidate lists
make index          # MiniLM embeddings + FAISS
make gru            # GRU grid + 3 seeds          (~1 h 40 m CPU)
make eval           # all systems, θ/α tuning, bootstrap
make report         # results/*.md|json|png
make verify         # audit: recompute metrics, check hashes, run record
make test           # pytest suite
make app            # demo: FastAPI + built React frontend on 127.0.0.1:8000
```

---

## 11. Engineering issues found and fixed (summary)

A complete log is in `docs/PROCESS.md §10`. The notable ones:

1. **Padding side bug in GRU training** — `pack_padded_sequence` expects right-padding; the initial left-padded data silently packed the wrong timesteps, caught by a unit test and fixed.
2. **Candidate-list generation was O(users × catalogue)** and effectively hung; rewritten with oversampling + filtering (94,762 users × 2 splits in ~15 s).
3. **`run_csa` return-arity bug** (3 values unpacked as 2) crashed the first eval attempt at θ-tuning.
4. **OpenAI strict-schema constraint** — free-form `dict[str,str]` reasons are not allowed; output schema changed to a list of `{item_id, reason}`.
5. **Bootstrap memory blow-up** — a naive 10,000 × 94,762 resampling matrix (~7.5 GB) was OOM-killed; rewritten to chunked sampling with a 20,000-user cap (means still computed on all users).

---

## 12. Conclusion

We built, trained and evaluated an ARAG-inspired agentic recommender fused with a trained GRU. Under a deliberately ranking-isolating protocol (frozen uniform-negative lists):

- Fusing the trained GRU into ARAG yields a **large, significant improvement over ARAG alone (H1 supported)**.
- The fused system does **not significantly beat the GRU alone (H2 not supported)** at α\* = 0.1; the LLM contributes a small, non-significant increment.
- A **popularity baseline is nearly as strong as the GRU**, showing that the uniform-negative protocol itself is the dominant factor behind the LLM systems' weak showing.

The main scientific value of the result is the quantification itself: *on this task and protocol, the trained sequential model carries the ranking signal, and the agentic LLM layer adds little.* Future work: harder negatives (retrieval-based), a strictly time-separated split for the GRU, more test users for tighter CIs, and stronger LLM baselines to test whether the agentic increment survives with more statistical power.

---

## 13. Product demo (Phase B)

Beyond the research comparison, the project ships a working interactive demo that serves recommendations from the same components (ARAG agents, trained GRU, FAISS retrieval, fused at the evaluation's α\*).

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
configs/config.yaml      all paths, seeds, model names, θ/α grids
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
| Interactions / users / items | 814,586 / 94,762 / 137,269 |
| Frozen lists | 20 items (1 target + 19 negatives), seed 42, hashed |
| GRU vocab / config | 25,527 items; hidden 64, dropout 0.2; 3 seeds |
| GRU test (all users) | Hit@5 0.976 ± 0.001, NDCG@5 0.823 ± 0.004 |
| θ\* / α\* | 5 / 0.1 |
| H1 (ARAG+GRU − ARAG, NDCG@5) | **+0.349** [+0.257, +0.441] — significant |
| H2 (ARAG+GRU − GRU, NDCG@5) | +0.002 [−0.018, +0.022] — not significant |
| LLM calls / tokens / cost | 2,405 / 4.89M in + 0.42M out / **$0.9853** |
| Cached re-run | 136 s, zero new API calls |
| Audit | `make verify` → VERIFY OK |

## Appendix C — Anticipated questions and short answers

**Q1. Why frozen candidate lists instead of retrieving candidates with FAISS?**
Because the research question is about *ranking*, not retrieval. If each system retrieved its own candidates, a retrieval miss would be indistinguishable from a ranking miss, and methods would not be comparable. Freezing one 20-item list per user makes every comparison apples-to-apples. (Trade-off: uniform negatives are easier than real retrieval — listed as a limitation.)

**Q2. Why is the crude Popularity baseline nearly as good as the trained GRU?**
Under uniform random negatives, the target tends to be a popular item while the 19 decoys are mostly obscure. Ranking by global interaction count therefore scores high (NDCG@5 = 0.801 vs the GRU's 0.823). This is a property of the evaluation protocol, not a bug in the GRU; the GRU is still significantly better (Δ = −0.021, CI excludes 0 over 94,762 users), and on harder negatives the gap would widen.

**Q3. H2 says the fusion doesn't beat the GRU — is the project a failure?**
No. The project's value is *quantification*: H1 is strongly supported (the trained model adds a large, significant gain over the LLM), and H2 shows rigorously that, on this protocol, the LLM adds no measurable increment on top of the trained model. Reporting a null result with CIs is the correct scientific outcome — the alternative (claiming a gain that isn't there) would be fabrication.

**Q4. Was α\* = 0.1 chosen to make the fusion look like the GRU?**
No. α was selected **only on the 100 validation users**, never on test, by maximizing validation NDCG@5 across the full grid. Both the validation and test α-curves independently peak at 0.1. At α=0.1 the fusion is 90% GRU — that is the honest optimum found.

**Q5. Why gpt-4o-mini rather than the paper's gpt-3.5-turbo?**
The paper's exact model, prompts and data are not reproducible from its text. gpt-4o-mini is the modern successor at ~$0.15/1M tokens, making the full evaluation cost under $1. The code supports any provider (local qwen3:8b via Ollama is also wired in). We explicitly do **not** compare our numbers to the paper's.

**Q6. Why a GRU and not a transformer / LLM fine-tune?**
The spec asked for a compact, CPU-trainable sequential model. The GRU4Rec formulation is the standard, well-understood baseline for next-item prediction, trains in ~1.5 h on CPU here, and is exactly the "trained deep-learning component" the project needed. A transformer would add little on 25k items with ~6-item user sequences.

**Q7. How do we know the results are not fabricated?**
(a) `results/` is written only by pipeline code on real data; (b) `make verify` recomputes every metric from raw per-user predictions and diffs against `metrics.json` — VERIFY OK; (c) the frozen candidate lists are SHA-256-hashed and re-checked; (d) `run_record.json` records the git commit, model, config and dataset manifest; (e) the LLM cache makes the run idempotent — anyone can re-run and get identical numbers at $0.

**Q8. Why only 100 test users for the LLM systems?**
Cost and statistics: each LLM user costs ~7–12 API calls, and 100 paired users are enough to detect the paper's ~10–15 point effects. The CIs (±~10 points on Hit@5) are reported so small differences are not over-read. Non-LLM systems use all 94,762 test users, giving near-deterministic estimates.

**Q9. Why is ARAG worse than the baselines the paper claims it beats?**
Directionally the paper's ordering holds on our data (Recency 0.389 < ARAG 0.453; Vanilla-RAG 0.386 < ARAG), but the differences are not statistically significant at n=100, and the paper's ablations also don't reproduce (removing NLI didn't hurt). The likely cause is the protocol: with uniform negatives, global popularity dominates, and the agentic context machinery has little headroom — a finding we report rather than hide.

**Q10. What would you change with more time/budget?**
Harder negatives (retrieval-mined instead of uniform) to remove the popularity shortcut; ≥300 test users for tighter CIs; a strictly time-separated split for the GRU to remove leave-last-out leakage; and an LLM ranker that is explicitly given popularity/statistical features, to test whether the agentic contribution survives once the popularity confound is removed.

---

## Appendix D — One-paragraph summaries at three levels

**To a non-technical person:** We built a music-store-style recommender for video games that combines two very different "brains" — an AI assistant that reads what you bought and reasons about it, and a small trained neural network that has learned from a million past purchases what people typically buy next. We tested whether combining them beats each alone, by giving both the same 20 products to rank and checking how often the right product lands near the top. The trained network turned out to be far better than the AI assistant, and combining them was no better than the network alone — a clean, useful answer.

**To a computer-science student:** We implement an ARAG-style 4-agent LLM pipeline (user understanding → per-candidate NLI scoring → context summarisation → ranking) and fuse its permutation with a GRU4Rec-style next-item model via rank-derived score blending (`α·s_LLM + (1−α)·s_GRU`, α tuned on validation). On frozen 20-item lists (1 target + 19 uniform negatives) over Amazon Video_Games (94,762 users), the fused system significantly beats ARAG (+0.349 NDCG@5) but ties the GRU alone (+0.002, n.s.), with α\* = 0.1 — showing the trained model dominates under this protocol.

**To a recommender-systems researcher:** We quantify the marginal value of an agentic LLM ranker over a strong trained sequential baseline under a rank-isolating protocol. Contributions: (i) a reproducible ARAG+GRU fusion with α/θ tuned on validation and paired-bootstrap CIs; (ii) evidence that uniform-negative lists let a popularity baseline reach 0.80 NDCG@5 — nearly matching the GRU (0.82) and far exceeding every LLM variant (≤0.48) — which explains the null result for the LLM's increment; (iii) a fully audited artifact set (hashed frozen lists, per-user predictions, metric re-derivation, deterministic cache). Limitations: single category, n=100 for LLM systems, leave-last-out leakage into GRU training, no human evaluation.

