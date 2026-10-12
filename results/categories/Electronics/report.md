# ARAG + GRU — Results Report

**Model:** nvidia/nemotron-3-super-120b-a12b (nvidia)  
**Category:** Electronics  
**θ*:** 5  **α*:** 1.0  
**LLM cost:** $0.0000 for all calls in this evaluation (paid in the last run: $0.0000; the rest came from the cache)  
**Last run elapsed:** 251.2 min

## Main results (100 sampled test users, all systems)

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| random | 0.270 | 0.169 | 0.470 | 0.234 | 0.199 |
| popularity | 0.210 | 0.116 | 0.450 | 0.192 | 0.148 |
| content | 0.380 | 0.264 | 0.640 | 0.350 | 0.287 |
| itemknn | 0.180 | 0.114 | 0.240 | 0.134 | 0.147 |
| gru_seed0 | 0.220 | 0.120 | 0.430 | 0.186 | 0.149 |
| gru_text_seed0 | 0.290 | 0.189 | 0.530 | 0.265 | 0.215 |
| recency | 0.390 | 0.264 | 0.520 | 0.306 | 0.270 |
| vanilla_rag | 0.350 | 0.219 | 0.530 | 0.277 | 0.233 |
| arag | 0.350 | 0.259 | 0.640 | 0.353 | 0.292 |
| arag_wo_nli | 0.360 | 0.268 | 0.610 | 0.349 | 0.296 |
| arag_wo_nli_csa | 0.310 | 0.255 | 0.550 | 0.333 | 0.300 |
| arag_gru | 0.350 | 0.259 | 0.640 | 0.353 | 0.292 |
| arag_gru_text | 0.350 | 0.259 | 0.640 | 0.353 | 0.292 |
| arag_content | 0.420 | 0.274 | 0.650 | 0.348 | 0.280 |

## Non-LLM systems on all 100000 test users

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| random | 0.250 | 0.147 | 0.501 | 0.227 | 0.179 |
| popularity | 0.201 | 0.117 | 0.409 | 0.184 | 0.153 |
| content | 0.451 | 0.312 | 0.676 | 0.385 | 0.319 |
| itemknn | 0.186 | 0.132 | 0.278 | 0.161 | 0.168 |
| gru_seed0 | 0.216 | 0.127 | 0.432 | 0.196 | 0.162 |
| gru_text_seed0 | 0.350 | 0.223 | 0.577 | 0.296 | 0.238 |

## Conclusions (computed)

- ARAG+GRU vs ARAG (H1): Δ NDCG@5 = +0.000 (95% CI [+0.000, +0.000]) — not significant
- ARAG+GRU vs GRU (H2): Δ NDCG@5 = +0.139 (95% CI [+0.052, +0.229]) — significant
- LLM-Recency vs ARAG: Δ NDCG@5 = +0.005 (95% CI [-0.075, +0.081]) — not significant
- LLM-Vanilla-RAG vs ARAG: Δ NDCG@5 = -0.039 (95% CI [-0.124, +0.042]) — not significant
- ARAG w/o NLI vs ARAG: Δ NDCG@5 = +0.009 (95% CI [-0.034, +0.054]) — not significant
- ARAG w/o NLI&CSA vs ARAG: Δ NDCG@5 = -0.004 (95% CI [-0.048, +0.042]) — not significant
- Content similarity vs ARAG: Δ NDCG@5 = +0.006 (95% CI [-0.075, +0.084]) — not significant
- ItemKNN vs ARAG: Δ NDCG@5 = -0.144 (95% CI [-0.233, -0.057]) — significant
- Text-aware GRU vs ID-only GRU: Δ NDCG@5 = +0.069 (95% CI [+0.027, +0.117]) — significant
- Text-aware GRU vs ARAG: Δ NDCG@5 = -0.070 (95% CI [-0.168, +0.027]) — not significant
- ARAG+text-GRU vs ARAG (H1, text GRU): Δ NDCG@5 = +0.000 (95% CI [+0.000, +0.000]) — not significant
- ARAG+text-GRU vs text-GRU (H2, text GRU): Δ NDCG@5 = +0.070 (95% CI [-0.027, +0.168]) — not significant
- ARAG+content vs ARAG: Δ NDCG@5 = +0.015 (95% CI [-0.042, +0.069]) — not significant

## Limitations

- Leave-last-out means other users' later interactions can appear in GRU training.
- 100 test users ⇒ wide CIs (roughly ±10 points on Hit@5).
- Sampled negatives are easier than real retrieval candidates; not an exact paper reproduction.
- OpenAI gpt-4o-mini was used; results are ARAG-inspired, not a paper reproduction.

## Protocol

Frozen 20-item candidate lists (target + 19 negatives) shared by all systems. 
Negatives: sampled from items with train interactions, proportional to train interaction count, excluding anything the user interacted with. 
θ tuned on validation NDCG@5 over {4,5,6,7}; α tuned on validation NDCG@5 over [0,1]. 
The main table and all significance tests use the same sampled test users for every system; non-LLM systems are additionally reported on all test users. 
Paired-bootstrap 95% CIs (10,000 resamples).
