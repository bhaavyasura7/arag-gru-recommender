# Non-LLM results — Electronics

## All 100,000 test users

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| Random | 0.250 | 0.147 | 0.501 | 0.227 | 0.179 |
| Popularity | 0.201 | 0.117 | 0.409 | 0.184 | 0.153 |
| Content similarity | 0.451 | 0.312 | 0.676 | 0.385 | 0.319 |
| ItemKNN | 0.186 | 0.132 | 0.278 | 0.161 | 0.168 |
| GRU, ID only (seed 0) | 0.216 | 0.127 | 0.432 | 0.196 | 0.162 |
| GRU, ID only (seed 1) | 0.224 | 0.133 | 0.439 | 0.202 | 0.167 |
| GRU, ID only (seed 2) | 0.221 | 0.131 | 0.439 | 0.201 | 0.165 |
| Text-aware GRU (seed 0) | 0.350 | 0.223 | 0.577 | 0.296 | 0.238 |
| Text-aware GRU (seed 1) | 0.349 | 0.222 | 0.572 | 0.293 | 0.237 |
| Text-aware GRU (seed 2) | 0.341 | 0.215 | 0.562 | 0.286 | 0.231 |

### Paired bootstrap (ΔNDCG@5)

| Contrast | ΔNDCG@5 | 95% CI | Significant |
|---|---|---|---|
| Random − GRU, ID only (seed 0) | +0.020 | [+0.015, +0.026] | yes |
| Random − Text-aware GRU (seed 0) | -0.076 | [-0.082, -0.070] | yes |
| Popularity − GRU, ID only (seed 0) | -0.010 | [-0.011, -0.008] | yes |
| Popularity − Text-aware GRU (seed 0) | -0.105 | [-0.110, -0.102] | yes |
| Content similarity − GRU, ID only (seed 0) | +0.185 | [+0.174, +0.187] | yes |
| Content similarity − Text-aware GRU (seed 0) | +0.090 | [+0.078, +0.090] | yes |
| ItemKNN − GRU, ID only (seed 0) | +0.005 | [+0.003, +0.011] | yes |
| ItemKNN − Text-aware GRU (seed 0) | -0.091 | [-0.094, -0.084] | yes |
| GRU, ID only (seed 1) − GRU, ID only (seed 0) | +0.006 | [+0.003, +0.008] | yes |
| GRU, ID only (seed 1) − Text-aware GRU (seed 0) | -0.090 | [-0.095, -0.087] | yes |
| GRU, ID only (seed 2) − GRU, ID only (seed 0) | +0.004 | [+0.003, +0.007] | yes |
| GRU, ID only (seed 2) − Text-aware GRU (seed 0) | -0.092 | [-0.095, -0.088] | yes |
| Text-aware GRU (seed 0) − GRU, ID only (seed 0) | +0.095 | [+0.093, +0.100] | yes |
| Text-aware GRU (seed 1) − GRU, ID only (seed 0) | +0.095 | [+0.092, +0.099] | yes |
| Text-aware GRU (seed 1) − Text-aware GRU (seed 0) | -0.001 | [-0.004, +0.002] | no |
| Text-aware GRU (seed 2) − GRU, ID only (seed 0) | +0.088 | [+0.085, +0.091] | yes |
| Text-aware GRU (seed 2) − Text-aware GRU (seed 0) | -0.008 | [-0.012, -0.005] | yes |

## The 100 sampled test users (same users as the LLM systems)

| System | hit@5 | ndcg@5 | hit@10 | ndcg@10 | mrr |
|---|---|---|---|---|---|
| Random | 0.270 | 0.169 | 0.470 | 0.234 | 0.199 |
| Popularity | 0.210 | 0.116 | 0.450 | 0.192 | 0.148 |
| Content similarity | 0.380 | 0.264 | 0.640 | 0.350 | 0.287 |
| ItemKNN | 0.180 | 0.114 | 0.240 | 0.134 | 0.147 |
| GRU, ID only (seed 0) | 0.220 | 0.120 | 0.430 | 0.186 | 0.149 |
| GRU, ID only (seed 1) | 0.250 | 0.146 | 0.460 | 0.215 | 0.175 |
| GRU, ID only (seed 2) | 0.220 | 0.127 | 0.440 | 0.195 | 0.158 |
| Text-aware GRU (seed 0) | 0.290 | 0.189 | 0.530 | 0.265 | 0.215 |
| Text-aware GRU (seed 1) | 0.340 | 0.220 | 0.530 | 0.278 | 0.232 |
| Text-aware GRU (seed 2) | 0.330 | 0.207 | 0.550 | 0.277 | 0.222 |

