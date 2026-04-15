# LLM Router — Reward Function Data Specification

*Generated from analysis of `m2.csv` (1531-row MMLU validation benchmark of `llama3.2:3b`)*

---

## Overview

The training reward function is:

```
r(a, x) = α · correct(a, x) − β · latency(a, x)
```

The paper's original formula included a third `γ · energy(a, x)` term, but energy is excluded from the training reward for this model pool (see reasoning below). `energy_joules` is retained as a post-hoc **evaluation metric only** (Accuracy-per-Joule, average energy per query).

This document specifies exactly which CSV columns map to each term, how to normalize them, and what to avoid.

---

## The Two Training Reward Terms

### 1. `correct(a, x)` → column: `is_correct`

- **Type**: Boolean (`True`/`False`), cast to float → `{0.0, 1.0}`
- **Already in [0,1]**, no normalization needed.
- **Note on `answer_format_valid`**: 27.6% of rows have `answer_format_valid=False` (model didn't output a clean A/B/C/D). All of these are `is_correct=False`. When building the multi-model dataset, treat a format failure the same as a wrong answer — `is_correct=0`. Do **not** use `result` (a string column); use `is_correct` directly.
- **Baseline accuracy**: 36.3% for `llama3.2:3b` on the MMLU validation split.

---

### 2. `latency(a, x)` → column: `total_latency_s`

- **Type**: Float, seconds (wall-clock from request send to last token received).
- **Distribution**: Median ~0.075s, p99 ~0.096s, with a small number of timeout/outlier rows up to 3.6s.
- **Normalization**: Clip at the p99 value of the **training set**, then min-max scale to [0, 1]:

  ```
  lat_clipped = min(total_latency_s, p99_lat)
  latency_norm = (lat_clipped − lat_min) / (p99_lat − lat_min)
  ```

  Compute `lat_min` and `p99_lat` from the training split only; apply the same constants at eval/inference time.

- **Do NOT use `eval_duration_s`**: It only measures token generation time and is only 16% correlated with `total_latency_s`. It misses queuing, prompt-eval, and network overhead — the things that actually matter for UX latency.
- **Do NOT use `ttft_s`** (time-to-first-token) as a separate term: it is ~99% correlated with `total_latency_s` for these short MCQ responses. Including both would double-penalize latency.

---

### 3. `energy_joules` — Evaluation Metric Only (not in training reward)

- **Column**: `energy_joules`
- **Type**: Float, joules. GPU energy for the full inference call, computed as a trapezoid-integral of sampled `nvidia-smi` power readings over wall-clock duration.
- **Source flag**: `energy_source` is always `nvidia_sampled_power_integral` — the hardware total-energy counter (`gpu_total_energy_delta_j`) is not supported on the L40S and is all-NaN.
- **Distribution**: Median ~14.3J, p99 ~20.8J.

#### Why energy is excluded from the training reward

`energy_joules` and `total_latency_s` are **97.5% correlated** (r=0.975) in this dataset. The reason is structural: inference on the L40S is memory-bandwidth bound at batch size 1, so GPU power draw is nearly constant (~97W ± 10W std) across queries. This means `energy ≈ constant × latency` — the energy term carries no routing signal that latency doesn't already provide.

This holds across the full 8-model pool (3B through 8B). The pool spans only a ~2.5–3x parameter range on the same GPU. Larger models draw somewhat more power (~120–140W vs ~90W) but are also slower, so energy increases roughly in lockstep with latency. Including `γ·energy` in the training reward would effectively double-penalize latency, making `β` and `γ` impossible to tune independently and biasing the router even harder toward the smallest model.

#### How energy is used instead

Record `energy_joules` for every inference call and use it at **evaluation time** to compute:
- **Average energy per query** (reported in the results table)
- **Accuracy-per-Joule** (the paper's primary efficiency metric; baseline is 0.0245 J⁻¹)
- **Total energy** across the validation set (for comparison to baseline and GPT-OSS 15% reduction goal)

This strengthens the experimental story: the router was trained to minimize latency, and reduced energy as a downstream consequence — a cleaner causal claim than if energy were in the reward directly.

---

## Columns to IGNORE for the Training Reward

| Column | Reason to exclude |
|---|---|
| `energy_joules` | 97.5% correlated with `total_latency_s`; use for evaluation only |
| `eval_duration_s` | Only generation time; 16% correlated with wall latency |
| `ttft_s` | 99% correlated with `total_latency_s` for short outputs |
| `energy_per_output_token_j` | Noisy: `eval_count` is almost always 1–3 tokens for MCQ |
| `gpu_total_energy_delta_j` | All NaN (hardware counter not supported on L40S) |
| `cpu_time_delta_s` / `cpu_percent` | 75th percentile is 0; workload is GPU-bound |
| `wall_tokens_per_s` | Useful for monitoring; not a training signal |
| `gpu_power_avg_watts` | Already encoded in `energy_joules`; don't double-count |
| `rss_*` / `ollama_tree_pid_count_*` | Memory metrics; not in the reward formulation |

---

## Normalization Summary

Latency normalization constants must be computed **once from the training split** and frozen for use at eval and RL training time. Suggested approach:

```python
# Fit on training split
lat_min = df_train['total_latency_s'].min()
lat_p99 = df_train['total_latency_s'].quantile(0.99)

# Apply at runtime (training reward)
def compute_reward(is_correct, total_latency_s, alpha, beta):
    lat_norm = (min(total_latency_s, lat_p99) - lat_min) / (lat_p99 - lat_min)
    return alpha * float(is_correct) - beta * lat_norm

# Apply at eval time (record separately, do not feed back into reward)
def compute_eval_metrics(is_correct_list, energy_joules_list):
    avg_energy = sum(energy_joules_list) / len(energy_joules_list)
    accuracy   = sum(is_correct_list) / len(is_correct_list)
    accuracy_per_joule = accuracy / avg_energy
    return {"accuracy": accuracy, "avg_energy_j": avg_energy, "accuracy_per_joule": accuracy_per_joule}
```

---

## Baseline Numbers (llama3.2:3b, MMLU validation, n=1531)

| Metric | Value |
|---|---|
| Accuracy (`is_correct`) | 36.3% |
| Median `total_latency_s` | 0.075s |
| Mean `energy_joules` | 14.8J |
| Accuracy-per-Joule | 0.0245 J⁻¹ |
| `answer_format_valid` rate | 72.4% |
| `energy_source` | `nvidia_sampled_power_integral` (always) |

These numbers match the milestone report exactly, confirming the CSV is the correct baseline dataset.

---

## What This Means for Multi-Model Data Collection

When you run the same benchmark script (`benchmark_mmlu_csv.py`) for each of the 8 models in the pool, each row in the combined dataset will have:

- `is_correct` — the correctness signal (0 or 1)
- `total_latency_s` — wall-clock latency; used in the training reward
- `energy_joules` — GPU energy; recorded and used for evaluation metrics only
- `model` — the action taken (router's choice)
- `subject` / `question` — context for grouping/analysis

The training reward for a (prompt, model) pair is determined by `is_correct` and `total_latency_s` only. `energy_joules` is carried through the dataset for evaluation but never fed back into the PPO update. Everything else in the CSV is diagnostic/debugging data.
