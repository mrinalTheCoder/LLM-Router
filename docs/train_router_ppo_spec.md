# `train_router_ppo.py` Specification

## 1. Purpose
`scripts/train_router_ppo.py` trains an **online PPO routing policy** that selects one model from a candidate pool for each MMLU prompt.

Each policy action triggers a **live model query** and computes reward from:

`reward = alpha * correct - beta * normalized_latency - gamma * normalized_energy`

The script logs full training/evaluation metrics to Weights & Biases (W&B), including telemetry fields collected from online inference.

---

## 2. High-level architecture
1. **Prompt/data layer**
   - Loads MMLU prompts from HuggingFace parquet via Polars.
   - Builds prompt records (`question`, 4 choices, truth label).

2. **State encoder**
   - Uses a shared, frozen SentenceTransformers encoder (default: `all-MiniLM-L6-v2`) to generate a 384-d router observation vector for each prompt.

3. **Outcome provider abstraction**
   - `OutcomeProvider` protocol defines `query(prompt, model_name) -> ModelOutcome`.
   - `OnlineOllamaProvider` is the current implementation (live Ollama query + telemetry).
   - Future cache/database providers can implement the same protocol with no trainer-loop changes.

4. **Bandit environment**
   - `RouterBanditEnv` is a single-step contextual bandit:
     - Observation: prompt embedding
     - Action: selected model index
     - Transition: query selected model
     - Reward: weighted correctness/latency/energy objective

5. **Trainer**
   - Uses Stable-Baselines3 PPO (`MlpPolicy`) over the bandit environment.
   - The PPO policy consumes the 384-d MiniLM embedding directly and uses split heads:
     - Actor MLP: `384 -> 256 -> 128 -> |model_pool|`
     - Critic MLP: `384 -> 256 -> 64 -> 1`
   - With the default 8-model pool, the actor head outputs 8 action logits.
   - Supports checkpointing and periodic deterministic validation.

6. **Logging and artifacts**
   - W&B callback logs training metrics, eval metrics, per-model selection rates, and best/final model artifacts.

---

## 3. Configuration model
Configuration supports:
1. CLI flags
2. Optional YAML file (`--config-yaml`)

### 3.1 Precedence
When YAML is provided, values are loaded as parser defaults, then CLI is parsed.

**Final precedence:** `CLI args > YAML values > built-in defaults`

### 3.2 YAML key rules
- Keys must match argument destination names (e.g., `learning_rate`, `reward_alpha`).
- Nested mappings are allowed; nested keys are flattened with `_`.
  - Example: `reward: { alpha: 1.0 }` maps to `reward_alpha`.
- Unknown keys raise a config error.

---

## 4. Key runtime parameters

### 4.1 Data
- `dataset`, `config`
- `train_split`, `val_split`
- `train_limit`, `val_limit`

### 4.2 Model routing
- `model_pool` (comma string or YAML list)
- `host`, `timeout`, `gpu_layers`, `gpu_device`, `power_sample_interval`

### 4.3 Reward shaping
- `reward_alpha` (correctness weight)
- `reward_beta` (latency penalty weight)
- `reward_gamma` (energy penalty weight)
- `latency_norm_s`, `energy_norm_j` (normalization denominators)

### 4.4 PPO
- `learning_rate`, `total_timesteps`
- `n_steps`, `batch_size`, `ppo_epochs`
- `gamma_rl`, `gae_lambda`, `clip_range`
- `ent_coef`, `vf_coef`, `max_grad_norm`
- `actor_hidden_sizes`, `critic_hidden_sizes`, `n_envs`, `ppo_device`

### 4.5 Encoder
- `encoder_name`, `encoder_device`
- `embedding_batch_size`
- `embedding_normalize` / `no_embedding_normalize`

### 4.6 Logging and output
- `eval_every_steps`, `log_every_steps`, `checkpoint_metric`
- `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_mode`
- `seed`, `output_dir`

---

## 5. Metrics and logging

### 5.1 Training logs (W&B)
- Reward decomposition:
  - `train/reward_total`
  - `train/reward_correct`
  - `train/reward_latency_penalty`
  - `train/reward_energy_penalty`
- Quality/efficiency:
  - `train/correct`, `train/latency_s`, `train/energy_joules`
  - `train/ttft_s`, token throughput metrics
- System telemetry:
  - CPU effort/percent, RSS delta
  - GPU utilization/VRAM/power stats
- Routing behavior:
  - `train/model_selection_rate/<model>`
- PPO internals:
  - `sb3/*` optimization metrics exposed by SB3 logger

### 5.2 Evaluation logs (W&B)
- `eval/accuracy`, `eval/mean_reward`
- `eval/avg_latency_s`, `eval/avg_energy_joules`
- `eval/accuracy_per_joule`
- Per-model selection and selected-model accuracy

---

## 6. Outputs
In `output_dir`:
- `router_policy_final.zip` (final SB3 model)
- `run_config.json` (resolved runtime config)
- `run_summary.json` (final run summary + final eval when available)

When W&B is enabled:
- Final and best-model artifacts are uploaded.

---

## 7. YAML example
```yaml
dataset: cais/mmlu
config: all
train_split: test
val_split: validation
train_limit: 512
val_limit: 128

model_pool:
  - llama3.2:3b
  - qwen2.5:3b
  - granite3.3:8b

reward:
  alpha: 1.0
  beta: 0.2
  gamma: 0.2
latency_norm_s: 3.0
energy_norm_j: 200.0

learning_rate: 0.0003
total_timesteps: 5000
n_steps: 256
batch_size: 64
ppo_epochs: 10
actor_hidden_sizes: [256, 128]
critic_hidden_sizes: [256, 64]

encoder_name: sentence-transformers/all-MiniLM-L6-v2
encoder_device: cpu
embedding_batch_size: 64
embedding_normalize: false

eval_every_steps: 1000
log_every_steps: 100
checkpoint_metric: eval/accuracy_per_joule

wandb_project: llm-router
wandb_mode: offline
seed: 42
output_dir: outputs/router_ppo
```

Run:
```bash
python scripts/train_router_ppo.py --config-yaml path/to/router_train.yaml
```

Override from CLI:
```bash
python scripts/train_router_ppo.py --config-yaml path/to/router_train.yaml --learning-rate 1e-4 --reward-beta 0.3
```
