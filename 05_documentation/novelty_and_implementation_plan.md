# Reasoning Boundary Paradox in Small Language Models
## Consolidated Novelty Statement & Full Implementation Plan

---

## 0. One-paragraph pitch (use this in your abstract / intro)

> RLVR's boundary-shrinkage failure (Pass@k collapse via winner-take-all exploitation) has been studied almost exclusively on 7B+ models with multi-GPU clusters. We present the first controlled study of how this failure scales with *trainable capacity* under a fixed single-GPU (12GB) budget, show that existing accuracy-gated mitigations (SELF-style data curation, entropy control, adaptive rollout allocation) lose effectiveness as capacity shrinks, and introduce **Capacity-Budgeted GRPO (CB-GRPO)**, a lightweight gradient-mass gating mechanism that targets *capacity consumption* directly rather than *correctness*, and which outperforms accuracy-gated baselines specifically in the low-capacity regime.

Keep every section of the paper traceable back to this sentence. If an experiment doesn't support one of its three clauses, it's scope creep — cut it or move it to future work.

---

## 1. The Three Novelty Pillars (final, locked-in scope)

| # | Pillar | Type of contribution | Why it's defensible |
|---|--------|----------------------|----------------------|
| **N1** | **Capacity-Scaling Study of the Reasoning Boundary Paradox** | Empirical / scaling-law | No prior work (SELF, Yue et al., DeepSeek-AI, PSN-GRPO, PopuLoRA, EvoPref, HORA/AERO/etc.) measures boundary-shrinkage severity as a controlled function of trainable parameter budget at ≤3B scale on consumer hardware. |
| **N2** | **Mitigation Transferability Stress-Test** | Empirical / negative-or-positive result | Every mitigation we found (data curation, entropy regularization, adaptive rollout allocation, population-LoRA) was validated at 7B–14B+ on clusters. Whether they *still work* at 0.5B–1.5B under a 12GB budget is an open, checkable question. |
| **N3** | **Capacity-Budgeted GRPO (CB-GRPO)** | Algorithmic | A new gating signal — cumulative gradient-mass spent per prompt cluster — as opposed to every existing gate (SELF, O-SELF, curriculum methods) which gates on *accuracy/solve-rate*. This is a genuinely different mechanism, not a re-parameterization of an existing one. |

Everything else from your original plan (O-SELF, D-GRPO diversity penalty, the quantized/QLoRA/GRPO stack, zero-VRAM heuristic rewards) is **retained but reclassified** as either a *baseline you compare against* (O-SELF) or *supporting infrastructure* (QLoRA stack, heuristic rewards) — not headline novelty. Say this explicitly in your related-work section; it's what makes N1–N3 credible.

---

## 2. System Architecture Overview

```
                        ┌─────────────────────────────┐
                        │   Qwen2.5-{0.5B,1.5B,3B}      │
                        │   Instruct, NF4 4-bit frozen  │
                        └───────────────┬───────────────┘
                                        │
                          QLoRA (r=16, α=32, attn proj)
                                        │
                    ┌───────────────────┴────────────────────┐
                    │        GRPO Rollout Engine (G=4)         │
                    │  vLLM or HF generate, temp=0.8-1.0       │
                    └───────────────────┬────────────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
        ┌────────▼────────┐   ┌─────────▼─────────┐  ┌─────────▼─────────┐
        │ Format Reward     │   │ Correctness Reward │  │ Prompt Cluster ID │
        │ (regex, CPU)      │   │ (regex, CPU)        │  │ (precomputed)     │
        └────────┬────────┘   └─────────┬─────────┘  └─────────┬─────────┘
                 └──────────────────────┼──────────────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │   Pluggable GATE MODULE     │
                          │  vanilla | SELF | O-SELF |   │
                          │  adaptive-rollout | CB-GRPO  │  ← swap this to run
                          └─────────────┬─────────────┘     every experiment
                                        │
                          ┌─────────────▼─────────────┐
                          │  GRPO Advantage + Update    │
                          └─────────────┬─────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │   Eval loop: unbiased Pass@k estimator  │
                    │   Streamlit dashboard: entropy, Pass@k, │
                    │   capacity-spend histogram              │
                    └─────────────────────────────────────────┘
```

The single most important engineering decision: **build the "gate" as one pluggable interface from day one.** Every baseline (vanilla, SELF, O-SELF, adaptive-rollout, CB-GRPO) is just a different implementation of the same function signature. This is what makes N2 (the stress-test) cheap to run once N1's harness exists, and it's a clean thing to show a reviewer — "same trainer, five interchangeable gates."

```python
class Gate(ABC):
    @abstractmethod
    def weight(self, batch: RolloutBatch, state: TrainerState) -> torch.Tensor:
        """Return a per-sample multiplier in [0,1] applied to the GRPO
        advantage before the policy-gradient loss. Vanilla GRPO returns
        all-ones."""
```

---

## 3. Pillar N1 — Capacity-Scaling Study: Implementation

### 3.1 Fixed experimental variables (hold these constant across all model sizes)

- Dataset: GSM8K train split for RL; GSM8K test + a MATH subset (algebra/prealgebra, harder tier) for OOD Pass@k eval
- QLoRA: r=16, α=32, attention projections only
- GRPO group size G=4
- Optimizer: paged 8-bit AdamW, lr swept once per size (don't hand-tune per size beyond a small grid — you want capacity to be the explanatory variable, not per-run tuning)
- Same number of gradient steps and same effective batch size at every size
- `max_new_tokens` capped at 512–640 for all rollouts

### 3.2 Model matrix

| Model | Trainable LoRA params | Expected VRAM (rollout+train) | Priority |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | ~2.2M | Low — use for fast iteration | Core |
| Qwen2.5-1.5B-Instruct | ~8M | Fits comfortably in 12GB | Core |
| Qwen2.5-3B-Instruct | ~16M | Tight — may need micro-batch=1, grad accumulation | Stretch |

If 3B doesn't fit with G=4 rollouts, drop to G=2 for that row *and report this as a finding* ("hardware forces a capacity/exploration-width trade-off at 3B") rather than silently changing the protocol without flagging it.

### 3.3 Metrics & the actual scaling-law computation

**Unbiased Pass@k estimator** (you already have this — implement once, reuse everywhere):

```python
from math import comb

def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """n = total rollouts sampled for a prompt, c = number correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

Sample n=64 rollouts per held-out prompt at eval checkpoints (start, mid, end of training) at temperature ~1.0. Compute Pass@k for k ∈ {1,4,16,64}.

**Boundary shrinkage slope** — the headline number for N1:

```python
import numpy as np

def shrinkage_slope(pass_at_k_base: dict[int,float],
                     pass_at_k_rl: dict[int,float]) -> float:
    ks = sorted(pass_at_k_base.keys())
    delta = np.array([pass_at_k_rl[k] - pass_at_k_base[k] for k in ks])
    log_k = np.log(np.array(ks))
    slope, _ = np.polyfit(log_k, delta, 1)
    return slope   # negative slope = boundary shrinks faster at high k
```

Run this per model size → plot `slope vs. trainable_param_count` (log-x). **This single figure is your N1 result.** If the slope becomes more negative as parameter count drops, you've empirically confirmed your original hypothesis with real numbers — that's a strong, simple, citable figure for an IEEE paper.

### 3.4 Deliverable for N1

- Figure: Pass@k curves, 3 panels (one per model size), base vs RL-tuned
- Figure: shrinkage slope vs. LoRA trainable-parameter count
- Table: raw Pass@1/4/16/64 numbers per size, per checkpoint

---

## 4. Pillar N2 — Mitigation Transferability Stress-Test: Implementation

### 4.1 Baselines to implement as `Gate` subclasses

```python
class VanillaGate(Gate):
    def weight(self, batch, state):
        return torch.ones(len(batch))

class StaticSELFGate(Gate):
    """Nguyen et al. 2025: precompute solve-rate on a frozen base-model
    pass, filter out prompts above a solvability threshold. Static —
    computed once before training starts."""
    def __init__(self, solved_prompt_ids: set, threshold=0.7):
        self.excluded = solved_prompt_ids
    def weight(self, batch, state):
        return torch.tensor([0.0 if pid in self.excluded else 1.0
                              for pid in batch.prompt_ids])

class OSELFGate(Gate):
    """Your original online EMA solve-rate gate — kept as a baseline,
    not the headline contribution."""
    def __init__(self, alpha=0.1, threshold=0.9):
        self.ema_solve_rate = defaultdict(lambda: 0.0)
        self.alpha, self.threshold = alpha, threshold
    def weight(self, batch, state):
        w = []
        for pid, correct in zip(batch.prompt_ids, batch.correctness):
            prev = self.ema_solve_rate[pid]
            self.ema_solve_rate[pid] = self.alpha*correct + (1-self.alpha)*prev
            w.append(0.0 if self.ema_solve_rate[pid] > self.threshold else 1.0)
        return torch.tensor(w)

class AdaptiveRolloutGate(Gate):
    """Simplified HORA-style: reweight by posterior probability that
    additional rollouts on this prompt would be informative (non-saturated
    group). Implement the light version — full posterior tracking is
    optional; a simple running-variance proxy is enough for a fair
    comparison, cite HORA/AERO for the full method."""
    def __init__(self, window=20):
        self.reward_history = defaultdict(lambda: deque(maxlen=window))
    def weight(self, batch, state):
        w = []
        for pid, r in zip(batch.prompt_ids, batch.group_rewards):
            hist = self.reward_history[pid]
            hist.append(r)
            variance = np.var(hist) if len(hist) > 1 else 1.0
            w.append(min(1.0, variance * 4))  # low-variance (saturated) → down-weighted
        return torch.tensor(w)
```

### 4.2 Stress-test protocol

Run **each gate × each model size** (0.5B, 1.5B, [3B]) with everything else fixed. That's a 4×2 (or 4×3) grid — 8–12 runs total for N2, reusing N1's harness. For each cell record:
- Final Pass@1 and Pass@64
- Shrinkage slope (same formula as N1)
- Wall-clock / rollout-count cost (some gates, like adaptive-rollout, claim efficiency gains — verify whether that claim survives at small scale)

### 4.3 What "transferability failure" looks like (so you know it when you see it)

A mitigation "transfers" if its shrinkage-slope improvement over vanilla GRPO (Δslope) stays roughly constant across model sizes. It "fails to transfer" if Δslope shrinks toward zero (or reverses) as the model gets smaller. Plot Δslope vs. parameter count, one line per gate — this is your second headline figure, and it directly sets up why CB-GRPO is needed.

### 4.4 Deliverable for N2

- Figure: Δslope (vs. vanilla) per gate, per model size — the "does it transfer" plot
- Table: full metric grid (gate × size)

---

## 5. Pillar N3 — CB-GRPO: Full Implementation Detail

### 5.1 Motivation, stated precisely

Accuracy-based gates (SELF, O-SELF) ask *"has this prompt been solved?"* CB-GRPO asks a different question: *"how much of the model's scarce trainable capacity has already been spent reinforcing this prompt's solution pattern?"* These can diverge: a prompt can be solved cheaply (one clean, low-magnitude gradient update) or solved while consuming a disproportionate share of the LoRA's limited representational budget across many updates. In a parameter-starved model, the second case is the real driver of negative interference — and accuracy gates can't distinguish it from the first.

### 5.2 Step 1 — Prompt clustering (done once, before training)

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def build_prompt_clusters(prompts: list[str], n_clusters=16) -> dict[str,int]:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # tiny, CPU-friendly
    embeddings = embedder.encode(prompts, show_progress_bar=True)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(embeddings)
    return {p: int(c) for p, c in zip(prompts, km.labels_)}
```

Run this once on your full GSM8K training pool. `n_clusters` is a hyperparameter — start at 16, ablate {8, 16, 32} later (Section 5.6).

### 5.3 Step 2 — Gradient-mass tracking

Inside the GRPO loss, you already compute per-rollout advantages `A_i`. The policy-gradient contribution's magnitude scales with `|A_i|` — use this as your free "spend" signal, no extra backward pass required.

```python
class CBGRPOGate(Gate):
    def __init__(self, cluster_map: dict[str,int], n_clusters: int,
                 theta: float = 1.5, decay: float = 0.98, ema_alpha: float = 0.05):
        self.cluster_map = cluster_map
        self.spend = np.zeros(n_clusters)          # cumulative gradient-mass per cluster
        self.spend_ema = np.zeros(n_clusters)       # smoothed running spend
        self.theta = theta        # how far above mean spend before down-weighting kicks in
        self.decay = decay        # multiplicative decay applied to over-budget clusters
        self.ema_alpha = ema_alpha

    def weight(self, batch, state):
        w = []
        for pid, adv in zip(batch.prompt_ids, batch.advantages):
            c = self.cluster_map[pid]

            # 1. update running spend for this cluster
            grad_mass = float(torch.abs(adv))
            self.spend[c] += grad_mass
            self.spend_ema[c] = (self.ema_alpha * grad_mass
                                  + (1 - self.ema_alpha) * self.spend_ema[c])

            # 2. compare to population mean (cold-start safe: mean of zeros -> no gating early on)
            mean_spend = self.spend_ema.mean() + 1e-8
            ratio = self.spend_ema[c] / mean_spend

            # 3. soft down-weight, not hard zero — avoids starving a cluster forever
            if ratio > self.theta:
                over = ratio - self.theta
                mult = self.decay ** over          # smooth exponential decay past threshold
            else:
                mult = 1.0
            w.append(mult)
        return torch.tensor(w)
```

Design choices worth stating explicitly in the paper (reviewers will ask about these):

- **Soft decay, not hard zero-out** — unlike O-SELF's binary cutoff, CB-GRPO decays smoothly, so a cluster is never permanently excluded, only throttled. This avoids a failure mode where a whole reasoning-strategy family gets locked out.
- **Cluster-level, not prompt-level** — spend is tracked per cluster (group of similar prompts), not per individual prompt, because the underlying claim is about *reasoning-pattern capacity*, not memorization of individual problems. This is also more sample-efficient to estimate with limited training data.
- **Cold start is safe by construction** — with all spend at zero, `ratio` starts near 1.0 for every cluster, so CB-GRPO behaves like vanilla GRPO for the first several hundred steps until spend differentiation emerges.

### 5.4 Step 3 — Wiring into the trainer

If you're building on TRL's `GRPOTrainer`, the cleanest integration point is a custom reward/advantage post-processing hook — multiply the computed advantage tensor by `gate.weight(...)` immediately before the loss is formed, and nowhere else (don't touch the reward function itself — keep the CPU regex rewards exactly as planned; the gate only reweights the *gradient*, not the *reward signal* used for logging/eval).

```python
def training_step_hook(advantages: torch.Tensor, batch: RolloutBatch, gate: Gate, state: TrainerState):
    gated_advantages = advantages * gate.weight(batch, state).to(advantages.device)
    return gated_advantages  # feed this into the standard GRPO policy loss
```

### 5.5 Logging (feeds directly into your Streamlit dashboard)

Log per step: `spend_ema` histogram across clusters (this is your new diagnostic plot — nobody else in the related work has this specific view), plus the standard entropy and Pass@k trajectory.

### 5.6 Ablations to run (expected reviewer questions — pre-empt them)

| Ablation | What it tests |
|---|---|
| `n_clusters` ∈ {8, 16, 32} | Sensitivity to clustering granularity |
| `theta` ∈ {1.2, 1.5, 2.0} | How aggressive the gating threshold is |
| `decay` ∈ {0.9, 0.95, 0.98} | Soft vs. near-hard cutoff behavior |
| CB-GRPO vs. O-SELF, matched compute | Direct head-to-head — this is the money comparison |
| Cluster-level vs. prompt-level spend tracking | Justifies the cluster-level design choice |

### 5.7 Deliverable for N3

- Algorithm box (formal pseudocode) for the paper's methods section
- Figure: spend-distribution histogram, vanilla GRPO vs CB-GRPO (shows the "flattening" effect directly)
- Table: CB-GRPO vs. all N2 baselines, at 0.5B and 1.5B, on the shrinkage-slope metric
- Ablation table (Section 5.6 grid)

---

## 6. Supporting Infrastructure (not novelty — build these first, they're prerequisites)

1. **Quantized PEFT stack**: NF4 4-bit backbone via `bitsandbytes`, QLoRA via `peft`, paged 8-bit AdamW.
2. **Zero-VRAM heuristic rewards**: CPU regex checkers for `<think>...</think>` structural format and `\boxed{}` correctness extraction. Keep these deterministic and unit-tested — a silent regex bug here corrupts every downstream result.
3. **Streamlit diagnostic dashboard**: live-ingest training logs (JSONL is fine), render Pass@k trajectory curves, entropy over time, and (new) the CB-GRPO spend histogram.

---

## 7. Repository Structure

```
slm-boundary-paradox/
├── configs/
│   ├── model_0.5b.yaml
│   ├── model_1.5b.yaml
│   └── model_3b.yaml            # stretch
├── gates/
│   ├── base.py                  # Gate ABC
│   ├── vanilla.py
│   ├── static_self.py
│   ├── o_self.py
│   ├── adaptive_rollout.py
│   └── cb_grpo.py
├── rewards/
│   ├── format_reward.py
│   └── correctness_reward.py
├── clustering/
│   └── build_clusters.py
├── eval/
│   ├── pass_at_k.py
│   └── shrinkage_slope.py
├── dashboard/
│   └── app.py                   # Streamlit
├── train.py                      # single entrypoint, gate + model size are CLI args
└── results/
    ├── n1_scaling/
    ├── n2_stress_test/
    └── n3_cbgrpo/
```

Design `train.py` so a full run is `python train.py --model 1.5b --gate cb_grpo --seed 0` — this makes the whole N1+N2+N3 grid a simple shell loop, which matters a lot when you're running dozens of jobs on a single 12GB card over a semester.

---

## 8. Timeline (refined against N1/N2/N3)

| Weeks | Milestone |
|---|---|
| 1–2 | Repo scaffold, quantized stack, heuristic rewards, unit tests, `VanillaGate` working end-to-end at 0.5B |
| 3–4 | Pass@k estimator + shrinkage-slope code; first N1 data point (0.5B, vanilla) |
| 5–6 | N1 complete: vanilla GRPO at 0.5B and 1.5B, scaling figure draft |
| 7–8 | Implement remaining gates (SELF, O-SELF, adaptive-rollout); N2 grid running |
| 9–10 | CB-GRPO implementation + clustering + logging; first CB-GRPO runs |
| 11 | Ablation sweep (Section 5.6); 3B stretch run if time/VRAM allow |
| 12 | Dashboard polish, freeze all figures/tables |
| 13+ | IEEE write-up, formatting, internal review pass |

---

## 9. Related-Work Framing (paragraph you can adapt directly)

> Recent work has explored population-based LoRA adapters for RLVR exploration [PopuLoRA; EvoPref] and adaptive rollout budget allocation to combat reward saturation [HORA; AERO; SARA; XRPO], alongside accuracy-gated data curation [SELF] and negative/positive-only reward reformulations [NGRPO; NSR; POPO]. These methods are uniformly validated at 7B–14B+ scale on multi-GPU infrastructure. To our knowledge, no prior work systematically characterizes how boundary-shrinkage severity scales with trainable parameter capacity under single consumer-GPU constraints, nor tests whether these mitigations retain their effectiveness once capacity drops to the sub-2B regime. We further note that all existing gating/curation mechanisms operate on an accuracy or reward-variance signal; we introduce a capacity-native alternative that gates directly on cumulative gradient-mass consumption per reasoning-pattern cluster.

---

## 10. Risk Register (things that could go wrong — plan for them now)

| Risk | Mitigation |
|---|---|
| 3B doesn't fit in 12GB with rollouts | Drop G, or drop 3B to "future work," report the wall honestly |
| Clustering quality is noisy on GSM8K (relatively homogeneous problems) | Fall back to difficulty-based clustering (base-model solve-rate buckets) as a secondary clustering scheme, ablate both |
| CB-GRPO shows no improvement over O-SELF | Still publishable as N1+N2 alone with CB-GRPO as a negative result — be honest, it's still a contribution |
| Training instability at 0.5B (very small models can be erratic under RL) | Lower LR, increase warmup, consider a light SFT warm-start before RL if base instruct-tuning isn't enough |
| Time runs out before 3B stretch goal | It was always marked stretch — cut without guilt, N1+N2+N3 at two sizes is a complete paper |

---

**Next steps I can help with:** turning Section 5's CB-GRPO pseudocode into an actual runnable PyTorch/TRL module, drafting the IEEE paper's Methods section from this spec, or setting up the config/YAML files for the 0.5B run so you can start training this week.
