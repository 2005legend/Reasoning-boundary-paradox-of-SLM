# Project Summary: Mitigating Reasoning-Boundary Shrinkage in RLVR-Trained Small Language Models

**Purpose of this document:** a self-contained handoff/reference covering the research motivation, goals, novelty (both the original idea and the upgraded version), and full history of what's been built and debugged so far. Written so it can be picked up cold, in a new conversation, for continued research/writing work — separate from the engineering debugging thread.

---

## 1. Project Summary

This project studies **reasoning-boundary shrinkage** (also called "Pass@k inversion") in reinforcement learning with verifiable rewards (RLVR) applied to **small language models** (Qwen2.5-0.5B and 1.5B-Instruct) on grade-school math (GSM8K, with MATH-500 as a secondary/harder benchmark). The phenomenon: RLVR training reliably improves Pass@1 (single-sample accuracy) but can *shrink* Pass@k at larger k — meaning the base model could originally solve a broader diversity of problems (across many samples) than the RL-tuned model can, even though the RL model looks better on the metric everyone reports (Pass@1). We study two candidate mitigations (a topic-level "cluster budget" gate, and a per-prompt "already-solved" gate), diagnose why the topic-level one gave only a weak effect on its own, and design + implement a combined method (**H-CB-GRPO**) that addresses both a macro (topic-diversity) and a micro (per-prompt winner-take-all) mechanism at once. The project also includes a from-scratch custom GRPO training pipeline built for Kaggle's free 2×T4 GPU tier, and mechanism-level diagnostics adapted from the literature to explain *why* things work or don't, not just *whether* they do — this framing is deliberately chosen to be viable as a small-scale, honest, diagnostic-style research contribution suitable for a workshop paper and for a Master's application, rather than a "beat SOTA" claim that a two-GPU Kaggle budget can't support.

---

## 2. Background: the Phenomenon and the Base Papers

- **Core citation / origin of the phenomenon:** Yue et al., *"Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"* (arXiv:2504.13837) — established that RLVR-tuned models can have a *narrower* reasoning boundary (lower Pass@k at large k) than their own base model, even while Pass@1 improves.
- **The paper this project's title and mitigation baselines come from:** Nguyen et al., *"The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models"* (arXiv:2510.02230) — proposes **SELF**, a data-curation method that identifies the mechanism as **winner-take-all reinforcement**: GRPO disproportionately reinforces the single most-likely correct solution mode for problems the base model can already solve, while gradient for problems whose correct answers sit in low-likelihood regions vanishes. SELF's fix: exclude ("gate out") training signal for prompts the model has already effectively solved (via greedy-decode correctness), redirecting learning toward low-likelihood/boundary problems. This paper also supplies the **interference metrics** (Δ⁺, ‖Δ‖ — Definition 4.1) used later in this project's diagnostics.
- **Closest recent competing/adjacent work** (all 2025–2026, important to cite and differentiate from):
  - *"When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion"* (PBA, arXiv:2607.20543, July 2026) — anchors training on rare base-model-correct trajectories **per problem**. Very close in spirit; this project's cluster-level anchoring idea (see §5, stretch item) is a scoped-down, differentiated cousin of this.
  - *"Limits of Difficulty Scaling: Hard Samples Yield Diminishing Returns in GRPO-Tuned SLMs"* (arXiv:2604.06298, April 2026) — **same exact experimental setup** as this project (Qwen 0.5B–3B, LoRA, GRPO, GSM8K/MATH). Must be cited/differentiated for the capacity-scaling research question (N3).
  - *"Uniform-Correct Policy Optimization"* (arXiv:2605.00365) — clusters *responses within a prompt* by reasoning strategy; conceptually adjacent to this project's clustering idea but at a different granularity (intra-prompt vs. this project's inter-prompt/topic-level clustering).
  - *"Understanding Diversity Collapse in RLVR via the Lens of Overtraining"* (arXiv:2606.15455) — cautions that Pass@k decline can be partly explained by overtraining/redundant reinforcement, not only "lost capability." Relevant to how shrinkage-slope results should be interpreted/caveated.
  - *"The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both?"* (arXiv:2510.04028) — two-stage account (early shrinkage, possible later expansion with prolonged training); relevant to the step-ablation experiment.
  - GRIP (arXiv:2603.00031), PSN-GRPO (arXiv:2602.02555) — adjacent macro-budget / composability precedents, useful as related-work citations.

---

## 3. Research Goals

- **N1 — Boundary-shrinkage detection:** confirm/characterize Pass@k shrinkage at SLM scale (0.5B, 1.5B), not just the larger models the original literature used.
- **N2 — Mitigation transfer:** test whether existing mitigations (O-SELF, Static-SELF, Adaptive-Rollout — all per-prompt/data-curation style) transfer to SLMs.
- **N3 — Capacity–shrinkage relationship:** does shrinkage severity scale with model capacity (0.5B vs. 1.5B)? Must differentiate from arXiv:2604.06298's closely related finding.
- **N4 — Headline novel contribution:** do a macro (topic-cluster) intervention and a micro (per-prompt) intervention address *distinct* components of shrinkage, such that combining them (H-CB-GRPO) is complementary/super-additive relative to either alone?
- **N5 — Diagnostic contribution (free, no extra training):** does the interference-metric drop (Δ⁺, from the SELF paper's own theoretical machinery) mediate the relationship between cluster-spend imbalance (Gini coefficient) and shrinkage slope — i.e., can we show *mechanistically*, not just by outcome, what each gate does and doesn't fix?

---

## 4. Original Design and Novelty (v1)

The original idea, **CB-GRPO (Cluster-Balanced / Capacity-Budgeted GRPO)**: cluster the ~7,500 GSM8K training prompts into 16 semantic topics (sentence-transformer embeddings + KMeans), track an EMA of "gradient mass" (advantage magnitude) spent per cluster during training, and softly down-weight (decay, not hard-exclude) samples from any cluster whose cumulative spend exceeds a threshold relative to the mean. The intent: prevent RLVR from over-training on a few "easy" topic clusters at the expense of others, hypothesized to help preserve solution diversity.

**Baselines this was originally designed to compare against** (from the SELF paper's family): Vanilla GRPO, O-SELF (online per-prompt EMA solve-rate gating), Static-SELF (precomputed solve-rate gating), Adaptive-Rollout (reward-variance-based filtering, in the spirit of DAPO's dynamic sampling).

**Original experiment plan:** Exp0 (base-model Pass@k baseline), Exp1–Exp3 (gate comparisons across model sizes, seeds, and training-step checkpoints), with EARS-format requirements (`requirements.md`, ~38 requirements) and a detailed design document (`design.md`) written in a spec-driven-development style, targeting a single-GPU Google Colab notebook with `unsloth`/TRL-based training.

**Empirical result that motivated the upgrade:** CB-GRPO alone showed only a small effect on shrinkage slope relative to Vanilla GRPO — this was the starting point for the novelty upgrade below.

---

## 5. Novelty Upgrade (v2 — the addendum)

### 5.1 Root-cause diagnosis (the key insight)

CB-GRPO's cluster-level budget operates on **topic diversity** (a corpus-level, cross-prompt axis). But the literature's actual mechanism for shrinkage (from the SELF paper) is **per-prompt, per-solution-mode winner-take-all** — a finer-grained effect a 16-way topic partition can't see, because a single cluster mixes together "already-solved, winner-take-all" prompts and "genuinely hard, boundary" prompts. Throttling the whole cluster once it "overspends" dilutes exactly the signal that should be suppressed (over-reinforcement of already-dominant correct solutions) while also suppressing useful corrective learning on the cluster's hard problems. This mismatch — not a flawed premise — is the most likely explanation for CB-GRPO's weak measured effect, and is treated as a genuine, citable finding rather than a failure.

A secondary contributor identified: CB-GRPO's original "spend" metric accumulated `|advantage|` (absolute value), which conflates *harmful* over-reinforcement of correct answers with *useful* corrective negative gradient on wrong answers in hard clusters — refined to accumulate only `max(advantage, 0)` (positive mass), so the gate specifically targets winner-take-all reinforcement.

### 5.2 Headline new method: H-CB-GRPO (Hierarchical/Composite Gate)

A `CompositeGate` combining two components multiplicatively:
- **Macro (refined CB-GRPO):** cluster-level EMA of positive-advantage spend; throttles clusters that are over-reinforcing already-dominant solutions.
- **Micro (O-SELF, reused not reinvented):** per-prompt EMA solve-rate; throttles individual prompts the model has already effectively solved, regardless of which topic cluster they're in.

`gate_weight = macro_weight × micro_weight` (other combination operators — min, harmonic mean — supported as an ablation axis).

The core ablation this enables: Vanilla vs. CB-GRPO (macro-only) vs. O-SELF (micro-only) vs. H-CB-GRPO (both) — if H-CB-GRPO beats both single-axis gates by more than seed noise, that's direct evidence the two mechanisms are complementary, not redundant; if it doesn't, that's also a reportable, honest finding (the two are redundant), provided the mechanism-level diagnostics below explain why.

### 5.3 Mechanism-level diagnostics (the "free" novelty — no extra training)

Adapting the SELF paper's own theoretical machinery (Definition 4.1) to compute, on checkpoints already being saved:
- **Δ⁺ (interference):** mean change in log-probability assigned to a fixed probing set of base-model-correct completions, across training — a negative value is the literature's "negative interference" signature.
- **‖Δ‖:** magnitude of influence (mean squared change in log-probability).
- **Token entropy** of the sampling distribution during generation, tracked over training.
- **Exploratory correlation** between cluster-spend Gini coefficient and shrinkage slope/interference, across all trained (gate, seed) conditions — explicitly reported as descriptive/exploratory given the small number of data points (not a hypothesis test with a p-value).

This turns "CB-GRPO helped a little" into "here is exactly what it did and didn't fix, mechanistically" — a stronger paper contribution regardless of the headline numbers.

### 5.4 Supporting methodological upgrades

- **Difficulty-aware clustering ablation:** cluster prompts on `[semantic embedding, base-model solve-rate]` instead of embedding alone, to directly test whether CB-GRPO's effectiveness depends on the clustering *basis* (topic vs. difficulty) — a clean, cheap, diagnostic side experiment.
- **Multi-seed statistical protocol:** minimum 3 seeds per core condition, with a bootstrap confidence interval on the *difference* in shrinkage slope between a condition and Vanilla — explicitly refusing to claim "improvement" when the CI includes zero (a single-seed comparison is not trusted at all under this protocol).
- **Evaluation range / ceiling-effect fixes:** raised eval sample count (n) well above the original default of 10 so Pass@k can be measured at larger k (shrinkage is a large-k phenomenon in the literature); added an explicit ceiling-effect check per dataset, since GSM8K can be near-saturated for these model sizes, in which case MATH-500 becomes the primary evidence source for shrinkage claims rather than GSM8K.

### 5.5 Optional stretch (not required for the core story)

Cluster-level base-anchoring: when a cluster's macro spend crosses its threshold, mix in a small forward-KL/SFT loss against a cache of frozen base-model-correct completions for that cluster's under-sampled prompts — a scoped-down, cluster-level cousin of the PBA paper's per-problem anchoring (arXiv:2607.20543). Deliberately treated as future-work/stretch, not part of the headline method, to keep the core contribution to one clean, well-ablated idea.

---

## 6. Positioning and Framing (for the paper / application)

- Framed explicitly as a **diagnostic-and-mitigation** paper, not a SOTA paper — realistic given 0.5B/1.5B models and Kaggle-tier compute.
- Target audience: workshop tracks (main-track NeurIPS/ICLR/ICML would expect either much larger scale or a bigger theoretical contribution than this can offer); an arXiv preprint with a clean, working repo is valuable for a Master's application on its own, independent of formal acceptance.
- The honest, stated limitation set: small model scale, GSM8K/MATH-500 only, Kaggle 2×T4 compute budget, limited seed count relative to ideal — explicitly written up as a "Limitations" section rather than hidden, since reviewers trust papers more, not less, for stating this plainly.
- Draft contribution statement (from the earlier addendum, adapt once results are in): *"We study whether reasoning-boundary shrinkage under RLVR training of small language models arises from a single mechanism or from two distinct, separately-addressable components... We show that a cluster-level gradient-mass budget (CB-GRPO) targeting [topic diversity] alone produces only a marginal reduction in shrinkage slope, and use interference and entropy diagnostics to show this is because it leaves [per-prompt winner-take-all] largely unaddressed. We introduce a hierarchical gate (H-CB-GRPO) combining both, and show [result — fill in once multi-seed data is in]..."*

---

## 7. Engineering Build: What Was Actually Implemented

A from-scratch GRPO training pipeline (deliberately **not** wrapping `trl.GRPOTrainer`, since gating requires modifying post-normalization advantages, and TRL's internal method names for that have changed across versions — reaching into private internals would be fragile right before a deadline). Built for **Kaggle's free 2×T4 GPU tier**, using HuggingFace `accelerate` for real multi-process DDP (each GPU gets its own full QLoRA-quantized model copy via `device_map={"": local_process_index}`, not `device_map="auto"`, which would fight DDP by model-sharding instead of data-paralleling).

**Modules built and unit-tested (pure-logic pieces verified with synthetic data before touching a GPU):**
- `config.py` — experiment configuration, compute tiers (smoke/standard/final), gate configs.
- `rewards.py` — output parser, pretty-printer (round-trip tested), sympy-based correctness checking, format/correctness reward.
- `gates.py` — `VanillaGate`, `CBGRPOGate` (refined), `OSELFGate`, `CompositeGate` (H-CB-GRPO) — including a verified-correct **distributed state-sync protocol** (gate state is kept identical across both GPUs via all-reduce every step, proven via a 2-rank-vs-1-combined-rank synthetic equivalence test).
- `evaluation.py` — unbiased Pass@k estimator, shrinkage-slope regression with bootstrap CI, multi-seed Δslope significance testing, Gini coefficient.
- `diagnostics.py` — interference metrics, entropy calculations, exploratory correlation utilities.
- `clustering.py` — semantic and difficulty-aware KMeans clustering, with a synthetic proof that difficulty-aware clustering recovers a hard/easy split semantic-only clustering is blind to.
- `model_utils.py`, `grpo_core.py` — QLoRA model loading, rollout generation, memory-efficient log-prob scoring, the custom GRPO advantage/PPO-clip loss (validated in pure numpy before being transcribed into torch).
- `train.py` / the merged Kaggle notebook (`rlvr_training_pipeline.ipynb`) — main training loop, checkpointing/resume, periodic evaluation.
- `analyze_results.py` / notebook §14 — cross-run aggregation producing the Δslope significance table and the exploratory Gini/shrinkage correlation.

The user (project owner) independently merged this package into a single Kaggle notebook, which was then reviewed cell-by-cell against the original package for fidelity.

---

## 8. Debugging History (chronological, for context on current state)

This is the messy-but-important part — a real account of what's been found and fixed while getting the notebook to actually run on Kaggle:

1. **Notebook-merge review:** content was faithfully transcribed (confirmed correct: refined CB-GRPO, O-SELF, CompositeGate, PPO-clip math, `disable_adapter()` reference-policy trick, per-process device placement). Found and fixed: (a) a **baseline-path collision bug** — `baseline_pass_at_k.json` was keyed at a path that stripped both `seed` *and* `model_size`, meaning 0.5B and 1.5B runs of the same gate type would silently share/corrupt each other's base-model baseline, directly threatening the N1/N3 cross-scale comparison; (b) `CBGRPOGate` was missing a `diagnostic_snapshot()` method that only `CompositeGate` had, limiting which conditions could contribute to the Gini/shrinkage correlation.
2. **Critical infra gap:** running the training cell inline in the notebook only ever uses **one** of the two T4s — `accelerate_config.yaml`'s `num_processes: 2` only takes effect when `accelerate launch` is what *starts* the process (it spawns N independent OS processes); calling `Accelerator()` inside an already-running Jupyter kernel is always `world_size=1`. Fixed by adding a cell that writes the assembled code to a standalone `train_script.py` (verified byte-for-byte round-trip, since embedding source-as-a-string risks corrupting backslashes in regex/LaTeX patterns) and launching it via `accelerate launch` as a subprocess.
3. **HF Hub cache race condition:** both DDP processes downloading the same model simultaneously caused a 30-minute stall (lock contention). Fixed with a pre-download step / recommended `accelerator.main_process_first()` wrapping.
4. **NCCL collective timeout (10 min default):** the single-process baseline Pass@k computation (200 problems × n samples) legitimately took longer than 10 minutes while the other rank waited at a barrier, and NCCL's watchdog killed the whole job assuming a hang. Fixed by extending the process-group timeout (`InitProcessGroupKwargs(timeout=timedelta(hours=2))`).
5. **OOM in `score_sequences` (two rounds):** first, a naive `log_softmax` over the full ~152k-token Qwen vocabulary for every position materialized a many-GB float32 tensor; switching to `cross_entropy`'s fused kernel helped but didn't fully fix it, since it still needs a buffer proportional to `(batch × seq_len) × vocab`. Final fix: **chunk the batch dimension inside `score_sequences` itself** (process a few sequences at a time, concatenate results, gradients still flow correctly) — this bounds peak memory independent of the configured group size / prompts-per-step.
6. **Impractical ETA (67 hours for 800 steps):** traced to `grad_accum_steps=4` quadrupling the generate+score+backward cost per logged step, compounded by inherently slow 4-bit (QLoRA) generation on a T4. Temporary debug-speed levers given: reduce `grad_accum_steps`, `max_new_tokens`, `rollouts_per_prompt` while iterating; a bigger structural fix (using a separate bf16, non-quantized copy of the model for generation-only workloads) was noted as a valuable future optimization but not yet implemented.
7. **Dead reward signal (the most recent and most important finding):** `fmt=0.00, cor=0.00` at step 0 — and it turned out to be the same root cause behind *all* the earlier all-zero results (probing set, baseline Pass@k, training reward), since they all route through the same `parse_output` function. Diagnosis, confirmed by inspecting raw completions: **the base model reliably produces `\boxed{answer}` but does not follow the custom `<reasoning>/<answer>` XML tag scheme zero-shot** — a very common small-model behavior. Since `parse_output` only searched for `\boxed{}` *inside* an `<answer>` block, genuinely correct answers (e.g., a completion that correctly computed `\boxed{3}` matching ground truth `'3'`) were being scored as wrong. **Fix (just written, not yet confirmed by a full run):** `parse_output` now falls back to searching the whole completion for `\boxed{}` when no `<answer>` tag is found; `format_reward` now gives partial credit (0.5) for a parseable boxed value without the tags, full credit (1.0) only with true tag compliance — so RLVR has real reward signal from the very first step instead of a permanently-zero objective.

---

## 9. Current Status and Immediate Next Steps

- **Not yet confirmed:** whether the reward-signal fix (§8, item 7) actually gets training producing nonzero reward/correctness once run. This is the single most important thing to verify before trusting any further results.
- **Still open:** the 67-hour ETA problem (§8, item 6) — debug-speed settings were suggested but a full-speed, statistically adequate `standard`-tier run hasn't yet completed.
- **Not yet run at all:** the actual multi-seed, multi-gate ablation (Vanilla / CB-GRPO / O-SELF / H-CB-GRPO × 3 seeds) that N4's headline claim depends on — all debugging so far has been on a single `vanilla / 0.5B / seed 0` smoke-adjacent attempt.
- **Recommended immediate order of operations:** (1) confirm the reward fix works (nonzero fmt/cor within the first handful of steps) on a fast/cheap config; (2) once confirmed, re-tune batch/step settings for a tractable per-step wall-clock time within Kaggle's session limits; (3) run the core 4-gate × 3-seed `standard`-tier comparison at 1.5B (this is what N4's significance claim depends on); (4) run cross-run analysis (§7's `analyze_results.py` / notebook §14) to see whether H-CB-GRPO's improvement over Vanilla/CB-GRPO/O-SELF individually excludes zero in the bootstrap CI; (5) only then consider a `final`-tier confirmatory run on the winning condition, and the difficulty-aware clustering ablation, and the interference-diagnostics writeup, time permitting.

---

## 10. Key Files Reference

- `requirements.md`, `design.md`, `tasks.md` — original EARS-format spec trio (requirements currently only formally defined up to Requirement 38; the novelty upgrade's Requirements 39–43, referenced in `design.md`'s newer algorithm sections, have not yet been formally back-filled into `requirements.md` — a known inconsistency still worth reconciling before finalizing any spec-facing writeup).
- `novelty_upgrade_addendum.md` — the full novelty-upgrade proposal (root-cause diagnosis, H-CB-GRPO algorithm, diagnostics, related-work table, prioritized roadmap, positioning advice) referenced throughout §5–6 above.
- `rlvr_pipeline.zip` — the original from-scratch package (config/rewards/gates/evaluation/diagnostics/clustering/model_utils/grpo_core/train/eval_full/analyze_results.py + Kaggle accelerate config/README).
- `rlvr_training_pipeline_fixed.ipynb` — the user's merged Kaggle notebook, with the baseline-path and Gini-diagnostic-coverage bugs fixed, plus the added §7b (real multi-GPU launch via `accelerate launch` subprocess).
- Standalone patch files produced during live debugging (apply these into the notebook's big script cell if not already merged): `fixed_score_sequences_v2.py` (chunked, memory-safe log-prob scoring — supersedes an earlier, insufficient `cross_entropy`-only patch), `fixed_rewards_patch.py` (the `\boxed{}` fallback + partial-credit format reward — the dead-reward-signal fix), `fixed_probing_set_cell.py`, `fixed_launch_cell.py`, `diagnose_dead_reward.py`.
