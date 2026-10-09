# HEG-GRPO Progress Log

**Purpose:** Running changelog for the project's novelty work, kept up to date as results
come in. This is the source document to pull from when writing the paper's Method,
Experiments, and Related Work sections — each entry below is dated and states *why* a
decision was made, not just what changed, so the reasoning survives even after the code
has moved on.

Related docs: [novelty_upgrade_addendum.md](../novelty_upgrade_addendum.md) (full algorithm
spec, EARS-format requirements, related-work table, positioning guidance) ·
[EVALUATION_RESULTS_CBGRPO_vs_VANILLA.md](EVALUATION_RESULTS_CBGRPO_vs_VANILLA.md) (prior
0.5B CB-GRPO vs Vanilla results — see §2 below for a critical re-read).

---

## 1. Where the project stands (as of 2026-09-22)

| Condition | Model | Status |
|---|---|---|
| Vanilla GRPO | 0.5B | Trained (Kaggle, 2026-09-11), 800 steps. Evaluated — **see §2, eval methodology is suspect.** |
| CB-GRPO (sampler-throttling variant) | 0.5B | Trained (Kaggle, 2026-09-10/11), 800 steps, 9h28m. Evaluated — **see §2.** |
| Vanilla GRPO | 1.5B | Trained (friend's run, per `07_for_friend/`). Not yet evaluated in this log. |
| CB-GRPO (gate-multiplier variant) | — | Not yet trained under the new `rlvr_training_pipeline_fixed.ipynb` pipeline. |
| **HEG-GRPO (headline, Req 44)** | 0.5B | Implementation complete (2026-09-22). Smoke-tier (80-step) validation run launched on Kaggle 2×T4 — **in progress, not yet reviewed.** |

**Important architecture note:** the 0.5B CB-GRPO result in §2 was trained with an
earlier, *different* CB-GRPO implementation — a **sampler-based throttling** approach
(`CBGRPOSpendTracker` in `03_cbgrpo_implementation/`, wraps the reward function to track
spend but changes which clusters get *sampled* more/less often). The current
`rlvr_training_pipeline_fixed.ipynb` pipeline (`CBGRPOGate` in `gates.py`) is a **gradient/
advantage-gating** approach — it multiplies the GRPO advantage by a per-sample weight,
sampling stays uniform. Both are legitimate "capacity-balanced GRPO" ideas, but they are
not the same mechanism and their numbers are not directly comparable. State this precisely
in the paper's method lineage section — don't present the §2 CB-GRPO numbers as evidence
for the current `CBGRPOGate`/HEG-GRPO's effectiveness; they're evidence for a related but
different prior method.

---

## 2. Critical re-read of the 0.5B CB-GRPO vs Vanilla results (2026-09-22)

Re-checked `EVALUATION_RESULTS_CBGRPO_vs_VANILLA.md` and the raw training logs behind it
before trusting its "Primary Hypothesis: ✅ VALIDATED" claim. Two findings that should be
resolved before this comparison is cited anywhere in the paper:

### 2.1 The "vanilla didn't learn" claim is very likely an eval artifact, not a training fact

The eval doc's headline finding is: *"Vanilla GRPO shows identical performance to base
model... suggesting training may have been ineffective."* But the vanilla run's own
`trainer_state.json` (`exp2_n1_vanilla_0.5B/checkpoint-800/trainer_state.json`,
`log_history`) shows `rewards/compute_reward_for_grpo/mean` climbing steadily and clearly
across training:

| Step | Mean reward |
|---|---|
| 10 | 0.020 |
| 20 | 0.035 |
| 30 | 0.178 |
| 760 | 0.323 |
| 800 | 0.488 |

That is unambiguous evidence the vanilla model **did** learn during training — reward
climbed ~24x from step 10 to step 800. This directly contradicts the eval doc's
conclusion. The most likely explanation: the eval script used a **strict regex**
(`r'<answer>\s*(-?\d+...)\s*</answer>'`) that only credits an answer inside `<answer>`
tags. If the vanilla-trained model tends to answer with a bare `\boxed{}` (common for
small instruct models, and the exact failure mode described in the project's own earlier
bug history — see §2.2), the eval script would silently undercount correct answers
regardless of real capability, and could do so *differently* for the vanilla vs CB-GRPO
checkpoints if their tag-adherence rates differ for unrelated reasons.

**Action before citing this comparison:** re-evaluate both checkpoints with an eval script
that has the raw-text `\boxed{}` fallback (see §2.2 — now present in the new pipeline's
`parse_output`), and/or manually inspect a sample of the vanilla model's raw completions
from that eval run to confirm whether `<answer>` tags were actually being produced. Until
then, treat the "CB-GRPO learned, vanilla didn't" framing as **unconfirmed**, not as a
result to build the paper's narrative around.

### 2.2 Sample size: differences reported are within noise

Absolute Pass@1 on GSM8K was 0.01–0.02 (i.e., ~0.3–0.6 "correct" out of 30 problems,
n=16 samples/problem). A "2x improvement" at this scale is not distinguishable from noise
without a seed/CI analysis — none was run. This is exactly the gap Requirement 42
(multi-seed Δslope CI, already implemented in the current pipeline's §12) exists to close.
Don't repeat this pattern for HEG-GRPO: no "X% improvement" claim should be written up
without a bootstrap CI that excludes zero (Req 42.4).

### 2.3 `parse_output` fallback — found a live inconsistency in the current notebook

While re-checking the reward pipeline, found that the "fallback: scan raw text for
`\boxed{}` if no `<answer>` tag is found" fix (present in the actual training script
embedded in cell 45, `TRAIN_SCRIPT_SRC`) was **missing from cell 14**, the notebook's own
visible `rewards.py` module — meaning the notebook's own §4 self-tests (cell 33) were
validating a different, stricter `parse_output` than the one actually used in training.
**Fixed** (2026-09-22): ported the fallback into cell 14 so both copies match; re-ran the
full offline self-test suite locally afterward — all pass, including the pre-existing
rewards/gates/evaluation/diagnostics tests. This was a documentation/consistency bug, not
a training-time bug (cell 45 already had the fix), but worth noting since it's the same
failure class as §2.1 above (strict tag matching silently zeroing out correctness reward)
— if you see `cor=0.00` stuck in any future training log, this is the first thing to check.

---

## 3. HEG-GRPO: headline novelty (2026-09-22)

### 3.0 ⚠️ Novelty check (2026-09-22) — read before writing the paper's contribution claim

Ran a literature check on arXiv/Google Scholar specifically for MEG's core mechanism
("correctness-aware, intra-group, embedding/entropy-based redundancy reweighting to fight
GRPO winner-take-all"). Result: **the mechanism itself is not novel.** Three verified,
real, published papers do essentially the same thing:

- **GCPO** (arXiv:2605.11461, May 2026) — closest match. Explicitly targets winner-take-all
  within GRPO groups; states "only correct and non-redundant rollouts contribute" to its
  diversity credit. This is MEG's exact idea, via a determinant-volume measure instead of
  clustering+entropy.
- **DRA-GRPO** (arXiv:2505.09655, May 2025) — intra-group embedding-redundancy reweighting
  (Submodular Mutual Information), applied to all rollouts (not correctness-gated).
- **EDAS** (arXiv:2605.17333) — the mirror image: entropy-based, intra-group, but only on
  *incorrect* rollouts; explicitly found embedding clustering unreliable for distinguishing
  wrong answers (76.4% of numerically-different wrong answers had cosine similarity >0.95)
  — a live methodological risk worth checking for MEG's own correct-side clustering too.

All three verified by fetching arXiv directly (titles/authors/dates/abstracts/method
sections confirmed real, not hallucinated).

**Decision (with the user, 2026-09-22): reposition, don't rebuild.** Keep the current
`MEGGate`/`heg_grpo` implementation as-is — it's already tested and mid-smoke-test on
Kaggle — but change what the paper claims about it. The defensible contribution is the
**hierarchical composition** of the cross-topic macro budget (`CBGRPOGate`) with an
intra-group micro diversity term from this emerging GCPO/DRA-GRPO/EDAS family, evaluated
through this project's reasoning-boundary/Pass@k-shrinkage diagnostic lens at SLM scale
under heavy compute constraints — not "we invented intra-group diversity gating." This
combination did not turn up in the same search. `novelty_upgrade_addendum.md` §2, §4.4,
and §11 have been rewritten to reflect this; the abstract draft in §11 now cites
GCPO/DRA-GRPO/EDAS in its second sentence rather than burying them in related work.

**Caveat:** this was one focused search session, not an exhaustive systematic review —
before final submission, re-search closer to the deadline (the field is moving fast per
the addendum's own §0) and specifically check whether anyone has since published the
macro+micro combination this project is now claiming as its gap.

### 3.1 Why this exists

`H-CB-GRPO` (macro cluster-budget × `OSELFGate` micro term) was the original composite
gate. Its micro term is a *cross-step, per-prompt binary proxy*: it flags a prompt
"solved" only after many past steps' evidence accumulates into an EMA, and treats the
whole prompt as one unit rather than looking at the individual rollouts within a single
GRPO group — which is where winner-take-all reinforcement literally happens
(arXiv:2510.02230 §4–5). That's a real gap between what the gate measures and the
mechanism the project is diagnosing.

**HEG-GRPO** (`CBGRPOGate` macro × `MEGGate` micro) replaces that proxy with a direct,
per-step, stateless measurement: embed the `group_size` rollouts already generated for a
prompt this step, cluster them into solution "modes" by cosine similarity, and down-weight
only the *correct* rollouts that share an over-represented mode. Incorrect rollouts always
keep full gradient (mirrors the macro gate's own positive-mass fix — never throttle
corrective signal). Full algorithm and EARS-format acceptance criteria: Requirement 44 in
`novelty_upgrade_addendum.md`.

### 3.2 What changed in the notebook

All changes are in `rlvr_training_pipeline_fixed.ipynb`, mirrored into both the notebook's
own module cells and the embedded `TRAIN_SCRIPT_SRC` (the actual `accelerate launch`
script) so the two stay consistent — see §2.3 above for why that consistency matters.

| Cell | File | Change |
|---|---|---|
| 12 | `config.py` | New `GateType` entries `meg`, `heg_grpo`; new `GateConfig` fields `meg_lambda`, `meg_tau_mode`, `meg_embedding_model`; `meg`/`heg_grpo` explicitly exempted from the 0.5B incompatibility rule (stateless — no per-prompt EMA array) |
| 14 | `rewards.py` | `parse_output` fallback fix ported from cell 45 (§2.3) |
| 18 | `clustering.py` | Cached, CPU-pinned `get_sentence_model()` loader (avoids re-downloading/reloading the embedding model every step, keeps it off the T4's VRAM) |
| 24 | `gates.py` | New `compute_group_mode_freq()`, new `MEGGate` class, `Gate`/`CompositeGate` interface extended to thread `mode_freq`/`group_size` through, `build_gate` factory extended |
| 28 | `grpo_core.py` | `run_training_step` computes mode frequencies when the active gate needs them, logs mean group-mode entropy per step |
| 32, 34 | self-tests | New assertions: 0.5B+heg_grpo allowed, MEGGate math (dominant vs. rare mode, incorrect-never-throttled, no-op without mode_freq), HEG-GRPO composite behavior |
| 38, 46 | experiment config / launch | New gate types selectable; both currently set to `GATE_TYPE="heg_grpo"`, `MODEL_SIZE="0.5B"`, `COMPUTE_TIER="smoke"` for the first validation run |
| 49, 53, 57, 58, 59 | diagnostics / ablation / checklist | `meg`/`heg_grpo` added to condition lists, plot colors, and the experiment checklist table (Exp 5/6) |

**Verification done before considering this "implemented":** full offline self-test suite
run locally (config/rewards/gates/evaluation/diagnostics — all pass), plus a standalone
test of `compute_group_mode_freq` with real sentence-transformer embeddings confirming it
correctly separates a different-reasoning-path completion from duplicates. Every notebook
cell and the full embedded training script were syntax-checked (`ast.parse` / `py_compile`).
**Not yet done:** an actual GPU training run — that's the smoke test in progress.

### 3.3 Differentiation for related work (for the paper)

- **vs. GCPO (arXiv:2605.11461) — closest match, lead with this citation:** already does
  correctness-gated intra-group redundancy reweighting. MEG differs only in formalization
  (discrete clustering + entropy vs. determinant-volume) and in being composed with a
  separate macro budget, which GCPO doesn't have. Claim the composition, not the mechanism.
- **vs. DRA-GRPO (arXiv:2505.09655):** intra-group embedding-redundancy reweighting via
  Submodular Mutual Information, uniform across correct/incorrect rollouts. Cite as origin
  of the idea; MEG's correctness-gating and macro composition are the deltas.
- **vs. EDAS (arXiv:2605.17333):** entropy-based, intra-group, but exclusively on incorrect
  rollouts — the opposite side of the group from MEG. Also flags embedding-clustering as
  unreliable for their (exact-answer) use case — worth a validation check on MEG's own
  clustering (are the modes tracking real solution-method differences, or surface style?).
- **vs. UCPO (arXiv:2605.00365):** UCPO enforces a hard uniform-mass *constraint* across
  intra-prompt response clusters (a new loss term). MEG is a soft *multiplicative gate* on
  the existing GRPO advantage — no new loss term, consistent with this project's existing
  "soft decay, no hard cutoff" design commitment.
- **vs. PBA (arXiv:2607.20543):** PBA anchors training on rare base-model-correct
  completions per problem (adds an SFT-style loss term, needs a cached completion store).
  MEG adds no anchoring loss and stays purely on-policy, operating one level finer (within
  a rollout group, not across a prompt's whole training history).
- **vs. this project's own H-CB-GRPO:** same `CompositeGate` wrapper, different micro
  mechanism — a cross-step correctness proxy (`o_self`) vs. same-step within-group
  redundancy measurement (`meg`, adapted from the GCPO family). Keep both in the ablation
  table; the HEG-GRPO vs. H-CB-GRPO comparison — not the micro mechanism alone — is where
  this project's evidence now has to come from.

---

## 4. Open items to resolve before writing the paper

- [ ] **Resolve §2.1** — re-run the 0.5B CB-GRPO/vanilla eval with a boxed-fallback-aware
      scorer (or manually inspect raw completions) before citing that comparison anywhere.
      If it doesn't hold up, the paper should not lean on it at all — HEG-GRPO's own
      multi-seed results (§4.2 below) are the real evidence base.
- [x] Review the HEG-GRPO smoke-test log — see §6. Reward signal confirmed alive; found
      and fixed two real bugs (`mean_group_mode_entropy` not logged, gradient accumulation
      silently not happening) before the standard-tier run.
- [ ] Decide on wall-clock mitigation (§6.2) and re-launch the smoke tier once to confirm
      the `gradient_accumulation_steps` fix changes the LR-schedule/reward trajectory as
      expected, before committing Kaggle hours to `standard` tier.
- [ ] Bump `COMPUTE_TIER` to `standard` and run the full 6-condition ablation
      (Vanilla / CB-GRPO / O-SELF / H-CB-GRPO / MEG / HEG-GRPO) at 1.5B, ≥2 seeds each
      (Req 42.1), before any "X% improvement" claim is written down.
- [ ] Run §11 interference diagnostics (Δ⁺, ‖Δ‖) on saved checkpoints once available —
      free evidence, no extra training.
- [ ] Decide whether the old sampler-based CB-GRPO result (§2) is worth re-running under
      the current gate-based `CBGRPOGate` for a clean, directly-comparable baseline, given
      §1's note that the two are architecturally different methods.
- [ ] Before final submission, re-run the novelty search from §3.0 one more time closer to
      the deadline — the field moves fast, and the macro+micro combination this project now
      claims as its gap should be re-checked, not assumed to still be open.

---

## 5. Smoke-test review (2026-09-23)

First real GPU run of `heg_grpo` (0.5B, smoke tier) reviewed from Kaggle log output
(`step 0` and `step 5` progress lines). Two real bugs found and fixed in
`rlvr_training_pipeline_fixed.ipynb` cell 45 as a result — the currently-running Kaggle
job predates both fixes, so re-launch after pulling the updated notebook.

### 5.1 Reward signal: confirmed alive, `\boxed{}` fallback confirmed working

`fmt=0.00`, `cor=0.36` at step 0 — the model isn't emitting `<reasoning>/<answer>` tags yet,
but correctness reward fires anyway via the raw-text `\boxed{}` fallback (§2.3's fix, now
live). This is the single most important thing to check per the project's own gotcha list,
and it passed.

### 5.2 Wall-clock: `smoke` tier alone projects to ~7-8h; `standard` tier is not feasible as configured

Step 0→5 took 27m44s (~5.5 min/step), matching the script's own printed ETA
(`07h13m08s` at step 0). At this per-step cost, `standard` tier (800 steps, 10x `smoke`'s
80) projects to **~73 hours** — not completable in one Kaggle session (~9-12h cap) and well
past the weekly T4 quota. Root cause: cell 45's launch script runs
`rollouts_per_prompt=4`, `grad_accum_steps=4`, `max_new_tokens=512` (heavier than cell 12's
defaults of `2`/`1`/`256` — the pre-existing cross-cell drift noted when HEG-GRPO was first
implemented). **Not yet resolved** — before launching `standard`, reduce cost via one or
more of: `grad_accum_steps: 4→1` (largest single lever, ~4x, no group-size/statistics
impact), `rollouts_per_prompt: 4→2` (impacts GRPO group-relative advantage quality),
`max_new_tokens: 512→256` (impacts long-completion accuracy — check completion truncation
rate first).

### 5.3 Bug found and fixed: gradient accumulation was silently a no-op

`Accelerator()` was constructed without `gradient_accumulation_steps=cfg.optim.grad_accum_steps`.
`accelerator.accumulate(model)` needs that value at construction time to gate
`no_sync()`/`AcceleratedOptimizer`/`AcceleratedScheduler` correctly; without it, every
micro-iteration inside the `for micro in range(cfg.optim.grad_accum_steps)` loop was
treated as its own full sync boundary — meaning `optimizer.step()` and `scheduler.step()`
(both called unconditionally in that loop) ran as up to 4 independent real steps per logged
"step" instead of one step over properly accumulated gradients. Two consequences: (a) the
effective per-update batch size was 4x smaller than `OptimConfig.grad_accum_steps=4`
implied, and (b) the LR cosine schedule (built for `num_training_steps=cfg.training_steps`)
was advancing up to 4x faster than its schedule assumed, decaying toward zero long before
the logged step counter suggested. **Fixed**: added
`gradient_accumulation_steps=cfg.optim.grad_accum_steps` to the `Accelerator()` constructor
call. This does not change wall-clock time (§5.2 is a separate issue) — it changes training
*correctness*: gradients now actually accumulate across the 4 micro-batches before one real
optimizer/scheduler step, matching what `OptimConfig.grad_accum_steps` was always meant to
do. **This likely affects every prior run made with this custom pipeline where
`grad_accum_steps > 1`** (i.e., any run launched from cell 45's defaults) — if any earlier
result under this pipeline used `grad_accum_steps > 1`, its effective batch size and LR
schedule were not what the config claimed; re-check before citing any such run.

### 5.4 `mean_group_mode_entropy` logging gap: found and fixed

Computed in `run_training_step` (`StepMetrics.extra`) but never read by the training loop's
`agg` dict, so it never reached `train_log.jsonl` despite being computed every step for
`heg_grpo`/`meg`. Fixed: `agg` now merges any keys present in `step_metrics[i].extra` before
writing the log line. Confirm this key appears in the log on the next run.

### 5.5 Second re-launch (2026-09-23): confirmed the §5.3 fix worked, found a second scheduler bug

Reviewed `lr` at printed steps 0/5/10/15/20 from the re-launched run
(`2.50e-07, 9.92e-07, 9.10e-07, 7.50e-07, 5.44e-07`). Fit these exactly (to 3 significant
figures, all 5 points) against a cosine-with-warmup formula where
`real_step_count = (printed_step + 1) * accelerator.num_processes` — confirming two things
at once: (a) §5.3's fix worked (only 1 real optimizer/scheduler update per printed step now,
not 4), and (b) a **second, separate bug**: `accelerate.scheduler.AcceleratedScheduler.step()`
(verified by reading the installed `accelerate==1.15.0` source directly) calls the wrapped
scheduler's `.step()` `accelerator.num_processes` times per real update — documented,
intentional accelerate behavior to account for DDP, but the scheduler here was built with
`total_steps=cfg.training_steps` (not scaled by `num_processes`), so with `world_size=2` the
LR schedule was completing (decaying to ~0) at printed step 40 of 80 — half the run trains
at a dead learning rate. **Fixed**: `total_steps`/`warmup_steps` passed to
`get_cosine_schedule_with_warmup` are now multiplied by `accelerator.num_processes`.
Re-derived the corrected schedule numerically and confirmed it now spans the full 80 printed
steps (peaks mid-run, reaches ~0 at printed step 79, not 40). **This affects every prior run
under this pipeline with `num_processes > 1`** (i.e. any 2xT4 Kaggle run), on top of the
`grad_accum_steps` issue in §5.3 — the two compounded (4x from §5.3, additional 2x from this)
in the very first smoke-test log, which is why that run's LR had crashed to `3.02e-08` by
printed step 10.

---

## 6. Local training track (2026-09-23) — separate from the Kaggle notebook, not yet built

> **Superseded 2026-10-02 by §7 (AWS track).** The user now has $200 of AWS credits and chose
> AWS EC2 over local GPUs. Steps 2 and the single-GPU design below were implemented in
> `heg_grpo_aws/` (without Unsloth, so the Unsloth compatibility risk no longer applies).

Started because the wall-clock problem (§5.2 — `standard` tier projects to ~73h on Kaggle's
2xT4 as configured) makes Kaggle's session cap a real obstacle, and the user has two local
GPU options: their laptop (RTX 3050, **4GB VRAM**) and a desktop they can borrow (RTX 3060,
**12GB VRAM**) — a Turing/16GB-per-card pair (T4×2) vs a single newer-architecture Ampere card
with native bf16 tensor-core support (T4/Turing has no native bf16 tensor cores; this
pipeline's QLoRA compute_dtype is bf16, so this is a real, not marginal, difference).

**Decision: use Unsloth for local training, not the notebook's plain
`AutoModelForCausalLM` + `bitsandbytes` + `peft` path.** Reasoning: VRAM is the binding
constraint locally (4GB on the laptop is very tight even for the 0.5B model once you add
KV-cache for `group_size` rollouts up to `max_new_tokens`), Unsloth gives large VRAM/speed
wins on exactly this (QLoRA on a single consumer Ampere/Turing GPU), and it has first-party
Qwen2.5 + GRPO-style-workflow support. **Risk to manage, not yet resolved:** this project's
GRPO loop is hand-rolled (`grpo_core.py`, not `trl.GRPOTrainer` — see the project skill's gotcha list on this), and it alternates `model.generate()` and backward passes within one training step.
Unsloth's `FastLanguageModel` needs explicit `for_inference()`/`for_training()` mode toggles
around that alternation, and `model.disable_adapter()` (used for reference log-probs) needs to
be confirmed compatible with Unsloth's PEFT wrapper. **Not yet verified — do this first**
before trusting any local training output.

**Environment status (laptop, RTX 3050 4GB, Windows + WSL2):** WSL2 is installed
(`Ubuntu-24.04`, default distro) and **can already see the GPU** (`nvidia-smi` inside WSL
reports the RTX 3050 correctly, driver 566.07, CUDA 12.7) — this is the right path for
`bitsandbytes`/Unsloth reliability on Windows, native-Windows install is the fallback only if
WSL2 isn't available on the desktop. Blocked on one manual step: `python3.12-venv` isn't
installed in WSL and needs `sudo apt-get install -y python3.12-venv python3-pip
build-essential`, which needs an interactive password this tool can't supply — the user needs
to run that once themselves, in WSL, on whichever machine ends up doing the work.

**Plan (not started — do this in a fresh session, ideally run directly on the desktop with
the RTX 3060 the user is borrowing):**
1. WSL2 + venv + PyTorch (CUDA build matching the driver) + `unsloth` + the same deps the
   notebook uses (`transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `sympy`,
   `scikit-learn`, `sentence-transformers`).
2. Extract the platform-independent modules (`config.py`, `rewards.py`, `gates.py`,
   `evaluation.py`, `diagnostics.py`, `clustering.py` — none of these touch model loading, so
   they can be reused byte-for-byte from the notebook cells) into standalone `.py` files
   rather than re-embedding them in a new giant string — this also sidesteps the
   "two copies drift" problem (§2.3, §5.3-5.5) for at least the local track.
3. New `model_utils_local.py` (Unsloth `FastLanguageModel` load + generate + score, with the
   `for_inference()`/`for_training()` toggling resolved) and a single-GPU `train_local.py`
   (no `accelerate launch`/DDP needed locally — `sync_fn=identity_sync`, so none of §5.3/§5.5's
   scheduler-scaling bugs apply to the local track at all).
4. Validate on the laptop's 4GB card first with conservative settings (small batch,
   `rollouts_per_prompt`, `max_new_tokens`) as a cheap correctness check, before moving to the
   3060 desktop and scaling batch/rollout settings up.
5. Keep the reward/gate/eval logic byte-identical to the Kaggle notebook (step 2 above already
   guarantees this for the reused modules) so results from the local track and Kaggle track
   are actually comparable for the paper.

---

## 7. AWS training track (2026-10-02) — `heg_grpo_aws/`, built and tested, not yet run on AWS

**Why:** $200 AWS credits (expire end of 2027). Kaggle projected ~60-73h per run, and the full
ablation is 12+ runs. User decisions (2026-10-02): **bf16 LoRA instead of 4-bit QLoRA** at 0.5B,
and **EC2 + JupyterLab** (not SageMaker). Training runs in `tmux`; JupyterLab is for
monitoring only.

**What it is:** a standalone package (`heg_grpo_aws/`, 195 KB) built from cell 45's
`TRAIN_SCRIPT_SRC`, which is the code that actually ran on Kaggle. `README.md` in that folder is
the step-by-step AWS runbook. It ends the "two copies of the training code drift" problem for
this track: there is one copy.

**Methodology differences vs the Kaggle notebook (state these in the paper's setup section;
the full table is in `heg_grpo_aws/README.md`):**
- bf16 base + fp32 LoRA adapters, single GPU, plain PyTorch loop (no accelerate/DDP, so the
  §5.3/§5.5 bug class cannot occur).
- 16 prompts x 8 rollouts per update (Kaggle: 32 x 4; same 128 completions). G=8 gives MEG
  enough rollouts per group to resolve modes.
- `lora_dropout` 0, so the update is exactly on-policy (old = new.detach()). On Kaggle,
  dropout 0.05 made old/new log-probs differ randomly.
- Scoring uses the exact sampled token ids (no re-tokenization of the text), fp32
  log-softmax, and chunked backward (tested: identical gradient to full-batch).
- Sampling is fully specified: Qwen's generation_config had silently added `top_k=20`,
  `repetition_penalty=1.05` to every Kaggle run; now `top_k` is off and the penalty is 1.0.
- **Two answer-scoring bugs fixed (both also present in the Kaggle notebook):**
  1. `\boxed\{(.*?)\}` truncated nested answers (`\boxed{\frac{1}{2}}` -> `\frac{1`), which
     would have corrupted MATH-500 scoring. Now the last balanced `\boxed{}` is used.
  2. `50%` was converted to 0.5, so a correct GSM8K percent answer vs ground truth `50`
     scored wrong. Now `%` is stripped.
- **Safety fix:** `sympy.parse_expr` `eval()`s its input, and that input was raw model output
  with no timeout. Input is now whitelisted and capped at 2 s, so a garbage answer like
  `9^9^9^9` cannot hang a paid run.
- Final eval: a fixed 500-problem subset x 32 samples, k up to 32, identical for base and
  every run. Per-problem correctness matrices are saved.

**Verification done (all local, $0):** 33 CPU tests pass. They port the notebook self-tests
and add the parser, injection guard, scoring-mask alignment vs a naive reference,
chunked-gradient equivalence and checkpoint round-trip. A real Qwen2.5-0.5B end-to-end run on
CPU confirmed: correctness 0.38 at step 0, the HEG gate down-weighted 37.5% of rollouts, grad
norm 0.35, and KL to base rose 0 -> 0.0031 after one update (weights genuinely move).
Resume-from-checkpoint, the config-mismatch guard, merged-adapter eval and the queue runner
were all exercised. `analyze.py` reproduced a known synthetic answer.

**Open decisions (the pilot decides them, not guesses):**
1. **Learning rate.** The spec default 1e-6 is low for LoRA (LoRA usually needs 1e-5 or
   higher). If the policy barely moves there is no shrinkage to measure.
   `queues/pilot.txt` runs 1e-6 vs 1e-5; `pilot_report.py` checks whether KL to base rises.
2. **Steps per run.** `pilot_report.py` converts measured seconds/step into $/run and the
   number of runs that fit in 85% of $200.
3. **The headline comparison needs 1.5B.** Req 33.4/39.8 forbid `o_self`/`h_cb_grpo` at
   0.5B, so HEG-GRPO vs H-CB-GRPO (the comparison the paper rests on) cannot be run at 0.5B
   as specified. Either budget the 1.5B queue (`queues/main_1.5B.txt`, ~2-3x per-step cost)
   or deliberately revisit that spec rule (its stated reason is "N2 techniques only validated
   at 1.5B+", a scoping choice, not a memory limit).

---

## 8. Third novelty check + design fixes (2026-10-03)

Full report: `claude science/novelty_verification_2026-10-03.md`. Summary:

- **Novelty**: all cited IDs verified real (full-text reads). New close prior art: Cue-GRPO
  (2608.03467) = MEG's mechanism; ReCo (2607.26862) = two complementary *within-prompt* levels;
  MT-GRPO (2602.05547), CurveRL (2605.24331), BBG (2606.15455) = cross-prompt reweighting. The
  surviving claim is the **complementarity test of a latent topic-level budget and intra-group
  credit redistribution** at SLM scale. Revised abstract in the report.
- **Corrections to the 23 Sep notes**: the 2606.15455 zero-success filter is a current-group bucket
  needing a signed REINFORCE loss (zero gradient under GRPO); 2607.02869 used the 0.5B *base* model;
  real SELF = greedy-failure selection.
- **Bottlenecks found and fixed in `heg_grpo_aws/`** (user decisions in brackets):
  1. O-SELF's EMA never fired (~1.7 visits/prompt vs ~12 needed) so `h_cb_grpo == cb_grpo`
     -> real SELF greedy-failure gate, greedy answer from pinned argmax rows in the same generate
     call [real SELF, allowed at 0.5B].
  2. GSM8K has ~0.7% multi-approach problems (2606.29985) -> train on MATH, macro topics = 7 subjects
     [switch to MATH].
  3. MEG was down-weight-only (confounded with a lower positive lr) -> mean-preserving rarity credit
     redistribution (Cue-GRPO rule, alpha 0.8) + random-partition control [credit redistribution].
  4. Embedding modes may be phrasing -> pre-registered calibration (embedding vs bigram) on base rollouts.
  5. Inert format reward -> correctness-only, Math-Verify, standard Qwen math prompt.
  6. Mode entropy only logged for MEG runs -> diagnostics for every condition, plus mean positive
     gate weight and group dumps.
  7. N4 underpowered -> `PREREGISTRATION.md`: P1 heg vs h_cb, P2 heg vs meg (Holm), interactions as
     bounds; `analyze.py` seed-paired contrasts with a seeds x problems bootstrap.
  8. Eval cost unmeasured -> `--eval_probe` in the pilot; `pilot_report.py` budgets from it.
  Extras queue (if budget allows): random-partition control + BBG-style baseline.
- **Spec deviations** (recorded in `heg_grpo_aws/PREREGISTRATION.md`): Req 4 (dataset), Req 8 (SELF),
  Req 13 (format weight 0), Req 33.4/39.8 (0.5B rule removed), Req 44 (MEG rule).
- **Verification**: 47 CPU tests (incl. greedy rows == do_sample=False, Math-Verify on MATH answers,
  BBG utility vs the paper's Table 4, analysis helpers vs reference); analyze.py recovers a known
  synthetic answer; real-model CPU end-to-end on MATH for every condition (see changelog).

---

## 9. Changelog

- **2026-09-22** — Reviewed existing 0.5B CB-GRPO/Vanilla training logs and eval doc;
  found the "vanilla didn't learn" claim contradicted by the vanilla run's own reward
  curve (§2.1); found and fixed a `parse_output` fallback inconsistency between cell 14
  and cell 45 (§2.3). Implemented HEG-GRPO (Mode-Entropy Gate + composite, Requirement 44)
  end-to-end in `rlvr_training_pipeline_fixed.ipynb`; verified offline; launched an 80-step
  smoke-tier validation run on Kaggle 2×T4 (0.5B, `heg_grpo`).
- **2026-09-22 (later same day)** — Ran a novelty check on MEG/HEG-GRPO against
  arXiv/Google Scholar. Found MEG's core intra-group mechanism is not novel — GCPO
  (2605.11461), DRA-GRPO (2505.09655), and EDAS (2605.17333) already do correctness-aware,
  intra-group, embedding/entropy-based redundancy reweighting (§3.0). Repositioned the
  paper's claim from "novel gate mechanism" to "novel hierarchical composition of a
  cross-topic macro budget with this emerging intra-group diversity-gate family" — no code
  changes, `MEGGate`/`heg_grpo` implementation unchanged. Updated
  `novelty_upgrade_addendum.md` §2 (added the three papers, verified real via direct arXiv
  fetch), §4.4 (positioning correction), and §11 (rewrote the draft contribution statement
  and abstract-framing guidance) accordingly.
- **2026-09-23** — Reviewed the first real GPU output from the `heg_grpo` smoke-test
  (step 0 and step 5 log lines, §5). Reward signal confirmed alive and the `\boxed{}`
  fallback confirmed working end-to-end on GPU (§5.1). Found and fixed two bugs in cell 45:
  (1) `mean_group_mode_entropy` was computed but never logged (§5.4); (2) `Accelerator()`
  was missing `gradient_accumulation_steps`, so gradient accumulation across the 4
  micro-batches was silently not happening and the LR schedule was advancing up to 4x too
  fast (§5.3) — **this likely affected every prior run under this pipeline with
  `grad_accum_steps > 1`**, re-check before citing any such run. Also flagged (not yet
  fixed): at the observed ~5.5 min/step, `standard` tier (800 steps) projects to ~73h,
  infeasible on Kaggle as currently configured (§5.2) — needs a wall-clock decision before
  that tier is launched. The currently-running Kaggle smoke test predates both code fixes;
  re-launch after pulling the updated notebook to get correct gradient accumulation.
- **2026-10-02** — Moved training to AWS ($200 credits). Built `heg_grpo_aws/`: a standalone
  single-GPU bf16-LoRA package with a runbook, setup script, unattended queue runner (resume,
  S3 sync, auto-shutdown, stop after 2 failures), pilot go/no-go report, final-eval and
  analysis scripts, and a JupyterLab monitor. Fixed nested-`\boxed{}` truncation, `%` scoring,
  and unguarded sympy `eval` (all three also present in the Kaggle notebook; not back-ported).
  Verified with 33 CPU tests plus a real-model CPU end-to-end run (§7). Next: AWS console
  setup, then the pilot.
- **2026-10-03** — Third novelty check with the scholar connectors (§8; report in
  `claude science/novelty_verification_2026-10-03.md`): claim narrowed to the topic-level x
  intra-group complementarity test. Found O-SELF never fired in the AWS config and that GSM8K leaves
  MEG no real approach diversity. Implemented: MATH training (subjects = macro topics), real SELF
  allowed at 0.5B, MEG as mean-preserving credit redistribution with calibrated partitioner and
  random-partition control, BBG-style baseline, correctness-only reward with Math-Verify,
  diagnostics for all conditions, pre-registration with Holm-corrected primaries and a seeds x
  problems bootstrap, measured eval cost in the pilot. Next: AWS setup, pilot, fill the `[pilot]`
  values in `heg_grpo_aws/PREREGISTRATION.md`, main queue.
- **2026-10-04** — Pre-AWS rehearsal on WSL Ubuntu 24.04 (same OS as the AMI): the real
  `setup_instance.sh` in a fresh home, then `start_jupyter.sh` and the control notebook's own cells
  driving a CPU-sized pilot queue through tmux and `run_queue.sh` in HF offline mode, then every
  monitor-notebook cell. Fixed what it found: `math-verify` was missing from `requirements.txt`
  (a fresh install would have silently used the weaker sympy checker on MATH; the test now fails
  instead of skipping); the hf-xet download client stalled the model prefetch indefinitely
  (prefetch now uses plain HTTP with retries: 44 s instead of a hang); `train.py` now refuses
  `meg`/`heg_grpo` without a calibration file instead of silently using the uncalibrated
  partitioner; the pilot budget now includes the main runs' in-training monitor evals; the pilot
  launch arms a 4 h power-off; stale README pointers in the scripts and notebook fixed. AMI: the
  "Base OSS Nvidia Driver" AMI is no longer listed; use "Deep Learning Base AMI with Single CUDA
  (Ubuntu 24.04)", x86 (20260721: driver 595.71, G5 supported; PyPI torch 2.13 is CUDA 13.0 and
  needs >= 580). Setup now installs the AWS CLI if the AMI lacks it.
- **2026-10-05** — Paper reframed around three research gaps (evidence: `claude science/research_gaps_2026-10-05.md`):
  1. **contradiction.** Is Pass@k shrinkage real? Yue/Nguyen/Wu&Xuan say it shrinks, ProRL/Wen say it
     expands, Yao says both, Yuan/Dragoi say it is partly a measurement artifact.
  2. **methodological.** All evidence is automatic. CoT-Pass@k judges accept corrupted chains (2609.32622), and
     approach diversity was judged by an LLM calibrated on 80 human pairs (2606.29985), so nobody has asked
     people what RLVR loses.
  3. **novelty.** The composition test, unchanged.

  Added a pre-registered, blinded two-annotator human audit:
  - **H1:** partition validity on 60 solution pairs (κ, balanced accuracy, 70% bar).
  - **H2:** valid-reasoning rate of base successes on exited vs difficulty-matched retained problems
    (Fisher test, pre-registered lucky / genuine / mixed reading).

  Code changes:
  - final evals now save all completions (`*_completions.jsonl.gz`), which is required for H2 and so had to
    land before the main runs;
  - `scripts/human_eval.py` (sheets and scoring) + `HUMAN_EVAL_GUIDE.md`; `PREREGISTRATION.md` dated addition;
    a `control.ipynb` step;
  - 50 tests pass; real-model eval checked writing completions.

  Paper: new title, abstract, introduction and related work (8 new arXiv references, verified from abstract-page
  citation metadata); a human-audit protocol and results; a discussion that answers the debate. Algorithm 1 was
  removed for space. The paper is still 6 pages.
- **2026-10-06** — First AWS pilot (A10G).
  - **Calibration (96 prompts, base correctness 0.378):** the pre-registered rule chose word-bigram at
    threshold 0.4 (median 3 modes, 90% multi-mode). Embedding τ=0.85, the old default, gives a median of 1 mode
    (12% multi-mode), which confirms on real data that the old MEG was near a no-op.
  - **OOM.** Both 30-step pilots ran out of memory at step 3, in backward with 4 sequences per chunk once
    completions reached 1,024 tokens. Steps took 146 s, 89% of it generation.
  - **Fix**, from measured benchmarks:
    - 16 prompts in one generate call (78.9 vs 2 × 64.6 ms per decode step);
    - rollouts sampled from a bf16 merged copy of the LoRA model (61.5 ms);
    - backward chunks of 2;
    - eval 8 problems per call;
    - expandable CUDA segments.
    The median step is now ~89 s.
  - **A runtime "merged copy carries the adapter" check false-alarmed twice.** Early on, the update is below
    bf16 noise. Verified on the step-15 adapter scaled ×1/×10/×100: per-token KL from the exact fp32 policy is
    4.9e-3 / 9.3e-3 / 6.8e-3 for the merged copy vs 5.3e-3 / 7.9e-3 / 4.4e-3 for the bf16 LoRA path, and
    1.2e-1 for the base model at ×100. The merged copy, and therefore the bf16-merged final eval, keeps
    realistic updates. The check is now log-only.
  - **CB early read:** spend EMA across subjects is 0.023–0.146 at step 15; only one subject is above
    1.5× the mean, giving ~0.2% throttling. This confirms the open "CB strength" decision.
- 2026-10-06 (later): **pilot finished, main runs launched.**
  - Pilot: heg_grpo at lr 1e-6 and 1e-5 (30 steps each), o_self (10 steps). Every check passed.
    Median step 89-95 s; final eval 14.8 s per MATH-500 problem (3.08 h per model with GSM8K).
    Zero-variance groups 0.51; truncation 13%; SELF skips 38-50% of prompts.
  - The logged "KL to base" sits at a bf16 noise floor (~2e-4) at either lr, so it says nothing in a short
    pilot. The lora_B norm is 10x larger at 1e-5, as expected. `pilot_report.py` now reports the adapter
    norm and treats KL as movement only above 1e-3.
  - CB never fired (subject spend ratios 0.38-1.46 at step 30). Changed before any main run (pre-registration
    deviation 6): θ = 1.1, δ = 0.18, positive-advantage rollouts only, mean-preserving. A 4-step smoke run
    down-weighted 10-22% of rollouts from step 1 with the mean positive weight at 1.000. 52 tests pass.
  - Decisions (user): lr 1e-5, 240 steps, levels 1-5, 3 seeds, full eval for every model, one instance.
    Projected ~170 GPU-h, about $171, about 7 days. Extras only if money is left.
  - Launched `queues/main_0.5B.txt` at 15:45 UTC in tmux `train`: `--train-args "--steps 240 --lr 1e-5"
    --max-hours 190 --shutdown-when-done`, S3 sync on.
- 2026-10-09: **main runs finished, analysed, paper filled.**
  - 16 runs: seeds 0-1 for all six conditions, seed 2 for heg_grpo, h_cb_grpo, meg and vanilla. Instance B was
    lost on 2026-10-07 (seed 1, step ~85: memory blow-up in answer checking, now guarded in a capped worker);
    the replacement resumed from the step-75 checkpoints.
  - Result: no shrinkage at 120 steps (~2 rollouts per training problem). Vanilla Pass@1 30.3 -> 30.4%,
    Pass@32 73.0 -> 75.3%, slope +0.66 pp/ln k; every condition has more entries than exits.
    P1 = -0.02 [-0.79, +0.68], P2 = +0.22 [-0.60, +1.05] (bootstrap, pp/ln k), Holm p = 1.0: bounded null.
  - Mechanism (scripts/mechanism.py, descriptive): more distinct correct modes went with MORE exits
    (rho +0.57, p 0.02, n 16); topic credit share does not predict topic Pass@32 change (rho +0.01, n 28).
  - Results copied to `results_0.5B/` (no checkpoints) and S3. Human-audit sheets built
    (results_0.5B/human_eval/annotator_A, _B); scoring waits for two annotators.
  - Paper: numbers from fill_numbers.py --artifacts ../results_0.5B --gpu_hours 86 --usd 87; R1-R18 rewritten for
    this outcome; human-audit sentences keep TODOs until the labels are in. Both instances stopped.
- 2026-10-09 (later): **human audit scored.** Files: results_0.5B/human_eval/annotator_A_labeled/ and
  "h1_pairs_labeled B.csv", "h2_solutions_labeled B.csv" (annotators are not authors). Scored with
  scripts/human_eval.py (now also writes per-annotator and consensus counts) into analysis/human_eval.json.
  H1: kappa 0.13; humans said SAME for 50 of 51 consensus pairs, so the partitioner's modes are mostly surface
  clusters. H2: kappa -0.06, inconclusive. Paper updated (abstract, V-D, discussion, limitations, conclusion), 6 pages.
- 2026-10-09 (evening): adopted the verified points of claude science/results_reframing_2026-10-09.md:
  PABAK and an always-same baseline for H1, contrast CIs restated as k=1->32 gain (slope x ln 32), interactions
  reported as open, BH-adjusted p for the descriptive correlations (0.16, not significant), between-seed slope SD
  (vanilla 0.04 vs MEG 0.40), training-intensity reading and the partition-as-weak-link discussion. Not adopted:
  the claim that the H1 bar was "not met" (balanced accuracy is 77%, reported with its caveat), an unverified EDAS
  statistic, and treating the intensity reading as confirmed (one intensity level was run).
- 2026-10-09 (night): humanizing pass on the user's Overleaf edit of main.tex; full-width Fig. 1 (training step);
  6 pages. Corrected errors in that edit: steps 240 -> \R{meta.steps} (120, cut from a planned 240, not 240 from 580),
  reproducibility statement restored (pre-registration first committed after the runs, 4cdc43a), AI disclosure made
  accurate. Added main_blind.tex (double-blind build: no authors, acknowledgment, annotator names, commit hash or repo link).
- 2026-10-09 (night): **reviewer-style feedback acted on, honestly.** Asked to predict 5-seed and 240-step results into the
  paper; declined, because predicted results are not results. Instead:
  - Done now with real data: equivalence reporting (`analyze.py --sesoi`; P1 and P2 are inside +-1 pp/ln k at 90%, S1 and S2
    are not) and a seed-planning table (`scripts/power.py`: with the observed between-seed sd, 3 seeds detect ~1.4 (P1) and
    ~2.0 (P2) pp/ln k, 5 seeds ~0.7 and ~1.0). Both are in the paper (R19).
  - Built for the future: `train.py --snapshot_steps` (adapter snapshots evaluated after training, `eval_step<N>/`),
    `analyze.py --eval_subdir`, `queues/extension_seeds_0.5B.txt` (14 runs, ~$76) and `queues/extension_240_0.5B.txt`
    (12 runs, ~$150). Not run. A trajectory over 120/180/240 steps cannot be shown until those runs exist.
  - Predictions F1-F4 written into PREREGISTRATION.md before any extension run, with refutation criteria.
  - Environment note: the old scratchpad venv was hollowed out (pandas, dill, matplotlib directories without files);
    tests now run in a fresh venv (math-verify, pymupdf). 55 tests pass.
