# Pre-registration: HEG-GRPO ablation at 0.5B (AWS track)

Written 2026-10-03, **before any main run**. The pilot fills in the four values marked
`[pilot]` and nothing else. Any later change to this file must be dated and justified in the
changelog at the bottom, and reported in the paper as a deviation.

## Question

Is reasoning-boundary shrinkage under RLVR in a small model reduced more by combining a
**topic-level** gradient budget (CB-GRPO, macro) with an **intra-group** credit redistribution
(MEG, micro) than by either alone, or than by combining the macro budget with the base paper's
**per-prompt** filter (SELF)?

## Design

- Model: Qwen2.5-0.5B-Instruct, bf16 base weights + LoRA (r=16, alpha=32, all projections), single A10G.
- Training data: MATH train split (EleutherAI/hendrycks_math, 7 subjects), levels 1-5 (pilot: 51% zero-variance groups, below the 80% switch to levels 1-3).
  Macro topics = the 7 MATH subjects.
- Rollouts: G=8 per prompt, temperature 0.7, top-p 0.95, top_k off, max 1024 new tokens.
  Reward = binary correctness (Math-Verify, with a guarded sympy fallback).
- Update: 16 prompts (128 rollouts) per optimizer step, generated in one call from a copy of the current
  policy with the LoRA adapters merged in; on-policy GRPO with PPO-clip, token-level loss over the update,
  cosine LR with 10% warmup, lr 1e-5, 120 steps (see deviation 7; the pilot-based plan was 240).
- MEG partitioner: chosen on base-model rollouts by `scripts/calibrate_meg.py` with this fixed rule:
  among candidates (embedding cosine tau in {0.80,0.85,0.90,0.95}; word-bigram Jaccard threshold in
  {0.3,...,0.7}) whose median number of modes, over groups with >= 4 correct rollouts, is in [2, 4],
  take the one closest to 3; ties go to bigram, then to the larger multi-mode share. Result: word-bigram Jaccard,
  threshold 0.4 (median 3 modes, 90% of groups multi-mode; the embedding candidates gave a median of 1).
- Conditions (2x3 factorial, macro {off, CB} x micro {off, SELF, MEG}):
  `vanilla`, `o_self`, `meg`, `cb_grpo`, `h_cb_grpo` (CB x SELF), `heg_grpo` (CB x MEG).
  Seeds 0, 1, 2; seed s fixes data order and LoRA init across conditions (blocks for pairing).
- CB rule (revised 2026-10-06, see deviation 6): per subject, the EMA (alpha 0.05) of positive advantage
  mass; a subject whose EMA exceeds theta = 1.1 x the mean gets decay^(ratio - theta), decay = 0.18 (a
  subject at 1.5x the mean gets half the credit of an in-budget one). Applied to positive-advantage
  rollouts only, clipped to [0.3, 3], rescaled to mean 1 over the step's positive rollouts.
- Extras, run only if budget allows, exploratory: `heg_grpo` with random partitions (control),
  `bbg` (BBG-style problem gate, GRPO-compatible variant).

## Outcome

Final evaluation on a fixed problem set shared by every model: MATH-500 (all 500, in-distribution), 32
samples per problem, temperature 0.6, top-p 0.95, unbiased Pass@k for k in {1,2,4,8,16,32}.

**Primary outcome:** the shrinkage slope on MATH-500, i.e. the least-squares slope of
(Pass@k(run) - Pass@k(base)) against log k. Higher = less shrinkage.

## Hypotheses and contrasts (paired by seed)

| id | contrast | role |
|---|---|---|
| P1 | heg_grpo - h_cb_grpo | primary: intra-group redistribution vs per-prompt filter, both under CB |
| P2 | heg_grpo - meg | primary: does the macro budget add to the micro term? |
| S1 | heg - cb - meg + vanilla | secondary: CB x MEG interaction (complementarity), reported as a bound |
| S2 | h_cb - cb - o_self + vanilla | secondary: CB x SELF interaction, reported as a bound |
| S3 | mean(CB conditions) - mean(no-CB conditions) | secondary: macro main effect |
| E1-E3 | heg - vanilla; heg - heg randpart; heg - bbg | exploratory |

Also descriptive, no test: shrinkage of vanilla vs base (N1), with rollouts per training problem
reported alongside (arXiv:2606.15455 shows this drives apparent shrinkage).

## Analysis (implemented in `analyze.py`, fixed now)

- For each contrast, per seed: d_s = sum of coefficient x slope over its conditions.
- 95% CIs: (a) t-interval over seeds; (b) hierarchical bootstrap, 2000 resamples of seeds and
  problems jointly (same problem indices for all conditions and the base model).
- P1 and P2: two-sided bootstrap p-values, **Holm-corrected** across the two; claim an effect only
  if the Holm-adjusted p < 0.05 **and** the hierarchical CI excludes 0.
- S1-S3: reported as point estimate + 95% bounds ("the interaction lies within [a, b]"), never as a
  verdict on their own.
- Exploratory: estimates and CIs only, labelled exploratory; diagnostics (mode entropy over training,
  boundary entry/exit counts, mean positive gate weight, interference Delta+) are descriptive.
- Seeds: 3 planned. If the budget ends early, the analysis uses every condition with >= 2 completed
  seeds (the queue is seed-major), and the paper reports n.

## How each outcome will be reported

- P1 and/or P2 significant: the composition helps beyond its parts, with the interaction bound as context.
- Neither significant, CIs narrow: a bounded null ("the composition changes the slope by at most X"),
  explained with the diagnostics. This is a planned, publishable outcome.
- Vanilla shows no shrinkage on MATH-500 at this budget: reported as the N1 finding; contrasts still reported.

## Human evaluation (added 2026-10-05, before any main run)

**Why.** In the literature on shrinkage, every piece of evidence is automatic:
- Pass@k;
- CoT-Pass@k, whose LLM judges accept corrupted reasoning almost as often as clean reasoning (arXiv:2609.32622);
- entropy and embedding/bigram similarity;
- LLM judges calibrated on 80 human-labelled pairs (arXiv:2606.29985).

No study has had humans directly judge what RLVR loses. Two blinded checks, run with
`scripts/human_eval.py` and `HUMAN_EVAL_GUIDE.md`. There are two annotators, who do not see the condition,
the partitioner label or the group.

**H1: are the partitioner's modes strategies?**
- *Data.* 60 pairs of correct base-model solutions to the same MATH training problem, taken from
  `calibration_groups.jsonl`. 30 pairs are ones the chosen partitioner puts in the same mode and 30 in
  different modes, with at most 2 pairs of each kind per problem.
- *Labels.* `SAME` or `DIFFERENT` approach.
- *Reported.* Cohen's κ, and the accuracy and balanced accuracy of the partitioner against the annotators'
  consensus.
- *Interpretation rule.* If balanced accuracy is ≥ 0.70, MEG's partitions are described as approach-level.
  Otherwise MEG is described as redistributing credit over surface-level clusters, and the paper says so
  wherever it interprets MEG.

**H2: what does vanilla GRPO lose?**
- *Data.* MATH-500 problems the base model solves (at least 1 of 32 samples) but vanilla GRPO solves in
  0 of 32 in a majority of seeds ("exited"). Up to 40 are drawn, each matched to a retained problem (solved
  by vanilla in every seed) with the closest base solve count. One random correct base-model solution is
  shown per problem.
- *Labels.* `VALID` or `FLAWED` reasoning.
- *Reported.* The valid-reasoning rate per group (consensus labels) with Wilson 95% CIs, a two-sided Fisher
  exact test, and Cohen's κ.
- *Interpretation rule.*
  - If the exited rate is below 50% and below the retained rate (Fisher p < 0.05), shrinkage on these
    problems is reported as mostly a loss of lucky hits.
  - If the exited rate is ≥ 50% and not significantly lower, it is reported as a loss of genuinely solved
    problems.
  - Anything in between is reported as mixed.

Both checks are secondary and descriptive. Neither changes the primary contrasts (P1, P2). Items the
annotators disagree on are excluded from the rates and counted. A third annotator may adjudicate them;
if so, both versions are reported.

## Deviations from the original Kaggle-era spec (decided 2026-10-03, before any main run)

1. Training data GSM8K -> MATH (Req 4): only 0.7% of GSM8K problems admit more than one solution
   approach (arXiv:2606.29985), leaving the intra-group mechanism nothing to act on.
2. O-SELF EMA -> SELF greedy-failure selection (Req 8): the EMA never reached its threshold at
   ~2 visits per prompt, which made h_cb_grpo identical to cb_grpo.
3. 0.5B restriction on the SELF family removed (Req 33.4 / 39.8): it was a scoping choice, not a
   memory limit, and it blocked the primary contrast at the affordable model size.
4. MEG reformulated as mean-preserving rarity credit redistribution (Req 44), the Cue-GRPO rule
   (arXiv:2608.03467): removes the confound that a down-weight-only gate also lowers the learning rate.
5. Format reward dropped (Req 13): it was identically zero for every rollout, hence inert.
6. CB made a working, mean-preserving budget (decided 2026-10-06, after the pilot, before any main run).
   With the spec's theta = 1.5, decay = 0.98 and 7 subjects, the gate never fired in the pilot (subject
   spend ratios 0.38-1.46 after 30 steps), and could not cut a weight below 0.98^5.5 = 0.90 even if it
   did, so every CB condition would have equalled its non-CB twin. New values in Design. It now
   redistributes positive credit across subjects (mean 1) instead of only removing it, for the same
   reason as deviation 4. The old rule stays available (`cb_mean_preserving=False`).
7. Training length 240 -> 120 steps and the GSM8K evaluation dropped (decided 2026-10-06, before any main
   run, because of a paper deadline). The 240-step plan needed ~170 GPU-hours (about 7 days on one GPU).
   The runs are split across two identical g5.xlarge instances (`queues/main_0.5B_A.txt`, `_B.txt`),
   which needs ~100 GPU-hours. GSM8K was the secondary, out-of-distribution outcome and costs a third of
   every evaluation; no hypothesis depended on it. Everything else is unchanged.
8. Seed 2 only for the conditions in the primary contrasts (decided 2026-10-07, before any result was
   analysed or looked at beyond the training logs). The second instance was lost overnight: both seed-1
   runs were killed at step ~85 by the kernel when checking one model answer exhausted host memory
   (sympy expanding an enormous integer inside a single C call, which the 2 s alarm cannot interrupt).
   The symbolic part of answer checking now runs in a worker process capped at 3 GB and 10 s; an answer
   that hits either limit counts as wrong. This changes nothing for any answer that could be checked
   before (those runs crashed instead). The lost time leaves room for 15 of the 18 runs: seeds 0 and 1 for
   all six conditions, and seed 2 for `heg_grpo`, `h_cb_grpo` and `meg` (all P1 and P2 terms), plus
   `vanilla` seed 2 if it finishes in time. P1 and P2 use 3 seeds; S1-S3 and E1 use the seeds every one
   of their conditions has. The seed-1 runs resume from their step-75 checkpoints.

## Changelog

- 2026-10-03: written. Pilot values pending.
- 2026-10-05: added "Human evaluation" (H1 partition validity, H2 valid reasoning on exited vs matched
  retained problems). Final evals now also save every sampled completion (`*_completions.jsonl.gz`),
  which H2 needs. Added before any main run; no results had been seen.
- 2026-10-06 (pilot stage, before any main run): the first pilot ran out of GPU memory at step 3 and
  showed 146 s/step, 89% of it generation.
  - Changed:
    - 16 prompts per update in one generate call, instead of 8 x 2 micro-steps;
    - rollouts sampled from a merged copy of the policy;
    - backward in chunks of 2 sequences instead of 4;
    - eval generation 8 problems per call instead of 4.
  - Measured effect: decoding 78.9 -> 61.5 ms per step at 144 rows, and one call instead of two.
  - The objective, prompts per update, rollouts and evaluation settings are unchanged. Now the CB EMA updates
    once per optimizer step, and the token-level loss averages over all 16 prompts.
- 2026-10-06 (after the pilot, before any main run): filled the four pilot values (levels 1-5, lr 1e-5,
  240 steps, bigram 0.4) and added deviation 6 (CB rule). Pilot evidence: three runs (heg_grpo at lr 1e-6
  and 1e-5, 30 steps; o_self, 10 steps), all checks passed; median step 89 s; final evaluation 14.8 s per
  MATH-500 problem. No main-run result existed. The extras (random-partition control, BBG) do not fit the
  budget at 240 steps and run only if money is left.
- 2026-10-06 (16:30 UTC, before any main-run result): deviation 7 (120 steps, MATH-500 only, two
  instances). The 240-step queue had only started the shared base-model evaluation, which is reused.
- 2026-10-07 (03:30 UTC): deviation 8 (guarded answer checking; seed 2 only for the P1/P2 conditions,
  plus vanilla if time allows). Only training logs and run status had been seen; no evaluation result.
