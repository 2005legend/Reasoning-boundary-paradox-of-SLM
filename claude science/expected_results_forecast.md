# Expected results, HEG-GRPO / AWS track

Forecast built from the one measured quantity that exists (the 0.5B base Pass@k curve from the
23 Sep smoke run), the external literature anchors verified on 23 Sep, and the design described
in the 2026-10-02 change summary. Numbers below are predictions with falsification thresholds,
not results.

---

## 1. The base curve, and what it licenses

Measured base Pass@k, Qwen2.5-0.5B on GSM8K: `{1: 0.226, 2: 0.346, 3: 0.421, 5: 0.512, 10: 0.620}`.

Fitting per-problem solve probability `p ~ Beta(a, b)` (so `Pass@k = 1 - B(a, b+k)/B(a, b)`)
gives **a = 0.440, b = 1.481**, reproducing all five points to within **0.003** absolute. Two
parameters, five points, residuals at the third decimal - the curve is well described, so
extrapolation within the new eval range is safe.

| k | 1 | 2 | 4 | 8 | 16 | **32** | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| predicted base Pass@k | 0.229 | 0.345 | 0.471 | 0.588 | 0.687 | **0.766** | 0.826 | 0.871 | 0.905 |

**Expected result 1: GSM8K at 0.5B is not ceiling-limited under the new 500 x 32 eval.**
Predicted base Pass@32 = **0.766**, against the 0.90 ceiling flag; the flag should not trip
until k ~ 256. So GSM8K remains a legitimate primary benchmark at 0.5B and MATH-500 is the
harder secondary, not a forced substitute. (Caveat: the Beta model has no atom at p=0, so it
over-predicts as k grows - trust k <= 32, treat k=256 as indicative.)

**Expected result 2: the problem population is well-shaped for detecting shrinkage.** The fit
implies **32.8%** of problems are near-unsolvable (p < 0.05), **0.4%** near-always-solved
(p > 0.95), and **66.8%** sit in the boundary band 0.05 < p < 0.95. Shrinkage is a
boundary-band phenomenon, and two thirds of the eval set is in that band. If the measured
population were mostly saturated or mostly unsolvable there would be nothing to detect; it
isn't.

**Expected result 3: the inversion crossover should land near k ~ 16, which is why the eval
change matters.** Under the forecast band in §3 (Pass@1 rising to 0.40-0.55, Pass@32 falling
4-8 pp), the trained and base curves cross at **k ~ 16**. Below that the trained model looks
strictly better at every k; the inversion is only visible above it. The old eval setting of
10 samples per problem sat *entirely below the crossover* - it could not have detected
shrinkage even if shrinkage were present. The move to 500 problems x 32 samples is therefore
not a precision upgrade, it is what makes N1 measurable at all. Expect the k<=8 columns of
the results table to show uniform improvement and carry no evidence either way.

## 2. The caveat that outranks everything else

**Training per-rollout correctness at step 0 was `cor = 0.36`, while base eval Pass@1 measured
0.226.** A 13-point gap, in the wrong direction: single-sample eval accuracy should not be
*below* mean per-rollout training correctness on comparable problems.

Three scoring bugs were found and fixed **only in the AWS package**: the non-greedy
`\boxed\{(.*?)\}` truncating nested answers, `50%` converting to 0.5 so a correct GSM8K answer
of 50 scored wrong, and the unbounded sympy path. If the eval scorer carried any of them, the
base curve is **biased low** - and a base curve biased low inflates apparent expansion and
masks shrinkage, which is precisely the quantity N1 claims to measure.

**Action, highest leverage in the project: re-measure base Pass@k with the AWS scorer before
anchoring anything on it.** Expected outcome: the base curve shifts *up*. If base Pass@1 lands
near 0.30-0.36 rather than 0.226, every forecast below rescales, and the k=32 ceiling verdict in
§1 needs rechecking (base Pass@32 would rise toward 0.85, close enough to 0.90 to matter).

Secondary note from the same diagnosis: the launched script's `format_reward` returns 1.0 only
when reasoning, answer *and* boxed are all present - strict by documented intent, not a bug. But
`fmt = 0.00` across a whole step-0 batch means that component is **identically zero for every
rollout**, so it contributes no variance and therefore no gradient under a group-relative
objective. It is harmless to the learning dynamics (a constant cancels in the advantage) but it
is inert: the tag format cannot be learned from it, and any paper sentence describing a
"format + correctness reward" is describing something that is effectively correctness-only.
Either apply the partial-credit ladder or drop the component and say so in the setup section.

## 3. Expected magnitudes per condition

| Quantity | Expected | Basis | Tripwire: suspect the run, not the method |
|---|---|---|---|
| Pass@1 after a full run | **0.40 - 0.55** | `2607.02869` reports 53.75% for outcome-only GRPO on Qwen2.5-0.5B/GSM8K - same model, same benchmark, same algorithm family | Pass@1 below ~0.35, or not clearing base 0.226 by a wide margin |
| Pass@32 change vs base | **-4 to -8 pp** (5-10% relative) | shrinkage is the premise under test; `2606.15455` argues RLVR is structurally biased against high-k | Pass@32 *rising* by more than a few pp across all conditions points at the §2 scoring bias, not at expansion |
| Shrinkage slope (log-k regression) | base slope is **+0.172**; expect trained slopes **lower**, by 0.02-0.05 | derived from the measured curve | a slope *increase* in every condition |
| `mean_group_mode_entropy` | must appear in the log and be **non-trivial from step 0** | MEG is stateless, so it acts at step 0; the smoke run showed `gate_w = 0.764` before any state accumulated | field absent, or pinned at a constant |

On that last row: `gate_w = 0.764` at step 0 is consistent with MEG working, and lets you
back out roughly what it saw. Taking the Req-44 formula at face value with macro weight ~1.0 and
incorrect rollouts at weight 1.0, `0.764 = 0.64 x 1.0 + 0.36 x w_correct` gives
`w_correct ~ 0.344`, i.e. a normalized mode-frequency term of **~0.94** - correct rollouts
almost entirely collapsed into a single mode before any training. That would be a genuinely
quotable finding (the mechanism MEG targets is at near-maximum strength at initialization), but
it rests on a chain of assumptions about the logged aggregate. **Confirm it by reading
`mean_group_mode_entropy` directly** rather than citing the reconstruction.

## 4. The uncomfortable one: N4 is probably underpowered

Minimum detectable effect in Pass@32 points, for a bootstrap CI excluding zero at 80% power:

| seed-to-seed SD | n=2 pairwise | n=3 pairwise | n=5 pairwise | n=2 interaction | **n=3 interaction** | n=5 interaction |
|---|---|---|---|---|---|---|
| 1.0 pp | 2.0 | 1.6 | 1.3 | 4.0 | **3.2** | 2.5 |
| 1.5 pp | 3.0 | 2.4 | 1.9 | 5.9 | **4.8** | 3.8 |
| 2.0 pp | 4.0 | 3.2 | 2.5 | 7.9 | **6.5** | 5.0 |
| 3.0 pp | 5.9 | 4.8 | 3.8 | 11.9 | **9.7** | 7.5 |

An interaction contrast is a difference of four cell means, so its variance is 4x a single
condition's - the MDE is **2x** the pairwise one. Against a total shrinkage signal of only
3.8-7.7 pp, and an interaction that is by construction a *second-order* slice of that, three
seeds will most likely return a CI that straddles zero.

**This revises a recommendation I gave on 23 Sep.** I argued for the 2x3 interaction as the
primary contrast because it is the cleaner test of complementarity. That is true of the design
and false of the statistics at n=3. Better plan, in order:

1. **Make HEG-GRPO vs H-CB-GRPO the primary contrast.** It is a single pairwise comparison
   (MDE 1.6-4.8 pp at n=3), it is what the project already calls "the comparison that makes the
   paper", and it isolates exactly one variable: the micro term, direct vs cross-step proxy.
2. **Report the interaction as a bounded quantity, not a verdict.** "The macro x micro
   interaction is bounded within +/-X pp at 95%" is a legitimate, honest, publishable result and
   is what the data will actually support.
3. **If seeds can be bought, buy them for the four interaction cells only** (vanilla, cb_grpo,
   meg, heg_grpo) rather than spreading across six conditions. Going 3 -> 5 seeds cuts the
   interaction MDE by ~23%.

Pre-commit to this before the queue runs, so the choice is not made after seeing which contrast
happened to clear zero.

## 5. Design questions the change summary leaves open

1. **0.5B queue conditions: CONFIRMED** as `vanilla`, `cb_grpo`, `meg`, `heg_grpo` - the
   complete 2x2 macro{off,CB} x micro{off,MEG} sub-factorial. Consequences: the interaction
   contrast *is* computable at 0.5B (as a bounded quantity per §4), the macro and micro main
   effects are both identified, and HEG vs H-CB can only be run at 1.5B, where `h_cb_grpo`
   lives. The 1.5B queue therefore carries the paper's primary contrast and must not be the
   one that gets cut if budget runs short.
2. **Does the pilot budget include eval?** 500 problems x 32 samples is 16,000 generations per
   model. At 13 models (4 conditions x 3 seeds + base) that is ~208k generations of pure eval,
   separate from training. For scale, the smoke run's base Pass@k computation took **27.9 min** on 2xT4 (01:30:49 -> 01:58:43) reaching only k=10. Its sampled subset size and samples-per-problem are not recorded in the log - only that 1319 eval problems were loaded - so this is a wall-clock datapoint, not a per-generation rate; measure the AWS eval rate directly before budgeting.
3. **The `zero_success_only` baseline** from `2606.15455` is still absent from the queues. It is
   `OSELFGate` at its threshold limit, needs no new code, and defuses the strongest reviewer
   objection (§5 of the novelty report). One condition x 3 seeds.
4. **Kaggle results cannot be mixed with AWS results** - three scoring bugs differ between them.
   Either re-score the Kaggle checkpoints with the new parser or exclude them, and state which in
   the setup section. This also finally settles the gotcha-7 "vanilla didn't learn" artifact.

## 6. What a clean negative result looks like

Worth naming in advance, because this project's framing makes it publishable. A clean negative
is: shrinkage is confirmed at 0.5B (Pass@1 up, Pass@32 down, CI excluding zero vs base); each
gate individually reduces the shrinkage slope relative to vanilla; and the HEG vs H-CB contrast
plus the bounded interaction together show the macro and micro terms are **redundant rather than
complementary**, with the interference and entropy diagnostics explaining why. That is a
coherent paper. The outcome that is *not* publishable is an inconclusive CI on every contrast
with no diagnostic account - which is what §4 warns against drifting into by default.
