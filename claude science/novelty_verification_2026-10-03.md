# Novelty verification, 3 Oct 2026 (third pass)

Follow-up to `novelty_verification_2026-09-23.md`, run because that report said its result was
"good for roughly a month" and asked for a re-check of the composition claim.

**Method.** alphaXiv discovery searches across four axes (pass@k / reasoning-boundary mitigation
with recency ranking; RLVR papers since 2026-09-01; cross-prompt topic/domain/cluster reweighting;
multi-level prompt x response reweighting), plus **full-text reads of 12 papers** (not abstracts),
including every paper the 23 Sep report relied on. Like the earlier passes, this is keyword- and
index-driven, so it lowers the probability of a scoop but cannot prove absence.

---

## 1. Verdict

**The claim survives, narrower again.** The intra-group mechanism has more prior art than the
23 Sep report knew, and the cross-prompt side now has three neighbours. What nobody has done is
combine a **latent topic-level** gradient budget with **intra-group** credit redistribution and
test whether the two are **complementary**. That test, at SLM scale with diagnostics, is the
entire novelty surface.

| Claim | Status (3 Oct) |
|---|---|
| MEG's intra-group mechanism is novel | **Dead, confirmed again.** Cue-GRPO (2608.03467) is the same mechanism: rarity-aware redistribution of positive credit over a partition of the correct rollouts, incorrect rollouts untouched, mean-restored. UARL (Hu et al. 2026, judge-built partitions) is the same idea. |
| "Two complementary levels" is new | **Must be qualified.** ReCo (2607.26862) already shows two *complementary* corrections, but both are *within one prompt* (response level and token level). Only the cross-prompt *topic* level x intra-group pairing is unoccupied. |
| Cross-prompt reweighting is new | **No.** MT-GRPO (2602.05547, ICML 2026) reweights explicit *tasks* for worst-task accuracy; CurveRL (2605.24331) and BBG (2606.15455) reweight *individual prompts* by pass rate. CB-GRPO's *latent topic* budget on positive advantage mass is still distinct. |
| Topic-level budget x intra-group redistribution, evaluated for complementarity | **Holds.** Not found in 12 full reads plus the 126 papers screened on 23 Sep. |

## 2. Citation check

All IDs relied on are real, and the claimed titles match the papers (read in full text):
`2510.02230` (SELF), `2605.11461` (GCPO), `2609.08650` (DATPO), `2608.27351` (ES), `2607.20543`,
`2606.15455`, `2607.01490` (FADE), `2607.02869`, `2601.03703` (TreeAdv), `2607.26862` (ReCo),
`2608.03467` (Cue-GRPO), `2602.05547` (MT-GRPO), `2605.24331` (CurveRL), `2606.29985`,
`2609.39321`. (The Scholar Sidekick citation verifier was unavailable: not subscribed.)

## 3. Corrections to the 23 Sep report and the forecast

1. **The zero-success baseline is not "OSELFGate at its limit".** 2606.15455 restricts updates to
   problems with zero successes *in the current rollout group* (bucket B0/8), and makes that bucket
   learn through a **signed REINFORCE loss** (r in {-1,+1}). Under GRPO's group normalization an
   all-wrong group has zero advantage, so that filter yields **no gradient at all** in this
   pipeline. Their full method, BBG, is a problem-level gate by success count; the AWS package adds
   it as `bbg` in a GRPO-compatible form (without the B0 term), labelled as such.
2. **2607.02869 trained Qwen2.5-0.5B *base*, not Instruct** (53.75% outcome-only GSM8K, single seed,
   no CI). Use it only as a loose anchor.
3. **SELF (2510.02230) selects problems whose *greedy* answer fails** (plus a forward-KL term), not
   an EMA solve-rate. The project's `o_self` was a project-specific variant; see Section 5.

## 4. New prior art and required differentiation

| Paper | What it does | How this project differs | Where to cite |
|---|---|---|---|
| **Cue-GRPO** 2608.03467 (+UARL) | Rarity credit redistribution over partitions of correct rollouts; deterministic "strategy cue" partitions; random-partition and mean-matched ablations | We adopt its rule for the micro term (and say so); our contribution is the macro level and the complementarity test | Method (micro term), related work, abstract sentence 2 |
| **ReCo** 2607.26862 | Normalizes response contributions by expected occurrence + variance-based token ratio; reports the two as complementary | Its two levels are within one prompt; ours are cross-prompt topic x intra-group | Abstract sentence 2, related work |
| **MT-GRPO** 2602.05547 | Task-level weights for worst-task accuracy, ratio-preserving sampler | Explicit tasks, robustness objective, sampler-based; ours: latent topics, positive-advantage budget, multiplicative gate, Pass@k objective | Related work (macro family) |
| **CurveRL** 2605.24331 | Prompt-level weights from the pass-rate quantile distribution | Per-prompt difficulty, not topic; no intra-group term | Related work (macro family) |
| **BBG** 2606.15455 | Problem-level gating by success count; shrinkage as "overtraining" | Different axis (problem success), and argues aggregate Pass@k decline partly reflects few rollouts per problem: report rollouts/problem with every curve | Related work, N1 framing, baseline `bbg` |
| **2606.29985** | Embedding cosine is the weakest proxy for approach diversity; bigram overlap does better; **only 0.7% of GSM8K problems admit >1 approach, 33% on MATH** | Motivates moving training to MATH and calibrating the partitioner (embedding vs bigram) | Method (partitioner), limitations |
| **2609.39321** | GRPO training dynamics for 1.5B-7B SLMs, pass@5 only, no mitigation | SLM-scale neighbour; we study shrinkage and its mitigation | Related work |

## 5. Bottlenecks found in this pass, and what was changed (AWS package, 3 Oct)

| Bottleneck | Consequence | Fix |
|---|---|---|
| O-SELF's EMA (alpha 0.1, tau 0.7) needs ~12 perfect visits; the AWS config visits each prompt ~1.7x | `o_self` never fires, so `h_cb_grpo` == `cb_grpo` and "HEG vs H-CB" compares nothing | Real SELF: greedy-failure selection each step; the greedy answer rides in the same generate call (pinned argmax rows), at almost no cost |
| GSM8K has ~no multi-approach problems (0.7%) | MEG's "modes" on GSM8K are mostly wording | Train on MATH; its 7 subjects become the macro topics (removes the arbitrary K=16) |
| MEG only scaled weights down | HEG partly acted as "lower learning rate on correct answers" | MEG = mean-preserving rarity redistribution (Cue-GRPO rule, alpha 0.8); `mean_pos_weight` logged for every condition |
| Embedding partitions track phrasing (EDAS, 2606.29985) | Modes may not be strategies | Partitioner chosen by a pre-registered calibration on base rollouts (embedding vs bigram); random-partition control run; group dumps for a manual audit |
| Format reward identically 0 | Inert component | Correctness-only reward; standard Qwen math prompt |
| Mode entropy logged only for MEG runs | Vanilla's collapse invisible | Diagnostics logged for every condition |
| Interaction underpowered at 3 seeds | CI likely straddles 0 | Pre-registered primaries P1/P2 (pairwise, Holm), interactions as bounds; seed x problem hierarchical bootstrap; `PREREGISTRATION.md` |
| Eval cost unmeasured | Budget guesswork | `--eval_probe` in the pilot; the pilot report budgets from measured eval time |
| 0.5B restriction on the SELF family | Primary contrast impossible at the affordable size | Lifted (scoping choice, not memory) |

## 6. Revised abstract (replaces the 23 Sep draft)

> Reinforcement learning with verifiable rewards (RLVR) raises single-sample accuracy while often
> shrinking Pass@k. Recent methods attribute this to concentration of credit, either within a
> rollout group (credit redistribution over solution modes: Cue-GRPO, ReCo, GCPO, DRA-GRPO, EDAS)
> or across prompts (problem- or task-level reweighting: BBG, CurveRL, MT-GRPO). We ask whether
> these act on distinct components of shrinkage: does a budget on positive gradient mass across
> latent topics complement intra-group credit redistribution, or duplicate it? In a pre-registered
> 2x3 factorial on Qwen2.5-0.5B trained on MATH (macro {off, topic budget} x micro {off, per-prompt
> SELF filter, intra-group redistribution}, 3 seeds), we find [RESULT], and use mode-entropy,
> boundary entry/exit and interference diagnostics to explain what each level does. The
> contribution is not a new gate but the first complementarity test of topic-level and intra-group
> interventions, at a scale and compute budget where a bounded null is as informative as a gain.

## 7. Reviewer objections to pre-empt, in order of danger

1. "MEG is Cue-GRPO." Agree in the paper, in the method section: we use their rule on purpose.
2. "Your shrinkage is an artifact of few rollouts per problem" (2606.15455). Report rollouts per
   training problem, boundary entry/exit counts, and the `bbg` baseline.
3. "Your modes are phrasing." Calibration rule, random-partition control, audit samples.
4. "Three seeds." Pre-registration, Holm, hierarchical bootstrap, bounded-null framing.

## 8. Re-search before submission

Re-run at submission time, and specifically re-check (a) any topic-/domain-level x intra-group
composition, (b) follow-ups to Cue-GRPO / ReCo / MT-GRPO, (c) GRPO shrinkage studies at <= 1.5B.
