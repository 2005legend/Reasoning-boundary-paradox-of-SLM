# Novelty verification, 23 Sep 2026

Independent re-check of the 2026-09-22 positioning decision, run because that check
self-describes as "one focused session, not a systematic review" and its own hygiene note
asks for three specific follow-up searches before any novelty claim is finalized.

**Method.** 22 arXiv queries across the three axes the hygiene note names (MEG's intra-group
mechanism; the macro x micro composition claimed as the gap; Pass@k inversion at SLM scale),
plus 7 exact-title verification lookups and a date-bounded sweep of Jun-Sep 2026.
**126 unique papers screened.** Search is arXiv-only and keyword-driven, so this lowers the
probability of a scoop but cannot prove absence.

---

## 1. Verdict

**The repositioned claim survives, but it is narrower than it was yesterday.**

| Claim | Status |
|---|---|
| "MEG's intra-group mechanism is novel" | **Dead** - already abandoned on 09-22. Confirmed correctly abandoned; GCPO's abstract explicitly frames prior entropy-regularization / diversity-bonus work as the space MEG sits in. |
| "Hierarchical composition of a cross-prompt topic budget with an intra-group diversity term is unoccupied" | **Holds.** Nothing in 126 papers does this. The four dedicated composition probes - `two-level AND reinforcement learning AND reasoning AND diversity`, `GRPO AND prompt-level AND response-level`, `pass@k AND small language model`, `shrinkage AND reasoning boundary AND mitigat` - all returned **zero** hits. Two broader axis queries also bore on the gap: `multi-granularity AND verifiable rewards` returned zero, and `cluster-level AND reinforcement learning AND LLM AND reasoning` returned a single hit (PlexRL, `2605.20863`), which is serving/orchestration infrastructure for RLVR rather than a training-composition method and was screened out. |
| "...evaluated through a Pass@k-shrinkage diagnostic lens at SLM scale" | **Weakened.** Two papers now occupy parts of this: DATPO targets the same outcome metric with a two-axis method, and 2607.02869 runs the identical Qwen2.5-0.5B/GSM8K/GRPO setup. |

**Six papers were missed by the 09-22 check**, one of them serious. None of them kills the
project; one of them changes what the abstract may claim, and one adds a baseline the
six-condition matrix does not currently contain.

## 2. Citation verification

All seven load-bearing IDs on file resolve to the correct title, date, and ID - **no
hallucinated citations.** Verified by exact-title arXiv lookup:

`2510.02230` `2605.11461` `2505.09655` `2605.17333` `2605.00365` `2607.20543` `2604.06298`

(`2510.02230` is `cs.CL`, not `cs.LG` - it is real; my first sweep restricted category and
missed it. `2504.13837` and `2510.04028` were not re-verified this round; they were confirmed
on 09-22 and are not load-bearing for the novelty claim.)

## 3. The serious one: DATPO (arXiv:2609.08650, 8 Sep 2026)

*Difficulty-Adaptive Tree-Structured Policy Optimization for Expanding Reasoning Coverage in RLVR*

This is the closest thing to the project's gap that exists, and it was published **two weeks
before** the 09-22 search that missed it. It matters on three counts at once:

1. **Same objective.** It explicitly targets Pass@k expansion under RLVR, framed as the failure
   of RLVR to "expand the model's intrinsic reasoning coverage (pass@k)" - the project's N1/N4
   framing in different words.
2. **It has a micro diversity term.** A "sibling-diversity advantage term" promoting semantic
   diversity within a rollout group. That is MEG's function, reached by a different construction.
3. **It has a coarser second axis.** "Difficulty-adaptive rollout," which the paper argues is a
   Pass@k mechanism in its own right and "not just an efficiency heuristic."

So DATPO is already a two-axis method aimed at Pass@k. The composition claim therefore cannot
be "we are the first to compose two granularities." It must be the specific pair:

> a **corpus-level topic-cluster gradient-mass budget** composed with an intra-group diversity
> term - where the macro axis is *semantic topic coverage across the training distribution*,
> not per-problem difficulty, and the composition is a multiplicative gate on existing
> advantages rather than a modified rollout procedure.

That is still unoccupied, and it is honestly differentiable. Two further points in the
project's favour worth stating explicitly: DATPO changes the rollout *procedure* (tree search,
forking), which costs generation compute the project does not have; MEG/CB-GRPO leave
generation untouched and reweight what already exists. And DATPO's difficulty axis is
per-problem, which makes it a cousin of PBA's granularity, not of a topic partition.

**Uncomfortable corollary for the difficulty-aware clustering ablation (§5.4):** DATPO argues
difficulty-adaptive allocation is itself a Pass@k mechanism. That ablation is no longer a
neutral diagnostic about clustering basis - it is a partial replication of DATPO's principle 1.
Either cite DATPO there and frame it as a check of *their* claim at SLM scale, or drop it.

## 4. Also missed - add all five

| Paper | Why it matters | What to do |
|---|---|---|
| **FADE**, `2607.01490` (1 Jul 2026) - *Don't Let Gains FADE: Breaking Down Policy Gradient Weights in RL* | Decomposes any advantage into **positive and negative gradient mass** along a sign axis and a difficulty axis, and shows the sign imbalance "collapses either entropy or weight geometry." This is a formalization of exactly the insight behind refining CB-GRPO's spend from `abs(advantage)` to `max(advantage, 0)`. | Cite at the point where the spend-metric refinement is introduced. Reframe that refinement as *instantiating* a known decomposition in a cluster-level budget, not as an independent discovery. Cheap to do, and a reviewer who knows FADE will otherwise flag it. |
| **Reward Granularity in RLVR**, `2607.02869` (3 Jul 2026) | Qwen2.5-0.5B + GRPO + GSM8K - the project's exact setup. Reports GSM8K test accuracy of **53.75% outcome-only** vs **63.73% process-only**. | Two uses. (a) Near-neighbour citation, same obligation as `2604.06298`. (b) **External ceiling anchor**: ~54% outcome-reward accuracy at 0.5B means GSM8K is *not* ceiling-saturated at 0.5B, so GSM8K remains a legitimate primary benchmark there and the §5.4 ceiling worry applies mainly at 1.5B. That is a useful, citable justification for a design choice currently taken on faith. |
| **Evolution Strategies**, `2608.27351` (27 Aug 2026) | Shows ES attains **higher Pass@K than GRPO** where GRPO shows entropy collapse, links Pass@K to verifier-projected Jensen-Shannon population diversity, and proposes sequential GRPO-ES. | Related work, as the *non-gating* alternative route to the same goal. Pre-empts "why gate advantages instead of changing the optimizer?" Answer it in one sentence: gating is a drop-in modification compatible with a 2xT4 budget; ES is a different training paradigm. |
| **Diversity Collapse via Overtraining**, `2606.15455` (13 Jun 2026) | Known from `project_summary.md` but **absent from the verified `related_work.md`** - it fell out of the citation list that is actually being maintained. See §5; this is the strategic risk. | Add to `related_work.md` with differentiation text. Non-optional. |
| **TreeAdv**, `2601.03703` (7 Jan 2026) | Entropy-driven branching plus advantage redistribution across group rollouts. Another member of the intra-group advantage-reshaping family. | One line in related work, grouped with GCPO/DRA-GRPO/EDAS. Low risk. |

## 5. The strategic risk: 2606.15455 is both a missing baseline and a reframing threat

Its abstract makes two claims that bear directly on this project:

1. **A simpler intervention already beats the base model.** "Restricting updates to problems
   with zero observed success lifts Pass@256 above the base model." That is a one-line
   data-selection rule, and it is stronger than anything the six-condition matrix currently
   contains - O-SELF gates on an EMA solve-rate, which is a soft cross-step proxy for the same
   thing. A reviewer will ask why the headline method is a two-component composite gate when a
   hard zero-success filter reportedly clears the bar.
2. **It undercuts N1's interpretation.** It argues RLVR is *structurally* biased against high-k
   Pass@k under few rollouts per problem, so "its aggregate decline does not by itself mean
   that no new reasoning gains occurred." N1 as currently framed ("confirm shrinkage at SLM
   scale") measures something this paper says is partly an artifact of the measurement regime.

**Recommended response, and it is cheap.** Add a seventh condition: `zero_success_only`, a hard
filter reproducing 2606.15455's intervention. It needs no new code - it is `OSELFGate` with the
threshold at its limit. It converts the strongest objection into a baseline you beat, tie, or
lose to honestly, and under this project's own framing a loss is still publishable. Budget cost
is one condition x 3 seeds against 18 already planned.

For N1, reframe from "does shrinkage occur at SLM scale" to "**how much** of the observed
shrinkage at SLM scale survives the overtraining explanation" - and report rollouts-per-problem
alongside every Pass@k curve, since that is the quantity 2606.15455 says drives the artifact.

## 6. Revised abstract

Current draft leads with the method. Given §3-§5 it should lead with the diagnosis and place
the prior-art family in sentence two, as the 09-22 decision already requires. Suggested:

> Reinforcement learning with verifiable rewards (RLVR) improves single-sample accuracy while
> often *shrinking* Pass@k, and a growing family of methods attributes this to winner-take-all
> competition among rollouts within a group, addressing it with intra-group diversity
> reweighting (DRA-GRPO, GCPO, EDAS) or restructured rollouts (DATPO). We ask a question this
> family leaves open: whether shrinkage in small language models has a second, *corpus-level*
> component that intra-group methods cannot observe - concentration of gradient mass on a
> subset of problem topics. We evaluate a cluster-level positive-advantage budget (CB-GRPO), an
> intra-group mode-entropy gate (MEG) from the established family, and their multiplicative
> composition (HEG-GRPO), on Qwen2.5-0.5B/1.5B over GSM8K and MATH-500, across a saturated
> 2x3 factorial of macro and micro gates with N seeds per cell. We find [RESULT], and use
> interference (Delta+) and entropy diagnostics to characterize what each level does and does
> not address. Our contribution is not a new intra-group mechanism - that is prior art - but
> the first evaluation of whether topic-level and within-group interventions are complementary,
> conducted at a scale where the phenomenon is under-studied and under a compute budget that
> makes the negative result as informative as the positive one.

Two deliberate properties: the word "first" attaches only to the complementarity question, and
the last clause pre-commits to reporting a null result - which is the honest position given
that no results exist yet, and which reviewers reward rather than punish.

## 7. What each goal can claim, given evidence that exists today

No training results exist. This ladder is what to hold the writing to.

| Goal | Claimable now | Blocked on |
|---|---|---|
| N1 shrinkage at SLM scale | Nothing | Base vs trained Pass@k curves at matched rollout budget; must address 2606.15455's artifact argument |
| N2 mitigation transfer | Nothing | The o_self / meg cells |
| N3 capacity-shrinkage | Nothing, and **differentiate `2604.06298` before writing a word** | Both model sizes; only 3 of 6 conditions are cheap enough at 0.5B |
| N4 complementarity (headline) | The *design* is claimable: a saturated 2x3 factorial supports an interaction contrast, which is a stronger test than "composite beats both singles" | All six cells x >=3 seeds; bootstrap CI on the interaction, not on five pairwise comparisons |
| N5 diagnostics | Method is claimable | Checkpoints; also needs the cluster-spend Gini floor correction (size-Gini is 0.229 before any learning - see `smoke_startup_findings.md`) |

**Three objections a reviewer will raise, in order of danger:** (1) the zero-success baseline
from 2606.15455 (§5); (2) EDAS's own ablation finding that 76.4% of numerically-different wrong
answers had cosine similarity > 0.95 - their critique of embedding clustering applies to MEG's
correct-side clustering too, and needs a validation sampling MEG's mode clusters to show they
track distinct solution *methods* rather than phrasing; (3) 16 topic clusters is an unjustified
hyperparameter, and the partition is measurably imbalanced (sizes 203-954).

## 8. Re-search before submission

This check is good for roughly a month in this area - DATPO appeared 15 days before a search
that missed it. Re-run at submission, and specifically re-check whether anyone has published a
topic-level-plus-intra-group composition, which is now the project's entire novelty surface.
