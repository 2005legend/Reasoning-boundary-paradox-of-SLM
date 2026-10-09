# What the results mean, and what the paper has to become

Verified against the reported tables before writing any of this: refitting the shrinkage slope
from the four published Pass@$k$ columns reproduces every reported slope to within **0.022**
(Vanilla 0.663 vs 0.66 reported, MEG 0.130 vs 0.13). The numbers are internally consistent.

---

## 1. The premise inverted, and the pre-registration is why that is survivable

Base Pass@32 = 73.0, Vanilla = 75.3, slope **+0.66**. All six conditions positive; only 2 of 16
runs negative, one of them essentially zero. There is no Pass@$k$ inversion here.

This is not a failed paper, because §Pre-registered analysis already committed to it: *"if
vanilla GRPO shows no shrinkage, that is itself reported and the contrasts are still given."*
That sentence, written before the runs, converts what would otherwise look like a salvage job
into a planned contingency. **Lead with it.** Reviewers of pre-registered work reward exactly
this, and it is the difference between "we found nothing" and "we specified in advance what
finding nothing would mean."

## 2. The interpretive move: this is about training intensity, not model scale

The honest claim is **not** "no shrinkage at SLM scale." The run was 120 steps, ~2 rollouts per
training problem, **0.26 epochs**, and final KL from base of 0.0028 against a measurement noise
floor of 2e-4 — about 14x noise. The policy barely moved, and Pass@1 moved +0.1 to +1.1 points.

That matters because it is precisely what the overtraining account predicts. Yuan et
al.~(`2606.15455`) argue the decline is driven by repeated reinforcement once a problem's
contribution has saturated; at 2 rollouts per problem nothing has saturated. So:

> Shrinkage is a function of **training intensity**, not of model size. At the rollouts-per-problem
> a 0.5B model on a single-GPU budget can reach, RLVR operates in a regime where it can only
> expand the boundary. The shrinkage demonstrations in the literature train orders of magnitude
> longer.

This makes the null a **confirmation of an existing theory under a new regime**, not an absence
of evidence — and it predicts something testable: shrinkage should appear if rollouts-per-problem
is pushed up, even at 0.5B. Say that explicitly; it is the paper's cleanest contribution to the
Yue-versus-ProRL debate, and it positions the result as evidence *for* the two-stage
account~(`2510.04028`) rather than against the inversion literature.

**Report rollouts-per-problem in the abstract.** It is the axis the whole interpretation rests on.

## 3. Your H1 conclusion is better supported than you stated

You reported $\kappa = 0.13$ and read it as annotator disagreement. That reading is wrong, and the
correction strengthens you.

| | raw agreement | reported $\kappa$ | PABAK | marginals (A / B) |
|---|---|---|---|---|
| H1 (partition validity) | **51/60 = 85%** | +0.13 | **+0.70** | 0.97 / 0.85 |
| H2 (lost-problem validity) | 15/40 = 37.5% | −0.06 | −0.25 | 0.28 / 0.70 |

H1's annotators agreed on **85%** of pairs. $\kappa$ is near zero because ~98% of the consensus
labels fall in one category — the textbook **kappa paradox**: with a degenerate marginal
distribution, $\kappa$ is attenuated regardless of how well raters agree. Report raw agreement
*and* a prevalence-adjusted statistic (PABAK = +0.70, "substantial" on the conventional bands),
and say why $\kappa$ alone is misleading here. Two humans looked at these solutions and agreed, at
substantial rates, that almost none of them use different approaches.

H2 is the opposite case and your "inconclusive" is right: raw agreement is **below chance**, and
the marginals differ by 42 points (A calls 28% valid, B calls 70%). That is a rubric failure, not
a sampling failure — the two annotators were applying different thresholds for "valid reasoning."
Worth saying plainly, because it is actionable for anyone repeating the audit.

**And the partitioner result is stronger than "close to a coin flip."** Against a consensus that
is 50 SAME / 1 DIFFERENT, a trivial always-SAME classifier scores **98%**. Yours scored **55%**.
It is not uninformative — it is *worse than the degenerate baseline*, because it splits groups
humans consider identical. State it that way.

## 4. This failure is now the paper's most valuable finding

The pre-registered bar was 70% balanced accuracy to read modes as approach-level. It was not met,
so by your own rule the modes are **not** approach-level. Combine that with the pilot result that
embedding partitioners put almost everything in one mode, and you have a complete negative result
about the instrument:

- **Embedding partitions under-split** — nearly every group collapses to one mode (your pilot).
- **Bigram partitions over-split** — ~3 modes per group that humans say are one approach (H1).
- **Neither tracks what the methods assume they track.**

This bears directly on a whole family of published methods that reweight by intra-group
partition — Cue-GRPO, GCPO, DRA-GRPO, EDAS, ReCo — all of which depend on a partition being
approach-level. It also supplies the human labels that Lee et al.~(`2606.29985`) did not have
(their RLVR result came from an LLM judge calibrated on 80 pairs), and it independently echoes
EDAS's own ablation finding that 76.4% of numerically-different wrong answers sat above cosine
similarity 0.95.

**Reconcile, don't contradict, the 33% figure.** Lee et al. report 33% of MATH problems admit more
than one approach; your annotators found ~2% of sampled pairs to be different approaches. These
are compatible: their statistic is about problems admitting multiple approaches *in principle*,
yours is about what a 0.5B model actually samples. The diversity MEG needs is not absent from the
dataset — it is absent from the model's output distribution. That is a sharper finding than either
paper alone, and it explains mechanistically why every micro gate was inert.

## 5. A finding you have not claimed: the gates cost stability

| | Vanilla | SELF | CB | MEG | CB×SELF | CB×MEG |
|---|---|---|---|---|---|---|
| between-seed s.d. of slope | **0.042** | 0.049* | 0.325* | **0.395** | 0.279 | 0.249 |

(* from 2 seeds — a very unstable estimate; treat as indicative)

Vanilla's three seeds span 0.08; MEG's span 0.77. That is a **9.5x** ratio in standard deviation.
Levene's test across the four three-seed conditions gives p = 0.53, so this is **descriptive, not
significant** — variance tests at n=3 have almost no power, and you should say so rather than
imply a test was passed. But the raw pattern is clean and has a mechanism: SELF discards 61–64% of
problems and MEG redistributes credit, so each gated run learns from an effectively smaller and
noisier sample. The gates did not change the mean outcome; they made the outcome less reproducible.

For a paper whose own protocol insists on multi-seed evidence, "the proposed methods are less
seed-stable than the baseline they were meant to improve" is an honest and useful result.

## 6. Statistical corrections before this is written up

**6.1 The §8 correlations do not survive multiplicity.** Seven correlations, smallest p = 0.02.
Bonferroni gives 0.14; Benjamini–Hochberg gives 0.14. The "more variety → more problems lost"
relation (ρ = +0.57) is **not** significant after correction. You label the section descriptive,
which is correct, but the summary in §11 leans on it as a finding. Either report it as a
hypothesis generated by this study, or report the adjusted p alongside. Given §4 shows the
"variety" is mostly wording, the cleaner reading is that this correlation has no interpretation
worth defending.

**6.2 State the bounded null in units a reader can hold** — per contrast, not as one number.
ln 32 = 3.466, so ±1 pp per ln $k$ is ±3.5 Pass@32 points. Converting every pre-registered CI:

| contrast | CI (pp per ln $k$) | CI (Pass@32 points) |
|---|---|---|
| P1 CB×MEG − CB×SELF | [−0.79, +0.68] | **[−2.7, +2.4]** |
| P2 CB×MEG − MEG | [−0.60, +1.05] | **[−2.1, +3.6]** |
| S1 CB×MEG interaction | [−0.94, +1.69] | [−3.3, **+5.9**] |
| S2 CB×SELF interaction | [−0.71, +1.68] | [−2.5, **+5.8**] |
| S3 CB main effect | [−0.59, +0.25] | [−2.0, +0.9] |
| E1 CB×MEG − Vanilla | [−0.91, +0.30] | [−3.2, +1.0] |

The primaries are tight — "any difference between MEG and SELF as micro terms is smaller than
about 2.7 Pass@32 points" is a statement a reader can evaluate. **The interactions are not**: S1
and S2 admit effects up to +5.9 and +5.8 points, more than twice the primary bound and larger
than vanilla's entire +2.3 point gain at k=32. Do not describe the interactions as a tight
bounded null — quote each interval, and say that the interaction question remains open at this
seed count rather than answered. This is the n=3 power limit showing up exactly where it was
predicted to.

**6.3 Unequal seeds.** SELF and CB have n=2, the rest n=3. Mark it everywhere they appear as
peers, and keep them out of any variance comparison that is presented as like-for-like.

**6.4 Drop or re-scope the CB subject claim.** ρ = +0.01 (n=28 subject-runs) between a topic's
credit share and its Pass@32 change means the macro gate's premise is unsupported — the gate
acted (it down-weighted 6–17% of answers) and still moved nothing. That is a *stronger* negative
than "CB didn't help", because it isolates the failure to the premise rather than the
implementation. Note that this is only available because you fixed δ: at the original δ=0.98 the
gate never bound, and you could not have distinguished the two.

## 7. Title and abstract

Your own read is right that #1 and #3 are the reliable findings and #4 should not be the hook.
A title that carries both:

> **No Boundary to Shrink: RLVR Expands Pass@$k$ at Small-Model Training Budgets, and a Human
> Audit Finds Its "Solution Diversity" Is Mostly Wording**

Abstract skeleton, in the order the evidence actually supports:

1. The dispute, and the pre-registered design (2x3 factorial, 16 runs, blinded human audit).
2. **At 0.26 epochs and ~2 rollouts per problem, no shrinkage**: Pass@32 rises 73.0 → 75.3,
   slope +0.66, every condition positive, 14 of 16 runs positive.
3. **No fix helped**, with bounds: the two primary contrasts lie within [−2.7, +3.6]
   Pass@32 points of zero; the secondary interactions are wider, [−3.3, +5.9].
4. **The human audit explains why**: the partitioner scores 55% against a consensus where a
   trivial always-SAME baseline scores 98%, so the diversity these methods reweight is wording,
   not approach — with 85% annotator agreement (PABAK 0.70).
5. **Interpretation**: shrinkage tracks training intensity, not model scale; the regime a
   single-GPU SLM budget reaches is one where RLVR can only expand.
6. Release.

## 8. What dies in the current draft

Most `\RT` tags flip. The mechanism you built for this is doing exactly its job — the list is
mechanical, not a rewrite:

| tag | current wording | status |
|---|---|---|
| R1, R18 | H2 "mixed outcome", most lost problems validly solved | **inconclusive** — below-chance agreement |
| R2, R13, R14 | shrinkage real but overstated; genuine part within groups | **false** — no shrinkage |
| R3 | vanilla reproduces the inversion | **false** — opposite sign |
| R4 | every gate has a higher slope than vanilla | **false** — vanilla is highest |
| R5 | neither primary clears the bar | **holds** (the one that survives intact) |
| R6 | interactions bounded near zero | **holds**, restate in Pass@32 units |
| R7 | entropy decays under vanilla, MEG retains it | **false** — vanilla entropy rose 0.73 → 0.76 |
| R8 | macro gate rarely binds | **superseded** — it binds (6–17%) and still does nothing |
| R9 | boundary exits fall under the composite | **false** — 25.0 vanilla vs 25.0 CB×MEG; MEG worst at 28.3 |
| R10, R11 | random-partition and BBG controls | **not run** — remove, list as unrun deviations |
| R12 | GSM8K out-of-distribution | **not run** — drop the paragraph |
| R15 | embedding candidates collapse to one mode | **holds** — and is now load-bearing for §4 |
| R16 | CB flattens the subject profile | **false** — ρ = +0.01 |
| R17 | partitioner validated above the 70% bar | **false** — 55%, and the bar was pre-registered |

Four deviations need recording in the paper, all pre-results and therefore clean: 240 → 120 steps,
GSM8K dropped, seed 2 only for the primary conditions, and the random-partition/BBG controls not
run. The last one costs you E2 and E3 — without the random-partition control you cannot show the
partition *matters*, which is worth one line in Limitations now that §4 rests on partition quality.
