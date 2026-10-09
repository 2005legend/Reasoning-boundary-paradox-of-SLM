# Novelty Upgrade Addendum — RLVR Reasoning Boundary Paradox Project

**Purpose:** This document extends `design.md` and `requirements.md`. It does not replace them — it diagnoses why CB-GRPO's measured effect has been small, proposes a scoped set of upgrades to fix that (ranked by compute cost), and gives you positioning language for a conference submission and a Master's application. New requirements continue the numbering in `requirements.md` (39+).

---

## 0. TL;DR

- Your topic — RLVR's "Pass@k inversion" / reasoning-boundary shrinkage — is one of the most actively contested areas in LLM-RL right now. At least eight relevant papers appeared between April 2025 and July 2026, including one (arXiv:2604.06298) working on the *exact same setup* as you (Qwen 0.5B–3B, LoRA/QLoRA, GRPO, GSM8K/MATH). You are not going to win on "we tried a mitigation and it moved the needle a bit." You can win on **mechanism**: explaining *why* your gate barely worked, then fixing that specific gap.
- Most likely root cause of "very lil improvement": your gate operates on the wrong granularity. CB-GRPO throttles by **topic cluster** (16 KMeans clusters over ~7.4k prompts, using `|advantage|` as spend). The literature's actual mechanism for shrinkage is **per-prompt / per-solution-mode winner-take-all** — a much finer-grained effect that a 16-way topic partition can't see. That's a diagnosable, fixable, publishable gap — not a dead end.
- The single highest-leverage change: turn CB-GRPO into a **two-level (hierarchical) gate** that combines your existing cluster budget with a per-prompt solve-rate term (you've already spec'd `OSELFGate` — reuse it as the micro component instead of treating it as a separate baseline). This is a half-day of code changes given your existing Strategy-pattern `Gate` interface, not a new subsystem.
- Second highest-leverage change, and free: add the interference/entropy diagnostics from the base SELF paper. This costs zero extra training — it's a forward-pass-only analysis on checkpoints/logs you already save. It converts "CB-GRPO helped a little" into "here is exactly what CB-GRPO did and didn't fix, and why," which is a far stronger paper regardless of what the numbers show.
- Third: your eval protocol (n=10 rollouts, k≤10, GSM8K as primary dataset) may be *floor/ceiling-limited* — GSM8K is close to saturated for instruction-tuned Qwen2.5 at these sizes, and shrinkage in the literature is usually reported at k up to 256. You may be measuring in a range where the effect barely shows up for *any* method. This is worth fixing before drawing conclusions about CB-GRPO's effect size at all.
- Do **one** headline algorithmic contribution (the hierarchical gate), not three. Treat anchoring and difficulty-aware clustering as an ablation and a future-work paragraph, respectively. Reviewers (and admissions committees) trust one clean, well-ablated idea over three half-finished ones.

---

## 1. Why CB-GRPO Likely Shows Only a Small Effect

Your `CBGRPOGate` (design.md Algorithm 1) tracks an EMA of `|advantage|` per **semantic topic cluster** (KMeans, k=16, sentence-transformer embeddings) and soft-decays samples from clusters that have "overspent" relative to the mean. This is a reasonable and defensible idea — but it targets **topic diversity across the training corpus**, which is a different axis from the mechanism the field has converged on as the actual driver of Pass@k shrinkage:

- The base paper you're building on (Nguyen et al., "The Reasoning Boundary Paradox," arXiv:2510.02230 — this is literally where your project's title comes from) identifies **negative interference** and **winner-take-all** as the mechanism: on-policy GRPO disproportionately reinforces the *single already-most-likely correct solution mode* for problems the base model can already solve, while problems whose correct solutions sit in low-likelihood regions get vanishing gradient and are gradually suppressed. This is a **per-prompt, per-solution-mode** effect, not a per-topic one.
- Yue et al. (arXiv:2504.13837) and the two-stage dynamic-view paper (arXiv:2510.04028) confirm the same pattern across model families: the shrinkage is concentrated on specific problems / specific low-probability correct trajectories, not on specific subject-matter clusters.
- A cluster in your setup groups ~467 prompts by *topic* (word-problem theme), which mixes together "already-solved, winner-take-all" prompts and "boundary, low-likelihood" prompts within the very same cluster. Throttling the whole cluster once its aggregate spend crosses `theta` will suppress gradient for *both* the harmful winner-reinforcement AND the useful hard-problem learning inside that cluster, diluting any benefit.
- Separately: `spend` accumulates `|advantage|`, i.e., it does not distinguish *reinforcing an already-dominant correct mode* (the harmful dynamic) from *punishing wrong answers on a hard, under-solved cluster* (a useful dynamic). Mixing these into one scalar spend further weakens the signal your gate is throttling on.

None of this means CB-GRPO is a bad idea — it means it's solving a real but different problem (topic-level training-attention fairness) than the one that Pass@k shrinkage is caused by (prompt/solution-level winner-take-all). That mismatch is a legitimate, citable, paper-worthy finding on its own, and it points directly at the fix in §4.

**Two more mundane but important possibilities**, worth ruling out before you interpret any of this mechanistically (see §6):
1. **Measurement floor/ceiling.** GSM8K is close to saturated for Qwen2.5-Instruct at 0.5–1.5B, and your Pass@k range (k ≤ 10, n = 10) is narrow. Papers demonstrating strong shrinkage typically look at k up to 128–256, on harder benchmarks (MATH, AIME, Minerva). If both Vanilla and CB-GRPO already sit near 95–100% Pass@10 on GSM8K, there's no room left for either to show a difference — vanilla, CB-GRPO, or anything else.
2. **Single-seed noise.** RLVR training is high-variance. A single-run "Δslope" between Vanilla and CB-GRPO with no seed variation and no CI could easily be within noise, in either direction.

---

## 2. Related-Work Radar (read/cite these before you write the paper)

You need this section not because it's polite, but because a reviewer who knows this literature and sees you unaware of it will reject on those grounds alone, regardless of your results. All of these are 2025–2026 (i.e., likely outside what you'd find without searching — the field is moving fast under you).

| Paper | arXiv | Why it matters to you |
|---|---|---|
| Yue et al., "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" | 2504.13837 | The paper that established Pass@k inversion as a phenomenon; your N1 is a direct replication/extension of this at SLM scale. Cite as the origin claim. |
| Nguyen et al., "The Reasoning Boundary Paradox…" (SELF) | 2510.02230 | Your title and your `OSELFGate`/`StaticSELFGate` baselines come from here. Read §4–6 closely — the interference metrics (Δ⁺, ‖Δ‖) in §4 are directly reusable as diagnostics (see §5 below). |
| "When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion" (PBA) | 2607.20543 | July 2026 — very recent, very close to your framing. Proposes anchoring training on rare base-model-correct trajectories at the *per-problem* level. This is your closest potential competitor/complement — cite it and explicitly differentiate (cluster-level budgeting vs. per-problem anchoring; see §7 for how to combine rather than compete). |
| "Understanding Diversity Collapse in RLVR via the Lens of Overtraining" | 2606.15455 | Argues Pass@256 decline can be explained by overtraining/redundant reinforcement even *without* new-capability loss — a useful caution for how you interpret shrinkage-slope results. Cite when discussing what shrinkage slope does/doesn't prove. **Corrected 2026-10-03 (full read):** its zero-success intervention restricts updates to problems with zero successes *in the current rollout group* and learns from them through a signed REINFORCE loss, so it gives zero gradient under GRPO; its method BBG gates problems by success count (added to the AWS package as the GRPO-compatible baseline `bbg`). Report rollouts per training problem with every Pass@k curve. |
| "The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both?" | 2510.04028 | A two-stage account (early shrinkage, later possible expansion with prolonged training). Relevant to your N3/step-ablation (Exp3) — frame your intermediate-checkpoint analysis (Req 25) against this. |
| "Uniform-Correct Policy Optimization" | 2605.00365 | Clusters *responses within a prompt* by reasoning strategy and enforces uniform mass across them. Conceptually adjacent to your cluster idea but at a different granularity (intra-prompt response clusters vs. your inter-prompt topic clusters). Cite to pre-empt "isn't this the same idea" reviews, and explain the granularity difference explicitly. |
| "Limits of Difficulty Scaling: Hard Samples Yield Diminishing Returns in GRPO-Tuned SLMs" | 2604.06298 | **Closest neighbor to your exact experimental setup** (Qwen 0.5B–3B, LoRA, GRPO, GSM8K+MATH, difficulty-stratified analysis). You must read and differentiate from this directly for your N3 capacity-scaling claim, or a reviewer will flag it as prior art you missed. |
| "GRIP: Geometric Refinement and Adaptive Information Potential" | 2603.00031 | Uses macro cluster-level budgets + intra-cluster diversity sampling — but for *data curation/efficiency*, not RLVR shrinkage. Good citation to show your cluster-budget mechanism has a lineage, while being a different application. |
| PSN-GRPO (parameter-space noise for RLVR exploration) | 2602.02555 | Explicitly claims to be *orthogonal and composable* with other exploration/diversity methods. Useful precedent for your "hybrid gate is composable" framing. |
| **GCPO, "Breaking Winner-Takes-All: Cooperative Policy Optimization Improves Diverse LLM Reasoning"** | **2605.11461** | **Independently verified 2026-09-22 (title/authors/abstract fetched from arXiv, real paper, May 2026).** The single closest prior-art match to Requirement 44's `MEGGate`: explicitly targets winner-take-all collapse within GRPO rollout groups, uses semantic embeddings, and states "only correct and non-redundant rollouts contribute" to its diversity credit — i.e. correctness-gated, intra-group, embedding-based redundancy reweighting, the same core idea as MEG, formalized via a determinant-volume/coverage measure instead of clustering+entropy. **You must cite this and cannot claim the intra-group mechanism itself as novel** — see the revised positioning in §4.4 below. |
| **DRA-GRPO, "Your GRPO Needs to Know Diverse Reasoning Paths"** | **2505.09655** | **Independently verified 2026-09-22.** Downweights redundant completions *within a single rollout group* via embedding similarity (Submodular Mutual Information / Graph-Cut), applied uniformly to all rollouts (not correctness-gated, unlike GCPO/MEG). Same granularity and tool (sentence embeddings) as MEG, earlier (May 2025). Cite as the origin of the "intra-group embedding-redundancy reweighting" idea. |
| **EDAS, "Leveraging Error Diversity in Group Rollouts for Reinforcement Learning"** | **2605.17333** | **Independently verified 2026-09-22.** The mirror image of MEG: entropy-based (Shannon entropy over an error-class distribution), intra-group, but operates *exclusively on incorrect rollouts* — explicitly rejects embedding-based clustering for this (their own ablation: 76.4% of numerically-different wrong answers had cosine similarity >0.95). **Two implications:** (1) cite as the entropy-based sibling of MEG, positioned on the opposite (incorrect) side of the rollout group; (2) their embedding-clustering critique is a live methodological risk for MEG's own correct-side clustering — flag as a limitation/threat-to-validity in the paper, and consider validating (e.g. sample and manually inspect a few MEG mode clusters) that they track distinct *solution methods*, not just surface phrasing. |

| **Cue-GRPO, "When Correct Solutions Repeat: Rarity-Aware Credit Redistribution for GRPO"** | **2608.03467** | **Verified 2026-10-03 (full text).** MEG's mechanism exactly: redistributes positive credit over a partition of the correct rollouts by cluster rarity (weight ∝ \|C\|^-α, mean-restored), incorrect untouched; deterministic "strategy cue" partitions; random-partition and mean-matched ablations; cites UARL (judge partitions). The AWS package now uses this rule for the micro term, and says so. |
| **ReCo, "Reweighting GRPO Against Distributional Concentration"** | **2607.26862** | **Verified 2026-10-03.** Response-level normalization by expected occurrence + variance-based token ratio, shown to be *complementary* — but both levels are within one prompt. Our levels are cross-prompt topic x intra-group. Cite in the abstract's second sentence. |
| **MT-GRPO, "Multi-Task GRPO: Reliable LLM Reasoning Across Tasks"** (ICML 2026) | **2602.05547** | **Verified 2026-10-03.** Task-level weights for worst-task accuracy with a ratio-preserving sampler. Closest macro-level neighbour: explicit tasks, robustness objective, sampler-based vs our latent topics, positive-advantage budget, multiplicative gate, Pass@k objective. |
| **CurveRL, "Principled Distribution-Aware Context Reweighting"** | **2605.24331** | **Verified 2026-10-03.** Prompt-level weights from the pass-rate quantile distribution. Per-prompt difficulty axis, no intra-group term. |
| **"Are We Measuring Strategy or Phrasing?"** | **2606.29985** | **Verified 2026-10-03.** Embedding cosine is the weakest proxy for approach-level diversity (bigram overlap does better); **only 0.7% of GSM8K problems admit >1 approach vs 33% on MATH**. Reason the AWS track now trains on MATH and calibrates the partitioner. |
| DATPO | 2609.08650 | Verified 2026-10-03. Difficulty-adaptive tree rollouts + sibling-diversity advantage for Pass@k: a two-axis method, but changes the rollout procedure and its coarse axis is per-problem difficulty, not topic. |
| FADE | 2607.01490 | Verified 2026-10-03. Positive/negative gradient-mass decomposition; cite where CB-GRPO's positive-mass spend is introduced. |
| ES vs GRPO coverage | 2608.27351 | Verified 2026-10-03. Non-gating route to broader coverage; one sentence on why gating suits a small budget. |
| TreeAdv | 2601.03703 | Verified 2026-10-03. Entropy-branched trees + advantage redistribution; one line with the intra-group family. |
| Reward granularity at 0.5B | 2607.02869 | Verified 2026-10-03. Qwen2.5-0.5B **base** (not Instruct), GSM8K, single seed: loose anchor only. |
| GRPO training dynamics for SLMs | 2609.39321 | Verified 2026-10-03. 1.5B-7B, pass@5 only, no shrinkage mitigation: SLM-scale neighbour. |

**Verification note (2026-10-03):** every row from GCPO down, plus 2510.02230, 2607.20543 and 2606.15455, was read in full text with the alphaXiv connector on 3 Oct; see `claude science/novelty_verification_2026-10-03.md` for differentiation and the revised abstract.

**Verification note (2026-09-22):** the three rows above were checked by fetching arXiv directly (title, authors, dates, abstracts, and method sections confirmed to exist as described). The other rows in this table were written in an earlier session and have **not** been re-verified the same way in this pass — before submitting anywhere, re-check each arXiv ID resolves to the claimed paper (a couple of the IDs above are dated after this document's own earlier text implied, so don't assume the dates/claims in the rest of this table are exact without a final pass).

Action item: skim each abstract, note one sentence of differentiation, and keep this table (trimmed) as your related-work section skeleton.

---

## 3. Upgraded Research Objectives

Keep N1–N3 as your foundation (they're solid and already well-instrumented). Add:

- **N1 (unchanged):** Boundary-shrinkage detection across 0.5B/1.5B.
- **N2 (unchanged, but reframed):** Test whether existing mitigations (O-SELF, Static-SELF, Adaptive-Rollout) transfer to SLMs — keep as baselines, not as "the competition," since your headline method will *subsume* the per-prompt idea (§4).
- **N3 (unchanged):** Capacity–shrinkage relationship — **must** cite and differentiate from arXiv:2604.06298 (see §2).
- **N4 (new — this is your headline contribution):** Does topic-level (macro) and prompt-level (micro) gating address *distinct* components of reasoning-boundary shrinkage, such that combining them is super-additive relative to either alone?
- **N5 (new — free, diagnostic):** Does the SELF paper's interference metric (Δ⁺, ‖Δ‖) mediate the relationship between cluster-spend imbalance (Gini) and shrinkage slope — i.e., is CB-GRPO's partial effect explained by a partial reduction in interference?

---

## 4. Core Novelty Lever: Hierarchical Capacity-Budgeted GRPO (H-CB-GRPO)

### 4.1 Design

Two gate components, computed independently, combined multiplicatively:

- **Macro component (existing CB-GRPO, refined):** cluster-level EMA spend, but split by advantage sign so you only throttle *over-reinforcement of already-dominant correct solutions*, not corrective negative gradient on hard clusters.
- **Micro component (reuses your existing `OSELFGate` machinery):** per-prompt EMA of "is the greedy decode already correct." Down-weight prompts the model has already effectively solved, regardless of which cluster they're in.

This is not a new subsystem — it's a `CompositeGate` that wraps two gates you've already designed, which fits your existing Strategy pattern (`Gate` abstract interface) with no architectural changes.

```
ALGORITHM: H-CB-GRPO Composite Gate
INPUT: batch (prompts, cluster_ids, advantages), state (step, epoch)
OUTPUT: gate_weights (per-sample multipliers in [0, 1])

INITIALIZE:
  spend_pos_ema[c] ← 0 for all clusters c        # only positive-advantage mass
  greedy_solve_ema[p] ← 0 for all prompts p       # reuse from OSELFGate
  theta ← 1.5, decay ← 0.98, ema_alpha ← 0.05     # macro params (unchanged)
  tau_solve ← 0.7, lambda_self ← 0.3              # micro params (new)

FOR each sample i in batch:
  c ← cluster_ids[i];  p ← prompt_ids[i];  a ← advantages[i]

  # --- Macro: cluster budget on POSITIVE mass only ---
  pos_mass ← max(a, 0)
  spend_pos_ema[c] ← ema_alpha * pos_mass + (1 - ema_alpha) * spend_pos_ema[c]
  ratio ← spend_pos_ema[c] / (mean(spend_pos_ema) + 1e-8)
  IF ratio > theta:
    macro_w[i] ← decay ^ (ratio - theta)
  ELSE:
    macro_w[i] ← 1.0

  # --- Micro: per-prompt solve-rate gate (SELF-style, already built) ---
  IF greedy_solve_ema[p] > tau_solve:
    micro_w[i] ← lambda_self          # down-weight already-solved "winner" prompts
  ELSE:
    micro_w[i] ← 1.0

  # --- Combine ---
  gate_weight[i] ← macro_w[i] * micro_w[i]
END FOR

RETURN gate_weights
```

**What changed from your original CB-GRPO, and why each change matters:**
1. `spend` now accumulates `max(advantage, 0)` instead of `|advantage|` — negative gradient on hard clusters is no longer treated as "spend" that gets throttled. This directly targets winner-take-all reinforcement specifically, per the mechanism in arXiv:2510.02230 §4–5.
2. The micro term reuses `greedy_solve_ema`, which you already compute for `OSELFGate` — **no new state tracking required**, only a new combination function.
3. `lambda_self` (not a hard cutoff) keeps it soft/differentiable in spirit with your existing "soft decay, no hard cutoff" design philosophy (design.md line 1784).

### 4.2 The ablation that makes this a paper, not just a patch

Run four conditions at 1.5B (reuse your existing standard-tier budget, no new infra):
1. Vanilla GRPO
2. CB-GRPO (macro only — your current method, refined per §4.1's positive-mass fix)
3. O-SELF (micro only — already planned as a baseline)
4. H-CB-GRPO (macro × micro)

If (4) > max((2),(3)) by more than the noise floor (see §6.4 for how to test this), you have a clean, quotable result: **"macro (topic) and micro (per-prompt) gating address distinct components of shrinkage, and are compositionally beneficial."** If (4) ≈ max((2),(3)), that's *also* a valid, reportable finding ("the two mechanisms are redundant, not complementary") — either outcome is publishable as long as you have the mechanism-level diagnostics from §5 to explain *why*.

### 4.3 New Requirements (EARS format, continues from Requirement 38)

**Requirement 39: Hierarchical Composite Gate (H-CB-GRPO)**

*User Story:* As a researcher, I want a composite gate combining cluster-level and prompt-level signals, so that I can test whether macro and micro interventions address distinct components of boundary shrinkage (N4).

Acceptance Criteria:
1. THE System SHALL implement `CompositeGate` conforming to the existing `Gate` interface, accepting two child `Gate` instances.
2. THE System SHALL compute `macro_weight` using cluster-level EMA spend restricted to positive-advantage mass (`max(advantage, 0)`), replacing the `|advantage|` accumulator used by the original `CBGRPOGate`.
3. THE System SHALL compute `micro_weight` by reusing the `greedy_solve_ema` state already implemented for `OSELFGate`.
4. THE System SHALL combine weights as `gate_weight = macro_weight * micro_weight`.
5. THE System SHALL support a configuration flag selecting the combination operator (`multiplicative`, `min`, `harmonic_mean`) for ablation purposes.
6. THE System SHALL log `macro_weight`, `micro_weight`, and `gate_weight` separately per step for diagnostic purposes.
7. THE System SHALL train the 1.5B model with `CompositeGate` for the standard tier, alongside the existing Vanilla/CB-GRPO/O-SELF conditions (Requirement 22/23).
8. THE System SHALL validate that `0.5B + CompositeGate` is rejected at configuration time, consistent with the existing incompatibility rule for `OSELFGate` at 0.5B (Requirement 33.4).

---

## 4.4 Superseding Lever: Mode-Entropy Gate (MEG) / Hierarchical Entropy-Gated GRPO (HEG-GRPO)

> **⚠️ Update (2026-10-03), supersedes parts of the algorithm text below.** In the AWS package
> (`heg_grpo_aws/`) MEG is now the **mean-preserving rarity credit redistribution rule of Cue-GRPO
> (2608.03467)** applied to MEG's solution-mode partition (weights ∝ |C|^-0.8 over a group's correct
> rollouts, renormalized to mean 1; incorrect untouched), not the λ down-weight formula below; the
> partitioner (embedding vs bigram) is chosen by a pre-registered calibration; training moved to MATH;
> `o_self` is now real SELF (greedy-failure selection) and runs at 0.5B. The novelty claim narrowed to
> the complementarity test of a latent topic-level budget and intra-group redistribution. See
> `claude science/novelty_verification_2026-10-03.md` and `heg_grpo_aws/PREREGISTRATION.md`.

> **⚠️ Positioning correction (2026-09-22, read this before writing anything about MEG in the paper).**
> The text below originally framed MEG's intra-group redundancy mechanism itself as the headline
> novel contribution. A literature check (§2, verified against arXiv directly) found that is **not
> accurate**: GCPO (2605.11461), DRA-GRPO (2505.09655), and EDAS (2605.17333) already do
> correctness-aware, intra-group, embedding/entropy-based redundancy reweighting — GCPO in
> particular ("only correct and non-redundant rollouts contribute") is essentially the same idea MEG
> implements, published months before this document's revision. **Do not claim the intra-group
> mechanism as novel.** The defensible claim is narrower: this project's contribution is the
> **hierarchical composition** of a cross-topic macro budget (`CBGRPOGate`, §4.1–4.3, which has its
> own separate lineage — see GRIP, 2603.00031) with an intra-group micro diversity term from this
> emerging GCPO/DRA-GRPO/EDAS family — a combination that, as far as this project's search found,
> nobody has published — evaluated specifically through the reasoning-boundary/Pass@k-shrinkage
> diagnostic lens at SLM scale under heavy compute constraints. Frame the contribution as: *"we show
> that macro (cross-topic) and micro (intra-group, adapting the GCPO/DRA-GRPO family) interventions
> address distinct components of shrinkage and are compositionally beneficial"* — a combination-and-
> diagnosis paper, not a new-mechanism paper. This is still a legitimate, citable contribution; it is
> just a different (narrower, more honest) one than originally drafted. The implementation below is
> unchanged — only the claim about what's new in it changes.

**Why H-CB-GRPO (§4.1–4.3) is still not enough.** `OSELFGate` (the micro term in H-CB-GRPO) is a genuine improvement over cluster-only budgeting, but it is still a *proxy* for the mechanism, not a measurement of it: it flags a prompt "solved" only after many past steps' evidence accumulates into a cross-step EMA, and it treats the prompt as one binary unit — not the individual rollouts within a single GRPO group, which is where winner-take-all reinforcement literally happens (arXiv:2510.02230 §4–5: on-policy GRPO disproportionately reinforces whichever correct solution mode a rollout group happens to land on repeatedly). Requirement 44 replaces that proxy with a direct, within-group measurement — adapting the mechanism used by GCPO/DRA-GRPO/EDAS (see the positioning note above) rather than inventing it — and composes it hierarchically with the macro cluster budget. H-CB-GRPO (§4.1–4.3) is retained as an ablation baseline (Exp 4 in the checklist): the HEG-GRPO vs. H-CB-GRPO comparison is what supports the "macro+micro composition helps, and a direct intra-group measurement beats a cross-step proxy" claim.

**Requirement 44: Mode-Entropy Gate (MEG) and HEG-GRPO**

*User Story:* As a researcher, I want a micro gate that measures solution-mode redundancy directly within each GRPO rollout group, so that the gating signal targets the exact mechanism (within-group winner-take-all) the project is diagnosing, instead of a cross-step correctness proxy for it (N6).

**Algorithm.** For each prompt's `group_size` rollouts sampled this step (already generated for the GRPO update — no extra `.generate()` calls):
1. Embed each completion with a cached, CPU-resident sentence-transformer (keeps the T4's VRAM fully committed to the QLoRA policy model).
2. Agglomeratively cluster the group's embeddings by cosine distance (threshold `1 − tau_mode`) into solution "modes" — a stand-in for distinct reasoning paths, independent of correctness.
3. For rollout *i*, let `mode_freq[i]` = the fraction of its own group sharing its mode (`1/group_size` = a unique path this step; `1.0` = the whole group converged on one path).
4. `MEGGate` weight: for **correct** rollouts, `weight = 1 − lambda_meg · clip((mode_freq − 1/G) / (1 − 1/G), 0, 1)` — a continuous, soft down-weight of over-represented *correct* modes. For **incorrect** rollouts, `weight = 1.0` always (mirrors the positive-mass fix in §4.1: never throttle corrective negative gradient, only over-reinforcement of an already-dominant correct answer).
5. `HEG-GRPO = CBGRPOGate(macro) × MEGGate(micro)`, using the same generic `CompositeGate` wrapper as H-CB-GRPO (Requirement 39.1) — no new subsystem, only a new `Gate` implementation plugged into the existing Strategy interface.

Acceptance Criteria:
1. THE System SHALL implement `MEGGate` conforming to the existing `Gate` interface, with `local_increments`/`apply_synced_increments`/`state_dict` as no-ops (the gate is stateless: it is a pure function of the current step's rollout groups, not an EMA over past steps).
2. THE System SHALL compute intra-group mode assignment via agglomerative clustering (cosine distance, configurable threshold `meg_tau_mode`, default 0.85) over sentence-transformer embeddings of that step's already-generated completions — zero additional forward/generate passes.
3. THE System SHALL down-weight only CORRECT rollouts belonging to an over-represented mode, using a continuous `1 − lambda_meg · excess` formula (no hard cutoff, consistent with the project's existing soft-gating design philosophy) with configurable `meg_lambda` (default 0.7).
4. THE System SHALL leave INCORRECT rollouts' gate weight at 1.0 unconditionally, regardless of their mode's dominance, to preserve corrective negative gradient on hard/under-solved prompts.
5. THE System SHALL expose `heg_grpo` as a `CompositeGate(CBGRPOGate, MEGGate)` gate type, reusing the existing macro/micro composite machinery and configurable combination operator (Requirement 39.5).
6. THE System SHALL log mean intra-group mode entropy (Shannon entropy of the mode distribution, normalized to `[0, 1]`) per training step, as a diagnostic trend comparable to the interference/entropy diagnostics in Requirement 40.
7. THE System SHALL train the 1.5B **and** 0.5B model with `meg`/`heg_grpo` for the standard tier, alongside the existing Vanilla/CB-GRPO/O-SELF/H-CB-GRPO conditions.
8. THE System SHALL NOT apply the 0.5B incompatibility rule (Requirement 33.4/39.8) to `meg`/`heg_grpo`: unlike `OSELFGate`, `MEGGate` holds no persistent per-prompt state, so it carries none of the memory cost that motivated that restriction. This is a genuine practical advantage worth stating explicitly in the paper (cheaper to run at both scales than the baseline composite).

**Differentiation from cited prior work (update to §2's related-work table):**
- **vs. GCPO (arXiv:2605.11461) — the closest match, cite prominently:** GCPO already does correctness-gated, intra-group, embedding-based redundancy reweighting ("only correct and non-redundant rollouts contribute"), via a determinant-volume/coverage measure. MEG's *only* remaining distinctions are (a) discrete clustering + normalized Shannon entropy instead of a determinant-volume, which is a different formalization of a similar idea, not a different mechanism, and (b) MEG is composed hierarchically with a separate cross-topic macro budget, which GCPO does not have. Claim (b), not (a), as the contribution.
- **vs. DRA-GRPO (arXiv:2505.09655):** intra-group embedding-redundancy reweighting via Submodular Mutual Information, applied uniformly to all rollouts (not correctness-gated). Earlier and broader-scoped than MEG on the "which rollouts get reweighted" axis; cite as the origin of this idea, differentiate MEG's correctness-gating and macro composition.
- **vs. EDAS (arXiv:2605.17333):** the mirror image of MEG — entropy-based, intra-group, but exclusively on *incorrect* rollouts, and explicitly rejects embedding clustering (see §2's verification note) in favor of exact-answer canonicalization. Cite as the entropy-based sibling on the opposite side of the group, and treat their embedding-clustering critique as a limitation to address (validate that MEG's mode clusters track distinct solution methods, not just surface phrasing — see §4.4's positioning note).
- **vs. "Uniform-Correct Policy Optimization" (arXiv:2605.00365):** UCPO enforces a *hard uniform-mass constraint* across intra-prompt response clusters as part of the objective. MEG is a *soft multiplicative gate* — no new loss term, no constraint, fully compatible with the project's "soft decay, no hard cutoff" design commitment. Still worth citing, but GCPO/DRA-GRPO/EDAS are the more load-bearing comparisons now.
- **vs. PBA (arXiv:2607.20543, per-problem anchoring):** MEG adds no anchoring/SFT loss term and does not require caching base-model completions; it stays a pure on-policy soft-gate, operating one level finer than PBA's per-problem anchoring (within a group, not across a whole prompt's training history).
- **vs. this project's own H-CB-GRPO (§4.1–4.3):** both are macro × micro composites over the same `CompositeGate` wrapper; the only change is which mechanism the micro term measures — a cross-step correctness proxy (`o_self`) vs. a same-step, within-group redundancy measurement (`meg`, adapted from the GCPO/DRA-GRPO family). Keeping both in the ablation table is the evidence that granularity, not just "having a micro term," is what matters — this comparison, not the micro mechanism itself, is where this project's evidence has to come from now.

**Updated headline ablation (supersedes §4.2's four-condition table):** run six conditions at 1.5B (and, budget permitting, repeat `vanilla`/`meg`/`heg_grpo` at 0.5B, since those three are now cheap enough to run there): Vanilla, CB-GRPO, O-SELF, H-CB-GRPO, MEG, **HEG-GRPO**. The comparison that makes the paper is `HEG-GRPO` vs. `H-CB-GRPO`: if HEG-GRPO's Δslope and `Δ⁺` (Requirement 40) improvement exceed H-CB-GRPO's by more than the noise floor (Requirement 42), that is direct evidence that gating on the actual within-group mechanism outperforms gating on a proxy for it — a stronger, more specific claim than "our composite gate works a little."

---

## 5. Free Novelty: Mechanism-Level Diagnostics (No Extra Training Required)

This is the highest ratio of "paper strength gained" to "compute spent" available to you. All of it runs on checkpoints and rollout logs you are already saving (Requirement 14, 19).

**Requirement 40: Interference and Diversity Diagnostics**

*User Story:* As a researcher, I want to compute the interference metrics from the base SELF paper on my own checkpoints, so that I can causally connect cluster-spend imbalance to shrinkage rather than just correlating them.

Acceptance Criteria:
1. THE System SHALL construct a probing dataset by sampling G=4 base-model responses per training prompt prior to training (reuses your existing rollout-generation code; one-time cost).
2. THE System SHALL compute, at each saved checkpoint, the interference metric `Δ⁺(π_θ, π_b) = E[Δ log π_θ(y⁺|x)]` over the probing set, where `y⁺` are base-model-correct completions.
3. THE System SHALL compute the relative influence magnitude `‖Δ(π_θ, π_b)‖ = E[(Δ log π_θ(y|x))²]` over the same set.
4. THE System SHALL compute mean token-level entropy of the sampling distribution during rollout generation at each logged step (near-zero marginal cost — reuses logits already produced during generation).
5. THE System SHALL report `Δ⁺`, `‖Δ‖`, and entropy trends alongside Pass@k and shrinkage-slope curves for every gate condition (Vanilla, CB-GRPO, O-SELF, H-CB-GRPO, MEG, HEG-GRPO — see Requirement 44).
6. THE System SHALL compute the Pearson/Spearman correlation between per-condition cluster-spend Gini coefficient (already required, Requirement 24.4) and `Δ⁺`/`‖Δ‖`, in addition to the existing Gini–shrinkage-slope comparison.
7. THE System SHALL report this correlation as descriptive/exploratory (given the small number of conditions, ~8–12 data points) rather than as a hypothesis test with a p-value, and SHALL state this limitation explicitly in the summary report.

**Why this matters for your narrative:** if `Δ⁺` improves (less negative interference) under CB-GRPO but only modestly, and improves much more under H-CB-GRPO, that is direct mechanistic evidence — not just an outcome metric — for the macro/micro complementarity story in §4.2. This is the difference between "we propose a method and it works a bit" and "we propose a method, show what specifically it fixes and doesn't, and fix the rest" — the latter is what gets accepted.

---

## 6. Measurement Rigor Fixes (Do These Regardless of Which Novelty Lever You Pick)

These are not optional polish — if any of these is off, your headline number (CB-GRPO's Δslope) may be unmeasurable noise, and no algorithmic improvement will show through it.

### 6.1 Ceiling/floor effect on GSM8K

**Requirement 41: Evaluation Range and Ceiling-Effect Mitigation**

Acceptance Criteria:
1. THE System SHALL report Pass@1 and Pass@10 for the base model on GSM8K test, GSM8K-platinum, and MATH-500 *before* running any RL experiment, as a saturation check.
2. WHEN base-model Pass@10 on a dataset exceeds 90%, THE System SHALL flag that dataset as ceiling-limited for shrinkage-slope interpretation and SHALL treat MATH-500 (or the least-saturated dataset) as the primary evidence source for shrinkage claims, with GSM8K/platinum reported as secondary/robustness checks.
3. THE System SHALL increase `n` (solutions per problem) for Pass@k evaluation from 10 to the largest value tractable within the remaining Colab budget (target: n≥30 if feasible), to extend the measurable k range beyond k=10, since shrinkage in prior work is most visible at larger k.
4. THE System SHALL report, in the summary report, the maximum k at which Pass@k can be estimated without exceeding the standard error threshold implied by n (i.e., avoid reporting Pass@10 estimates that are effectively saturated at both base and RL model).

### 6.2 Single-seed noise

**Requirement 42: Multi-Seed Protocol and Significance Testing**

Acceptance Criteria:
1. THE System SHALL run the core comparison (Vanilla vs. CB-GRPO vs. H-CB-GRPO, 1.5B, standard tier) with a minimum of 3 random seeds per condition.
2. WHEN Colab time budget is constrained, THE System SHALL prioritize seed count over total training steps for the core comparison (e.g., prefer 3 seeds × standard tier over 1 seed × final tier).
3. THE System SHALL extend the existing bootstrap confidence-interval machinery (design.md Algorithm 3) to compute a bootstrap confidence interval on `Δslope = slope_condition − slope_vanilla` across seeds, not just within a single run's k-values.
4. THE System SHALL report whether the 95% CI on `Δslope` excludes zero, and SHALL avoid describing a result as "improvement" if the CI includes zero.
5. THE Final tier (1500–2000 steps, single seed) SHALL be reserved for at most one confirmatory run per headline condition, run only after the multi-seed standard-tier comparison identifies which conditions are worth confirming.

### 6.3 Difficulty-Aware Clustering Ablation (cheap, clean science)

**Requirement 43: Clustering Basis Ablation**

*User Story:* As a researcher, I want to test whether CB-GRPO's cluster definition (semantic topic vs. difficulty) changes its effectiveness, so that I can determine whether the "topic" framing or a "difficulty" framing better matches the shrinkage mechanism.

Acceptance Criteria:
1. THE System SHALL compute an additional per-prompt feature: base-model solve-rate (already required for `StaticSELFGate`, Requirement 9.1 — reuse, don't recompute).
2. THE System SHALL cluster prompts using KMeans on `[sentence-embedding, solve-rate]` (concatenated, solve-rate scaled) as an alternative to the existing embedding-only clustering, producing `difficulty-aware clusters`.
3. THE System SHALL re-run CB-GRPO (macro-only) with difficulty-aware clusters at 1.5B, standard tier, and compare shrinkage slope and Δslope-CI against the original semantic-only clustering.
4. THE System SHALL report this as a single, clearly labeled ablation table (not as a new headline method), since its purpose is diagnostic (does clustering basis matter) rather than a new algorithmic claim.

### 6.4 Reporting discipline

Add one paragraph to your summary-report template (`generate_summary_report`, design.md ~line 1678) explicitly stating: (a) number of seeds per condition, (b) whether the CI on Δslope excludes zero, (c) which dataset is primary vs. secondary for shrinkage claims, and (d) the ceiling-check result. This paragraph alone pre-empts the most common reviewer criticism of small-scale RLVR papers.

---

## 7. Optional Stretch (Only If Time Remains): Cluster-Level Base-Anchoring

If you have Colab budget left after the H-CB-GRPO ablation, this is the most novel-but-expensive addition, and it directly engages with the most recent competing paper (PBA, arXiv:2607.20543, July 2026). PBA anchors training on rare base-model-correct completions **per problem**. Your natural, differentiated extension: anchor **per cluster** instead — when a cluster's macro spend crosses `theta` (i.e., it's being throttled), mix in a small forward-KL (SFT-style) loss term against a small cache of frozen base-model correct completions for the currently under-sampled prompts in that cluster. This:

- Reuses your existing cluster infrastructure (no new clustering).
- Requires caching a small number of base-model correct generations per cluster up front (can be done once, during Exp0 base-model evaluation — you're already generating base-model rollouts there).
- Gives you a direct, explicit comparison point against PBA's per-problem version: "does anchoring at the cluster level capture most of the benefit at a fraction of the memory/compute cost of per-problem anchoring?" — a legitimate, citable empirical question.

**Do not build this unless H-CB-GRPO (§4) and the diagnostics (§5) are done and working.** Treat it as a "Requirement 44 (stretch)" and as a paragraph in your paper's future-work section if you run out of time — reviewers respond well to "we identified this direction and scoped it out explicitly" far better than to a rushed, under-ablated fourth method crammed in at the end.

---

## 8. Prioritized Roadmap

| Tier | Item | Extra Colab compute | Extra eng time | Payoff |
|---|---|---|---|---|
| **0 — do first, no retraining** | §6.4 reporting discipline; §5 interference/entropy diagnostics on existing checkpoints; ceiling check (§6.1.1–2) | ~0 (uses saved checkpoints/logs) | 0.5–1 day | Converts existing "lil improvement" result into a mechanism-level finding. Do this even if nothing else happens. |
| **1 — core novelty lever** | H-CB-GRPO composite gate (§4); re-run 4-condition ablation at 1.5B, standard tier, ×3 seeds (§6.2) | ~3× your current standard-tier 1.5B budget (3 seeds × 4 conditions vs. your current 1 seed × 2 conditions — roughly 4–6× total, but each run is standard tier not final tier, so net increase is manageable) | 1–2 days coding (CompositeGate wraps existing gates), rest is compute/monitoring | This is your headline contribution. |
| **1.5 — cheap ablation** | Difficulty-aware clustering ablation (§6.3) | 1× additional standard-tier 1.5B run | 0.5 day (clustering code change only) | Clean, low-risk secondary finding. |
| **2 — stretch, time permitting** | Cluster-level base-anchoring (§7) | 1–2× additional standard-tier 1.5B runs + one-time base-generation caching | 2–3 days (new loss term, generation caching) | Highest novelty ceiling, highest risk — only attempt after Tier 0/1 are solid. |

If your Colab/time budget is genuinely tight, **do Tier 0 and Tier 1 only**, and write Tier 1.5/2 into the paper as "future work" with one sentence each. A tight paper with one well-ablated, well-diagnosed idea beats a loose paper with three half-finished ones, both for reviewers and for a Master's application reader.

---

## 9. Positioning for Conference Submission (Reducing Rejection Risk)

- **Frame this as a diagnostic-and-mitigation paper, not a SOTA paper.** You are not going to beat labs running full fine-tuning on 7B+ models with production RL infra. You *can* produce the most careful small-scale, reproducible study of whether topic-level and prompt-level interventions are complementary — that's a real, defensible contribution at SLM scale.
- **Target workshop tracks, not main tracks, for your first submission.** Main-track NeurIPS/ICLR/ICML reviewers will (fairly) expect either much larger scale or a much bigger theoretical contribution. Workshops explicitly welcome "in-depth analysis of existing methods that provide new insights into limitations or behavior" (this is close to verbatim from NeurIPS 2026's own call for papers language) — that is exactly what §1 and §5 of this document give you.
- **Timing reality check:** several relevant NeurIPS 2026 workshops (e.g., MATH-AI) had paper deadlines in early September 2026, which has already passed as of this writing. Don't force a submission to a closed deadline. Check `https://aiworkshoptracker.com` and OpenReview for currently-open reasoning/efficiency/RL workshops (ICLR 2027 workshops, ICML 2027 workshops, and COLM typically open calls a few months out), and target the next realistic open window rather than rushing.
- **An arXiv preprint with a clean GitHub repo, submitted or not yet accepted anywhere, is a legitimate and common thing to list on a Master's application.** Don't let "must be accepted somewhere before I apply" become a blocker — a well-written preprint plus an honest account of what you learned is often more convincing to an admissions committee than a workshop acceptance with a thin idea, because it's easier for them to actually evaluate your thinking from the writing itself.
- **Explicitly write a "Limitations" section** covering: model scale (0.5B/1.5B only), single dataset family (GSM8K-derived), Colab compute constraints, and small number of seeds/conditions. Reviewers trust papers more, not less, when limitations are stated plainly — and it directly defuses the most common rejection reason for small-scale RL papers ("did the authors know their study was underpowered?").

---

## 10. Positioning for the Master's Application

What actually reads well to an admissions committee, roughly in order of weight:
1. **A clear, honest research narrative**: "I found an existing gap in a very recent paper's mitigation, diagnosed *why* it under-performed using the original paper's own theoretical machinery, and fixed the specific mechanism I identified." That sentence, almost verbatim, is a strong opening line for a research statement — and §1+§4 of this document is what makes it true rather than aspirational.
2. **Evidence of engaging with very recent literature** (papers from within the same year), which signals you can operate in a fast-moving field without hand-holding — the related-work table in §2 is exactly this evidence.
3. **A clean, documented, reproducible codebase** (your existing design.md is already unusually rigorous for a solo/small project — the checkpoint/resume system, property-based tests, and EARS-format requirements read like professional engineering practice, not just a notebook). Keep that; it's a real asset, mention it explicitly in application materials.
4. **Comfort with negative/mixed results.** A committee member who has done research knows that "my proposed method didn't fully work, and here's the rigorous reason why, and here's what I did about it" is a *stronger* signal of research maturity than "my method worked great" with thin analysis. Don't hide or oversell the small CB-GRPO effect — use it as the hook for the mechanism story.

---

## 11. Draft Contribution Statement (Adapt as Results Come In)

> **Revised 2026-09-22** to reflect the §2/§4.4 novelty check: MEG's intra-group mechanism is
> adapted from, not novel over, GCPO/DRA-GRPO/EDAS (all 2025–2026). The claim below is now about
> the *composition* and the *diagnostic evidence*, not about inventing the intra-group mechanism —
> do not revert to the earlier "we introduce MEG" framing.

> We study whether reasoning-boundary shrinkage under RLVR training of small language models (0.5B–1.5B) arises from a single mechanism or from (at least) two distinct, separately-addressable components: (i) coarse-grained, cross-topic training-attention imbalance, and (ii) fine-grained, within-group winner-take-all reinforcement of an already-dominant correct solution mode, as identified in prior work. We show that a cluster-level gradient-mass budget (CB-GRPO) that targets (i) alone produces only a marginal reduction in shrinkage slope. We further show that composing it with a cross-step, per-prompt correctness proxy for (ii) (H-CB-GRPO, using an EMA "already solved" signal) only partially closes the gap, and use interference and entropy diagnostics adapted from prior work to show this is because a cross-step proxy still under-targets (ii). We then compose the cluster-level budget with an intra-group solution-redundancy gate (MEG) — adapting the correctness-gated, embedding/entropy-based reweighting mechanism recently proposed by GCPO, DRA-GRPO, and EDAS (arXiv:2605.11461, 2505.09655, 2605.17333) for (ii) — into HEG-GRPO, the first composition, to our knowledge, of a cross-topic macro budget with this intra-group diversity-gate family. We show [HEG-GRPO produces a larger, statistically supported reduction in shrinkage slope than H-CB-GRPO / the proxy- and measurement-based micro terms are largely equivalent in effect despite the granularity difference — fill in once you have the multi-seed results], providing evidence that [macro and micro interventions from this emerging family are compositionally beneficial for SLM-scale RLVR / the two micro formulations are functionally redundant despite operating at different granularities]. We report this across 0.5B and 1.5B Qwen2.5-Instruct models trained under Colab/Kaggle-tier compute constraints — the intra-group term is stateless and therefore, unlike the baseline composite, runs at both scales — and discuss the capacity-scaling relationship in light of a closely related concurrent finding on capacity boundaries in GRPO-tuned SLMs (arXiv:2604.06298).

Both bracketed outcomes are genuinely publishable — write the abstract so it's true either way, then fill in the bracket once the multi-seed HEG-GRPO run finishes. **Do not claim "we introduce a novel gate that measures winner-take-all directly" as the headline — cite GCPO/DRA-GRPO/EDAS in the abstract's second sentence, not buried in related work, or a reviewer who knows this literature (a near-certainty, given how recent and close GCPO is) will read the omission as either unaware-of-the-field or deliberately obscuring it, both of which are worse outcomes than an honest "we compose and evaluate an existing family of methods" framing.**
