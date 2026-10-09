# The three research gaps (2026-10-05)

The paper claims three kinds of gap. This file records the evidence for each, so the claims can be defended
in review. Searches were run on 2026-10-05:
- web search and arXiv abstract pages, with citation metadata read from each paper's arXiv page;
- the Wiley Scholar Gateway connector, which returned only off-topic journal articles: it does not index arXiv
  ML preprints;
- alphaXiv was disconnected, and the Scholar Sidekick citation checker is not subscribed.

Re-run this search before submission: the area moves monthly.

---

## Gap 1: contradiction. Is Pass@k shrinkage real?

The literature disagrees, and the paper should say so in the first paragraph.

| Position | Papers | Claim |
|---|---|---|
| **Shrinks** | Yue et al. 2504.13837; Nguyen et al. 2510.02230; Wu & Xuan 2507.14843 ("Invisible Leash"); Zhou & Li 2608.29188; Zhou 2607.20543 | base models overtake RLVR at large k; RLVR stays inside the base model's support; contraction happens at the first decision of a solution |
| **Expands** | Liu et al. 2505.24864 (ProRL); Wen et al. 2506.14245 | prolonged KL-controlled training reaches problems the base model never solves; RLVR wins under CoT-Pass@k |
| **Both** | Yao et al. 2510.04028 ("The Debate on RLVR Reasoning Capability Boundary") | an exploitation stage shrinks the boundary, then an exploration stage expands it |
| **Measurement artifact** | Yuan et al. 2606.15455; Dragoi et al. 2510.08325 | the decline partly reflects few rollouts per problem; Pass@k is biased toward lucky hits (Cover@τ) |

**What this paper adds:**
- a pre-registered, multi-seed measurement at 0.5B, in the short-training regime;
- problem-level boundary entry and exit counts;
- rollouts per training problem, which answers Yuan et al.;
- a human check of whether the lost problems were ever solved by valid reasoning (Gap 2, H2), which bears
  directly on the "artifact" position.

## Gap 2: methodological. Everyone measured it; nobody asked people

Every measurement in the debate is automatic, and the closest uses of humans stop short of judging RLVR outputs.

| Paper | Human role | Why it is not this paper's audit |
|---|---|---|
| Lee et al. 2606.29985 (EMNLP 2026) | 17 annotators labelled **80 pairs** (MATH train) to calibrate an LLM judge, with 80% agreement; judge-human agreement 85% | the RLVR analysis was done by the LLM judge. Their own summary: humans did not directly judge RLVR solutions |
| Yang et al. 2605.09292 | AI coders with human adjudication, on 80 AMC/AIME problems | frontier models under prompting; no RL, no shrinkage |
| Taşaltı et al. 2609.32622 | native speakers checked translations only | shows CoT-Pass@k judges (DeepSeek V4-Flash, Qwen3.6-35B, R1-distill) **accept corrupted chains almost as often as clean ones**; Pass@k − CoT-Pass@k ≈ **19.7 points on Qwen2.5-generation solvers** (our model family) |
| Wen et al. 2506.14245 | none for CoT checking (an LLM verifier) | the "expansion" evidence depends on the judges that Taşaltı et al. found unreliable |

**Claim the paper makes, worded carefully:** "To our knowledge, no study has asked people what RLVR actually
loses: whether the solutions that disappear are distinct strategies, and whether the problems that leave the
boundary were ever solved by valid reasoning."

**What this paper adds:** a pre-registered, blinded two-annotator audit (`heg_grpo_aws/scripts/human_eval.py`,
`HUMAN_EVAL_GUIDE.md`).
- **H1 (partition validity):** 60 solution pairs. Cohen's κ, and the partitioner's balanced accuracy against
  the human consensus, with a 70% bar.
- **H2 (what is lost):** valid-reasoning rate of the base model's successes on exited vs difficulty-matched
  retained problems, compared with a Fisher exact test and a pre-registered reading: lucky hits, genuine loss,
  or mixed.

## Gap 3: novelty. Do intra-group and cross-prompt fixes compose?

Unchanged from `novelty_verification_2026-10-03.md`: no paper combines a latent topic-level budget with
intra-group credit redistribution and tests them for complementarity. The neighbours:
- ReCo: two corrections, both within one prompt;
- Cue-GRPO: the micro rule we adopt;
- MT-GRPO, CurveRL, BBG: cross-prompt reweighting only.

---

## Code changes this required (2026-10-05, before any main run)

- **Final evals now save every sampled completion** (`*_completions.jsonl.gz`). Without them H2 cannot be built.
  The size is roughly 50 MB uncompressed per model on MATH-500, and much less gzipped.
- `calibration_groups.jsonl` now includes the problem text, for H1.
- `scripts/human_eval.py`: builds the sheets and does the scoring.
  - Statistics are tested against scikit-learn (κ) and SciPy (Fisher).
  - There is also an end-to-end test on synthetic data.
- `PREREGISTRATION.md` has a dated "Human evaluation" section with interpretation rules.
- `control.ipynb` has a "Human evaluation" step, to run before terminating the instance.

## Sources

Abstract pages:
- [2510.04028](https://arxiv.org/abs/2510.04028), [2505.24864](https://arxiv.org/abs/2505.24864),
  [2506.14245](https://arxiv.org/abs/2506.14245), [2507.14843](https://arxiv.org/abs/2507.14843)
- [2510.08325](https://arxiv.org/abs/2510.08325), [2608.29188](https://arxiv.org/abs/2608.29188),
  [2605.09292](https://arxiv.org/abs/2605.09292)
- [2609.32622](https://arxiv.org/html/2609.32622), [2606.29985](https://arxiv.org/html/2606.29985) (full text read
  for the human-annotation details)
