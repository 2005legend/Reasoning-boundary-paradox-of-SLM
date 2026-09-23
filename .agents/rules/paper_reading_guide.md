# Paper Reading Guide & Core Context

This rule provides essential context for the "Reasoning boundary paradox of SLM" paper.

## 1. The 4 Essential Directories
Ignore old Kaggle files (01_-07_, cbgrpo, exp2_*, scratch, temp). Focus only on:
- `paper/`: The final submission.
- `claude science/`: Research gaps and verified prior work (novelty case).
- `heg_grpo_aws/`: The experiment (code, plan, pre-registration).
- `05_documentation/HEG_GRPO_PROGRESS_LOG.md`: The diary of decisions and bugs.

## 2. Core Claims & Novelty
- **Gaps**: (1) No one tested the combination. (2) Papers disagree on whether shrinkage is real. (3) No one asked human evaluators.
- **Novelty**: We are the FIRST to test topic-level × within-group together. We did NOT invent rarity reweighting (Cue-GRPO did).
- **Reviewer Defense**: When asked "how is this different from Cue-GRPO?", remember that Cue-GRPO invented rarity reweighting, but we are the first to combine topic-level (CB) and within-group (MEG/SELF).

## 3. Pre-registration (heg_grpo_aws/PREREGISTRATION.md)
- Covers the design, shrinkage slope outcome, hypotheses (P1/P2/S1-S3).
- **Deviations**: 120 steps, GSM8K -> MATH, O-SELF -> SELF, and MEG/CB reformulations.

## 4. Code Structure (heg_grpo_aws/rlvr/)
- `config.py`: All settings (lr, G=8, CB θ/δ, MEG α).
- `data.py`: MATH problems/answers loader.
- `rewards.py`: Answer judgment (includes memory guard).
- `grpo_core.py`: One training step (generate 8 -> reward -> advantages -> gate -> update).
- `gates.py`: Heart of the paper (CB moves credit between topics; MEG moves credit between solution approaches within one problem).

## 5. The 8 Critical Questions (To Memorize)
1. **Pass@k shrinkage**: What it is, and why Pass@32 matters more than Pass@1 here.
2. **Mechanisms**: CB (topic credit), SELF, and MEG (approach credit), in one sentence each.
3. **Design**: Why a 2×3 factorial instead of just "our method vs baseline".
4. **Hypotheses**: What are P1 and P2, and why only these two are primary.
5. **Prior Work**: How this is different from Cue-GRPO, SELF, and BBG.
6. **Pre-registration**: Why pre-register, and what were deviations 6, 7, and 8.
7. **Human Eval**: What H1 and H2 test, and why no other paper has them.
8. **Null Result Claim**: If vanilla shows no shrinkage (seed 0), what does the paper claim?

## 6. Action Items
- **Human Eval**: Need to find 2 annotators for once results are in (`heg_grpo_aws/HUMAN_EVAL_GUIDE.md`).
- **Editing**: Follow `paper/EDIT_GUIDE.md` for result-dependent sentences (R1-R18) and `paper/AI_WRITING_SIGNS.md` for Polish.
