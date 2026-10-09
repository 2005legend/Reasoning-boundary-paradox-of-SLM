# Seed planning from the observed between-seed spread (math500, 0.5B)

Effects in pp per ln k (x ln 32 = points of gain from k=1 to k=32). MDE = minimum detectable effect at 80% power, two-sided alpha 0.05; half-width = half of the 95% t-interval.

| contrast | seeds used | sd of per-seed value | MDE n=3 (pp/ln k | Pass@32 pts) | half-width n=3 | MDE n=5 (pp/ln k | Pass@32 pts) | half-width n=5 |
|---|---|---|---|---|---|---|
| P1  heg_grpo - h_cb_grpo | 3 | 0.45 | 1.38 \| 4.8 | 1.11 | 0.74 \| 2.6 | 0.55 |
| P2  heg_grpo - meg | 3 | 0.63 | 1.96 \| 6.8 | 1.57 | 1.05 \| 3.6 | 0.78 |
| S1  CBxMEG interaction | 2 | 0.89 | 2.75 \| 9.5 | 2.21 | 1.48 \| 5.1 | 1.10 |
| S2  CBxSELF interaction | 2 | 0.71 | 2.20 \| 7.6 | 1.76 | 1.18 \| 4.1 | 0.88 |
| S3  CB main effect | 2 | 0.15 | 0.45 \| 1.6 | 0.36 | 0.24 \| 0.8 | 0.18 |
| E1  heg_grpo - vanilla | 3 | 0.25 | 0.76 \| 2.6 | 0.61 | 0.41 \| 1.4 | 0.31 |

Interactions (S1, S2) combine four conditions, so their per-seed sd is the largest and they gain most from extra seeds. sd rests on 2-3 seeds, so every number here is itself uncertain.
