# Seed planning from the observed between-seed spread (math500, 0.5B)

Effects in pp per ln k (x ln 32 = points of gain from k=1 to k=32). MDE = minimum detectable effect at 80% power, two-sided alpha 0.05; half-width = half of the 95% t-interval.

| contrast | seeds used | sd of per-seed value | MDE n=3 (pp/ln k | Pass@32 pts) | half-width n=3 | MDE n=5 (pp/ln k | Pass@32 pts) | half-width n=5 |
|---|---|---|---|---|---|---|
| P1  heg_grpo - h_cb_grpo | 5 | 0.36 | 1.11 \| 3.9 | 0.89 | 0.60 \| 2.1 | 0.45 |
| P2  heg_grpo - meg | 5 | 0.45 | 1.38 \| 4.8 | 1.11 | 0.74 \| 2.6 | 0.55 |
| S1  CBxMEG interaction | 4 | 0.99 | 3.07 \| 10.6 | 2.46 | 1.65 \| 5.7 | 1.23 |
| S2  CBxSELF interaction | 4 | 0.78 | 2.40 \| 8.3 | 1.93 | 1.29 \| 4.5 | 0.96 |
| S3  CB main effect | 4 | 0.33 | 1.02 \| 3.5 | 0.82 | 0.55 \| 1.9 | 0.41 |
| E1  heg_grpo - vanilla | 4 | 0.54 | 1.67 \| 5.8 | 1.34 | 0.89 \| 3.1 | 0.67 |

Interactions (S1, S2) combine four conditions, so their per-seed sd is the largest and they gain most from extra seeds. sd rests on 2-3 seeds, so every number here is itself uncertain.
