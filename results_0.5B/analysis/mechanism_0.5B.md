# Mechanism analysis (0.5B, MATH-500, descriptive)

Base model: 6.73 correct modes per problem, top-mode share 0.43, normalized entropy 0.67.

| condition | seeds | slope | exits | entries | modes | d modes vs base | top share | d top share | train ent (last) | KL (last) |
|---|---|---|---|---|---|---|---|---|---|---|
| cb_grpo | 4 | +0.0028 | 26.2 | 36.0 | 7.08 | +0.535 | 0.413 | -0.019 | 0.709 | 0.0009 |
| h_cb_grpo | 5 | +0.0032 | 23.0 | 33.6 | 6.78 | +0.194 | 0.431 | +0.001 | 0.705 | 0.0005 |
| heg_grpo | 5 | +0.0038 | 24.0 | 34.0 | 6.97 | +0.403 | 0.414 | -0.017 | 0.727 | 0.0011 |
| meg | 5 | +0.0015 | 29.0 | 31.8 | 7.56 | +0.962 | 0.379 | -0.051 | 0.762 | 0.0024 |
| o_self | 4 | +0.0022 | 26.0 | 32.2 | 6.97 | +0.395 | 0.417 | -0.015 | 0.708 | 0.0007 |
| vanilla | 4 | +0.0042 | 26.0 | 33.2 | 7.44 | +0.943 | 0.386 | -0.048 | 0.767 | 0.0026 |

## Links across runs (Spearman)

| link | n | rho | p |
|---|---|---|---|
| diversity_change -> exits (modes_delta vs exits) | 27 | +0.40 | 0.039 |
| concentration_change -> exits (top_share_delta vs exits) | 27 | -0.37 | 0.058 |
| exits -> slope | 27 | -0.51 | 0.007 |
| diversity_change -> slope (modes_delta vs slope) | 27 | -0.07 | 0.747 |
| training entropy (last 20%) -> slope | 27 | +0.11 | 0.594 |
| policy movement (KL, last 20%) -> slope | 27 | +0.24 | 0.228 |
| topic spend share -> topic Pass@32 change (CB runs, pooled) | 98 | -0.02 | 0.852 |
