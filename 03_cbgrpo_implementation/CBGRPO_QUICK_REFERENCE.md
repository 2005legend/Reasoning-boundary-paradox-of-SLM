# CB-GRPO Quick Reference Card

## What is CB-GRPO?

**Capacity-Balanced GRPO** - A novel algorithm that tracks cumulative gradient-mass spend per cluster and applies soft decay to over-spending clusters.

---

## The Problem It Solves

```
Vanilla GRPO Training:
├── All clusters treated equally
├── Some clusters dominate gradient updates
├── Others under-trained
└── Result: Inconsistent reasoning (Pass@4 >> Pass@1)

Your Data:
├── Pass@1 = 25% (low consistency)
├── Pass@4 = 65% (high capability)
└── Gap = 40% (high variance)
```

---

## How CB-GRPO Works

```
1. Track Spend via EMA
   S_c(t) = α * |∇L_c(t)| + (1-α) * S_c(t-1)

2. Compute Weights
   w_c = decay^(S_c / S_mean - θ)  if S_c / S_mean > θ
       = 1.0                       otherwise

3. Reweight Loss
   L_final = L_grpo * mean(w_c)
```

---

## Key Parameters

| Parameter | Default | Range | What It Does |
|-----------|---------|-------|--------------|
| `n_clusters` | 16 | 8-32 | Number of prompt clusters |
| `ema_alpha` | 0.01 | 0.01-0.05 | Memory length (lower = longer) |
| `decay_factor` | 0.9 | 0.85-0.95 | Penalty strength (lower = stronger) |
| `theta_threshold` | 1.2 | 1.1-1.5 | Sensitivity (lower = more sensitive) |

---

## Expected Results

| Metric | Vanilla | CB-GRPO | Improvement |
|--------|---------|---------|-------------|
| Pass@1 | 25% | 30-35% | +5-10% |
| Pass@4 | 65% | 65-70% | +0-5% |
| Gap | 40% | 30-35% | -5-10% |

---

## Integration Checklist

- [ ] Upload `cbgrpo_config.py` to Colab
- [ ] Upload `cbgrpo_trainer.py` to Colab
- [ ] Add Cell 23 (CB-GRPO config)
- [ ] Add Cell 24 (CB-GRPO training)
- [ ] Add Cell 25 (CB-GRPO evaluation)
- [ ] Add Cell 26 (Comparison analysis)
- [ ] Run Cell 6 (clustering) if not done
- [ ] Verify `cluster_assignments.pkl` exists

---

## Monitoring During Training

### Good Signs ✅

- Gini coefficient < 0.3
- Gini decreasing over time
- Max/Min spend ratio < 3.0
- Stable over-spending clusters (some, not all)

### Warning Signs ⚠️

- Gini coefficient > 0.3
- Gini increasing
- Max/Min spend ratio > 5.0
- No clusters over-spending
- All clusters over-spending

---

## Quick Tuning Guide

**High Gini (> 0.3)?**
```
↓ theta_threshold (more sensitive)
↓ decay_factor (stronger penalty)
↓ ema_alpha (longer memory)
```

**No over-spending?**
```
↓ theta_threshold (lower threshold)
↑ ema_alpha (faster adaptation)
```

**Low Pass@1?**
```
↑ n_clusters (finer granularity)
Check cluster quality (Cell 6)
Analyze cluster semantics
```

---

## File Locations

### Modules (Upload to Colab)
```
/content/drive/MyDrive/RLVR_Research/modules/
├── cbgrpo_config.py
└── cbgrpo_trainer.py
```

### Outputs
```
/content/drive/MyDrive/RLVR_Research/
├── checkpoints/exp2_n1_cbgrpo_0.5B/
│   └── final_model/
├── results/exp2_n1_cbgrpo_0.5B/
│   ├── evaluation_metrics.json
│   ├── cbgrpo_balance_history.json
│   └── cbgrpo_balance_report.json
└── results/
    └── comparison_report.json
```

---

## Commands

### Run Training
```python
# Cell 23: Configure
cbgrpo_config = CBGRPOConfig(
    n_clusters=16,
    ema_alpha=0.01,
    decay_factor=0.9,
    theta_threshold=1.2
)

# Cell 24: Train
trainer = CBGRPOTrainer(
    model=model,
    args=cbgrpo_config.to_grpo_config(),
    train_dataset=train_dataset,
    cbgrpo_config=cbgrpo_config
)
trainer.train()
```

### Check Results
```python
# Cell 25: Evaluate
# Runs automatically

# Cell 26: Compare
# Generates comparison report
```

---

## Key Differences from Vanilla GRPO

| Aspect | Vanilla GRPO | CB-GRPO |
|--------|--------------|---------|
| Weighting | Uniform | Capacity-aware |
| Memory | None | EMA of full trajectory |
| Mechanism | Standard GRPO | Soft decay for over-spenders |
| Metrics | Loss, reward | + Gini, spend distribution |

---

## Novel Contribution

**What's New?**
- EMA-based spend tracking (not snapshot-based)
- Trajectory-aware reweighting (not current batch only)
- Soft decay (not hard clipping)

**Why It Matters?**
- Addresses capacity allocation problem
- Improves reasoning consistency
- First approach to track cumulative spend in GRPO

---

## Timeline

| Task | Time |
|------|------|
| Integration | 10 min |
| Smoke test | 30-60 min |
| Full training | 2-4 hours |
| Evaluation | 30-60 min |
| Analysis | 30 min |

---

## Success Criteria

✅ Pass@1 improvement > +5%
✅ Gap reduction > -5%
✅ Gini coefficient < 0.3
✅ Hard → Easy transitions > 1

---

## Need Help?

1. Check `CBGRPO_INTEGRATION_GUIDE.md`
2. Review `CBGRPO_IMPLEMENTATION_SUMMARY.md`
3. Check Colab logs
4. Verify prerequisites (Cell 6, cluster_assignments.pkl)

---

**Status**: ✅ Ready to run

**Next Step**: Upload modules to Colab and run Cell 23
