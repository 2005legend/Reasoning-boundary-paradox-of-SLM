# 🎯 CB-GRPO Training Checklist for New Session

## Before Starting Colab

### Files to Upload to Google Drive

```
/content/drive/MyDrive/RLVR_Research/
│
├── modules/                          # ← UPLOAD THESE
│   ├── cbgrpo_config.py             # From: 03_cbgrpo_implementation/
│   ├── cbgrpo_trainer.py            # From: 03_cbgrpo_implementation/
│   └── cell6_clustering.py          # From: 03_cbgrpo_implementation/
│
├── cluster_assignments.pkl          # ← COPY from Account 1
│
└── results/                          # (Optional - for comparison)
    └── exp2_n1_vanilla_0.5B/        # ← COPY from Account 1
```

---

## In Colab: Cell Execution Order

### ✅ MUST RUN (In This Order)

| # | Cell | Time | What It Does |
|---|------|------|--------------|
| 0 | **CELL_0_CBGRPO_SETUP** | 5 min | Master setup (mounts drive, installs packages, verifies modules) |
| 6 | GSM8K Dataset Loading | 2 min | Loads training and test datasets |
| 23 | CB-GRPO Configuration | 2 min | Sets up CB-GRPO parameters |
| 24 | CB-GRPO Training | 2-4 hr | Trains model with capacity balancing |
| 25 | CB-GRPO Evaluation | 30-60 min | Evaluates trained model |
| 26 | Comparison Analysis | 5 min | Compares vanilla vs CB-GRPO |

**Total Time**: ~3-5 hours

---

### ❌ DO NOT RUN (Skip These!)

| Cells | Why Skip |
|-------|----------|
| 1-5 | Covered by CELL_0_CBGRPO_SETUP |
| 7-19 | Utility functions (not needed) |
| **20-22** | **VANILLA GRPO** - Already trained in Account 1! |

---

## Visual Flow

```
START
  │
  ├─→ [0] CELL_0_CBGRPO_SETUP
  │     ├─ Mount Drive
  │     ├─ Install packages
  │     ├─ Create directories
  │     ├─ Verify CB-GRPO modules ✅
  │     ├─ Verify cluster assignments ✅
  │     └─ Set experiment config
  │
  ├─→ [6] Load GSM8K Dataset
  │
  ├─→ [23] CB-GRPO Configuration
  │     └─ EXPERIMENT_NAME = "exp2_n1_cbgrpo_0.5B"
  │
  ├─→ [24] CB-GRPO Training ⏳ 2-4 hours
  │     ├─ Load model with QLoRA
  │     ├─ Prepare dataset with cluster_ids
  │     ├─ Initialize CBGRPOTrainer
  │     └─ Train with capacity balancing
  │
  ├─→ [25] CB-GRPO Evaluation ⏳ 30-60 min
  │     ├─ Load trained model
  │     ├─ Evaluate on test set
  │     └─ Save results
  │
  └─→ [26] Comparison Analysis
        ├─ Compare with vanilla GRPO
        └─ Generate reports
  │
END
```

---

## Safety Checks

### Before Running Cell 24 (Training)

Verify these in your notebook:

- [ ] `EXPERIMENT_NAME = "exp2_n1_cbgrpo_0.5B"` (NOT vanilla!)
- [ ] `USE_SMOKE_TIER = True` (for quick test) or `False` (for full training)
- [ ] `cluster_assignments.pkl` exists
- [ ] CB-GRPO modules imported successfully
- [ ] GSM8K dataset loaded

### Red Flags (STOP if you see these)

- ⚠️ `EXPERIMENT_NAME` contains "vanilla"
- ⚠️ Cell mentions "vanilla GRPO" or "exp2_n1_vanilla"
- ⚠️ Output directory is `/checkpoints/exp2_n1_vanilla_0.5B/`

---

## Expected Output Structure

After completing all cells:

```
/content/drive/MyDrive/RLVR_Research/
│
├── checkpoints/
│   └── exp2_n1_cbgrpo_0.5B/
│       ├── final_model/              # Trained model
│       ├── checkpoint-100/           # Checkpoints
│       └── ...
│
├── results/
│   ├── exp2_n1_cbgrpo_0.5B/
│   │   ├── evaluation_metrics.json       # Pass@1, Pass@4
│   │   ├── evaluation_details.json       # Per-problem results
│   │   ├── cbgrpo_balance_history.json   # Capacity metrics
│   │   └── cbgrpo_balance_report.json    # Balance analysis
│   │
│   ├── exp2_n1_vanilla_0.5B/         # From Account 1
│   │
│   └── comparison_report.json        # Vanilla vs CB-GRPO
│
└── modules/
    ├── cbgrpo_config.py
    ├── cbgrpo_trainer.py
    └── cell6_clustering.py
```

---

## Quick Reference Commands

### Check Current Experiment
```python
print(f"Experiment: {config.exp_name}")
print(f"Mode: {'CB-GRPO' if 'cbgrpo' in config.exp_name else 'VANILLA'}")
```

### Emergency Stop
If you accidentally start wrong training:
1. Click **Runtime → Interrupt execution**
2. Click **Runtime → Factory reset runtime**
3. Start fresh from Cell 0

### Verify Everything is Ready
```python
# Quick verification
print("Setup Status:")
print(f"  Drive mounted: {Path('/content/drive/MyDrive').exists()}")
print(f"  Modules dir: {(Path('/content/drive/MyDrive/RLVR_Research/modules')).exists()}")
print(f"  Cluster file: {(Path('/content/drive/MyDrive/RLVR_Research/cluster_assignments.pkl')).exists()}")
print(f"  Experiment: {config.exp_name}")
```

---

## Timeline Summary

| Phase | Time | Cumulative |
|-------|------|------------|
| Setup (Cell 0) | 5 min | 0:05 |
| Dataset (Cell 6) | 2 min | 0:07 |
| Config (Cell 23) | 2 min | 0:09 |
| Training (Cell 24) | 2-4 hr | 2:09 - 4:09 |
| Evaluation (Cell 25) | 30-60 min | 2:39 - 5:09 |
| Analysis (Cell 26) | 5 min | 2:44 - 5:14 |

**Total**: ~3-5 hours

---

## What You'll Have at the End

1. ✅ Trained CB-GRPO model
2. ✅ Evaluation results (Pass@1, Pass@4)
3. ✅ Balance history and analysis
4. ✅ Comparison with vanilla GRPO
5. ✅ Research-ready results

---

## Need Help?

- Check `02_notebook_cells/NEW_SESSION_QUICKSTART.txt`
- Check `03_cbgrpo_implementation/CBGRPO_INTEGRATION_GUIDE.md`
- Verify files in `03_cbgrpo_implementation/`

---

**Status**: Ready to run! 🚀

**Next**: Upload files to Drive → Open Colab → Run Cell 0
