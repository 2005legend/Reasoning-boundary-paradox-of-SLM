# CB-GRPO Integration Guide

## Quick Start

This guide explains how to add CB-GRPO cells to your existing `rlvr_training_pipeline.ipynb` notebook.

---

## Files Created

| File | Purpose | Add to Notebook |
|------|---------|-----------------|
| `cbgrpo_config.py` | Configuration class with capacity balancing parameters | Upload to Colab |
| `cbgrpo_trainer.py` | Trainer class with EMA spend tracking and reweighting | Upload to Colab |
| `CELL_23_CBGRPO_CONFIG.txt` | Pre-flight configuration for CB-GRPO | Add as Cell 23 |
| `CELL_24_CBGRPO_TRAINING.txt` | Training execution for CB-GRPO | Add as Cell 24 |
| `CELL_25_CBGRPO_EVALUATION.txt` | Evaluation for CB-GRPO model | Add as Cell 25 |
| `CELL_26_COMPARISON_ANALYSIS.txt` | Comparison between vanilla GRPO and CB-GRPO | Add as Cell 26 |

---

## Step-by-Step Integration

### Step 1: Upload CB-GRPO Modules to Colab

In Colab, run this cell **before** Cell 23:

```python
# Upload CB-GRPO modules to Colab
import os
from pathlib import Path

# Create modules directory
modules_dir = Path("/content/drive/MyDrive/RLVR_Research/modules")
modules_dir.mkdir(parents=True, exist_ok=True)

print("📁 Upload these files to the modules directory:")
print(f"   {modules_dir}")
print()
print("   1. cbgrpo_config.py")
print("   2. cbgrpo_trainer.py")
print()
print("You can:")
print("  • Use the Colab file browser (folder icon on left)")
print("  • Drag and drop files into the modules directory")
print("  • Or copy-paste the code directly")
```

**Alternative**: Copy-paste the code directly into Colab cells:
1. Create a new cell before Cell 23
2. Copy the entire contents of `cbgrpo_config.py` into that cell
3. Create another cell
4. Copy the entire contents of `cbgrpo_trainer.py` into that cell

### Step 2: Add Cells to Notebook

Open your `rlvr_training_pipeline.ipynb` and add new cells:

1. **Cell 23**: Copy contents from `CELL_23_CBGRPO_CONFIG.txt`
   - This configures CB-GRPO parameters
   - Run AFTER Cell 22 (vanilla evaluation)

2. **Cell 24**: Copy contents from `CELL_24_CBGRPO_TRAINING.txt`
   - This runs CB-GRPO training
   - Takes 2-4 hours (or 30-60 min for smoke tier)

3. **Cell 25**: Copy contents from `CELL_25_CBGRPO_EVALUATION.txt`
   - This evaluates the CB-GRPO model
   - Uses identical protocol as Cell 22

4. **Cell 26**: Copy contents from `CELL_26_COMPARISON_ANALYSIS.txt`
   - This compares vanilla GRPO vs CB-GRPO
   - Generates comprehensive reports

---

## Expected Workflow

```
Existing Cells (already done):
├── Cell 1-19: Setup, data loading, model loading
├── Cell 20: Pre-flight config (vanilla GRPO)
├── Cell 21: Training (vanilla GRPO)
└── Cell 22: Evaluation (vanilla GRPO) ✅ COMPLETE
    └── Results: Pass@1=25%, Pass@4=65%

New Cells (CB-GRPO):
├── Cell 23: Pre-flight config (CB-GRPO)
│   └── Configure capacity balancing parameters
├── Cell 24: Training (CB-GRPO)
│   └── Run with EMA spend tracking
├── Cell 25: Evaluation (CB-GRPO)
│   └── Evaluate on same test set
└── Cell 26: Comparison Analysis
    └── Compare vanilla vs CB-GRPO
```

---

## Key Differences: Vanilla GRPO vs CB-GRPO

### Vanilla GRPO (Cell 20-22)

```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="...",
    learning_rate=5e-6,
    # ... standard parameters
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # Uniform weighting across all clusters
)
```

### CB-GRPO (Cell 23-25)

```python
from cbgrpo_config import CBGRPOConfig
from cbgrpo_trainer import CBGRPOTrainer

cbgrpo_config = CBGRPOConfig(
    # Standard GRPO parameters
    output_dir="...",
    learning_rate=5e-6,
    
    # CB-GRPO specific parameters
    n_clusters=16,           # From Cell 6 clustering
    ema_alpha=0.01,          # EMA decay rate
    decay_factor=0.9,        # Soft decay for over-spenders
    theta_threshold=1.2,     # Threshold multiplier
)

trainer = CBGRPOTrainer(
    model=model,
    args=cbgrpo_config.to_grpo_config(),
    train_dataset=train_dataset,  # Must have 'cluster_id' field
    cbgrpo_config=cbgrpo_config,
    # Capacity-aware weighting via EMA spend tracking
)
```

---

## Critical Requirements

### 1. Cluster Assignments (Cell 6)

CB-GRPO requires cluster IDs for each training sample.

**Ensure Cell 6 was run and created `cluster_assignments.pkl`:**

```python
cluster_assignments = cluster_training_prompts(
    prompts=gsm8k_train['problems'],
    n_clusters=16,
    cluster_file="cluster_assignments.pkl"
)
```

**Verify:**

```python
import pickle
from pathlib import Path

cluster_file = Path("/content/drive/MyDrive/RLVR_Research/cluster_assignments.pkl")
assert cluster_file.exists(), "Run Cell 6 first!"
```

### 2. Dataset with cluster_id Field

Cell 24 adds cluster IDs to the dataset:

```python
train_data = []
for prompt, problem, answer in zip(formatted_prompts, train_problems, train_answers):
    cluster_id = cluster_assignments.get(problem, 0)  # Get from Cell 6
    train_data.append({
        "prompt": prompt,
        "problem": problem,
        "ground_truth": answer,
        "cluster_id": cluster_id  # CRITICAL for CB-GRPO
    })
```

### 3. Same Test Set for Fair Comparison

Both Cell 22 and Cell 25 use the same test samples:

```python
test_problems = test_problems[:20]  # First 20 problems
test_ground_truths = test_ground_truths[:20]
```

---

## Hyperparameter Tuning Guide

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `ema_alpha` | 0.01 | 0.01-0.05 | Memory length for spend tracking |
| `decay_factor` | 0.9 | 0.85-0.95 | Penalty strength for over-spenders |
| `theta_threshold` | 1.2 | 1.1-1.5 | Sensitivity for detecting over-spending |

### Tuning Recommendations

**If Gini coefficient is high (> 0.3):**
- Decrease `theta_threshold` (more aggressive balancing)
- Decrease `ema_alpha` (longer memory)
- Decrease `decay_factor` (stronger penalty)

**If no clusters are over-spending:**
- Decrease `theta_threshold` (lower threshold)
- Increase `ema_alpha` (shorter memory, faster adaptation)

**If Pass@1 doesn't improve:**
- Increase `n_clusters` (finer granularity)
- Adjust `theta_threshold` based on spend distribution
- Check if clusters represent meaningful reasoning patterns

---

## Monitoring During Training

### Key Metrics to Watch

1. **Gini Coefficient** (logged every 50 steps)
   - 0.0 = Perfect balance
   - 0.3+ = High inequality (some clusters dominate)

2. **Over-spending Clusters**
   - Count of clusters with spend > threshold
   - Should stabilize over training

3. **Max/Min Spend Ratio**
   - Should decrease over training (more balanced)
   - High ratio (> 3.0) indicates imbalance

4. **Spend Distribution**
   - Visualize at end of training
   - Should be roughly uniform

### Example Output

```
[Step 50] Capacity Balance Metrics:
  Mean spend: 0.0234
  Std spend: 0.0089
  Max/Min ratio: 2.45
  Over-spending clusters: 3/16
  Gini coefficient: 0.1823 (0=perfect balance, 1=max inequality)
```

---

## Expected Results

Based on your current vanilla GRPO results:

| Metric | Vanilla GRPO | CB-GRPO (Expected) | Improvement |
|--------|--------------|---------------------|-------------|
| Pass@1 | 25% | 30-35% | +5-10% |
| Pass@4 | 65% | 65-70% | +0-5% |
| Gap | 40% | 30-35% | -5-10% |
| Easy | 4/20 | 6-8/20 | +2-4 |
| Hard | 7/20 | 4-6/20 | -1-3 |
| Incorrect | 9/20 | 6-8/20 | -1-3 |

**Why this improvement?**
- Better capacity allocation
- More consistent reasoning paths
- Less gradient budget consumed by dominant clusters

---

## Troubleshooting

### Error: "cluster_id field missing from dataset"

**Cause**: Cell 24 couldn't find cluster assignments

**Solution**: Run Cell 6 first to create `cluster_assignments.pkl`

### Error: "CBGRPOTrainer not found"

**Cause**: Modules not uploaded to Colab

**Solution**: Upload `cbgrpo_config.py` and `cbgrpo_trainer.py` to `/content/drive/MyDrive/RLVR_Research/modules/`

### Warning: "No clusters over-spending"

**Cause**: Threshold too high

**Solution**: Decrease `theta_threshold` (e.g., from 1.2 to 1.1)

### Warning: "High Gini coefficient"

**Cause**: Imbalanced capacity allocation

**Solution**: 
- Decrease `ema_alpha` (longer memory)
- Decrease `decay_factor` (stronger penalty)
- Decrease `theta_threshold` (more sensitive)

---

## Research Questions to Explore

1. **Capacity Allocation Dynamics**
   - How does spend distribution evolve during training?
   - Which clusters tend to over-spend?
   - Does the balance stabilize over time?

2. **Reasoning Pattern Analysis**
   - Which reasoning patterns benefit most from CB-GRPO?
   - Are certain problem types more sensitive to capacity balancing?

3. **Scale Effects**
   - Does CB-GRPO help more for smaller models (0.5B) or larger (1.5B)?
   - How does the optimal `n_clusters` change with model size?

4. **Comparison with Other Approaches**
   - Compare CB-GRPO with curriculum learning
   - Compare with static reweighting schemes
   - Analyze computational overhead

---

## Next Steps

1. ✅ Integrate Cells 23-26 into notebook
2. ⏳ Run CB-GRPO training (2-4 hours)
3. ⏳ Evaluate CB-GRPO model
4. ⏳ Compare results with vanilla GRPO
5. ⏳ Analyze balance history
6. ⏳ Tune hyperparameters if needed
7. ⏳ Run multiple experiments with different seeds
8. ⏳ Write research paper section on CB-GRPO

---

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Verify all prerequisites (Cell 6, cluster assignments)
3. Check Colab logs for error messages
4. Refer to `CELL_24_CBGRPO_TRAINING.txt` for detailed comments

Good luck with your CB-GRPO experiments! 🚀
