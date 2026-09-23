# 🚀 CB-GRPO Training - Complete Guide

## ✅ What Changed

The `CELL_CBGRPO_ALL_IN_ONE.txt` is now a **TRUE all-in-one cell** that:

1. **Automatically creates clusters** if they don't exist (one-time, ~3-5 min)
2. **Loads existing clusters** if they're already created (instant)
3. **Trains the model** with CB-GRPO (~30-60 min for 100 steps)

## 📋 How to Use

### Single Cell Approach (Recommended)

**File:** `CELL_CBGRPO_ALL_IN_ONE.txt`

Just copy and run! The cell will:
- Check if `cluster_assignments.pkl` exists in Drive
- If NOT found → Create clusters automatically (first run only)
- If found → Load clusters instantly
- Then proceed with training

**Total time:**
- First run: ~35-65 minutes (clustering + training)
- Subsequent runs: ~30-60 minutes (training only)

### Two Cell Approach (Optional)

If you want to separate clustering from training:

1. **First:** Run `CELL_0_CLUSTERING.txt` (creates clusters)
2. **Then:** Run `CELL_CBGRPO_ALL_IN_ONE.txt` (trains)

## 🔍 What's Included

### STEP 1: Clustering (Auto-Created if Needed)
- ✅ Installs sentence-transformers, scikit-learn
- ✅ Loads GSM8K dataset (7,473 problems)
- ✅ Embeds with all-MiniLM-L6-v2
- ✅ Clusters into 16 groups (KMeans)
- ✅ Saves to `/content/drive/MyDrive/RLVR_Research/cluster_assignments.pkl`

### STEP 2-9: Training
- ✅ Loads Qwen 0.5B with QLoRA
- ✅ Prepares dataset with cluster IDs
- ✅ Defines reward function (format + answer checking)
- ✅ Initializes CB-GRPO trainer
- ✅ Trains for 100 steps (smoke test)
- ✅ Saves model to Drive

## 📦 Output Files

All saved to: `/content/drive/MyDrive/RLVR_Research/`

```
RLVR_Research/
├── cluster_assignments.pkl          # Cluster mappings (auto-created)
├── checkpoints/
│   └── exp2_n1_cbgrpo_0.5B_smoke/
│       ├── checkpoint-50/           # Mid-training checkpoint
│       ├── checkpoint-100/          # Final checkpoint
│       └── final_model/             # Merged final model
├── results/
│   └── exp2_n1_cbgrpo_0.5B_smoke/
│       └── training_metrics.json    # Training stats
└── logs/
    └── exp2_n1_cbgrpo_0.5B_smoke/   # Training logs
```

## ⚙️ Configuration

Edit these values in the cell to customize:

```python
# Experiment
EXPERIMENT_NAME = "exp2_n1_cbgrpo_0.5B_smoke"
USE_SMOKE_TIER = True  # False for 800 steps

# CB-GRPO
N_CLUSTERS = 16
EMA_ALPHA = 0.01
DECAY_FACTOR = 0.9
THETA_THRESHOLD = 1.2

# Training
TRAINING_STEPS = 100  # or 800
BATCH_SIZE = 2
LEARNING_RATE = 5e-6
```

## 🎯 Expected Output

```
================================================================================
🚀 CB-GRPO ALL-IN-ONE TRAINING
================================================================================

⏳ Installing required packages...
✅ Packages installed

Start time: 2026-09-08 05:30:00
================================================================================

Experiment: exp2_n1_cbgrpo_0.5B_smoke
Training steps: 100
CB-GRPO enabled: n_clusters=16, ema_alpha=0.01

⏳ STEP 1/9: Creating/loading cluster assignments...
  ⚠️  Cluster file not found. Creating clusters now...
  This is a one-time operation (~3-5 minutes)

  → Loading GSM8K dataset...
  → Loaded 7473 problems
  → Embedding prompts with all-MiniLM-L6-v2...
  → Embeddings computed: (7473, 384)
  → Clustering with KMeans (k=16)...
  → Clustering complete (inertia: 12345.67)
  → Saved to: cluster_assignments.pkl (245.32 KB)
  → Cluster stats: mean=467.1, min=412, max=523
✅ Created 7473 cluster assignments

⏳ STEP 2/9: Setting up directories...
✅ Directories created

⏳ STEP 3/9: Loading model with QLoRA...
✅ Model loaded with QLoRA
   Trainable params: 2,162,688

⏳ STEP 4/9: Loading GSM8K dataset...
✅ GSM8K loaded: 7473 problems

⏳ STEP 5/9: Preparing dataset with cluster IDs...
✅ Dataset prepared: 500 examples with cluster IDs

⏳ STEP 6/9: Defining reward function...
✅ Reward function defined

⏳ STEP 7/9: Setting up CB-GRPO trainer...
✅ CB-GRPO trainer class defined

⏳ STEP 8/9: Initializing trainer...
   ✅ CB-GRPO initialized (n_clusters=16)
✅ Trainer initialized

================================================================================
🔥 STEP 9/9: STARTING CB-GRPO TRAINING
================================================================================

Training for 100 steps...
Expected time: 30-60 min

[Training progress...]

================================================================================
✅ TRAINING COMPLETE!
================================================================================
✅ Model saved: /content/drive/.../final_model
✅ Metrics saved: /content/drive/.../training_metrics.json

================================================================================
🎉 CB-GRPO TRAINING COMPLETE!
================================================================================
End time: 2026-09-08 06:30:00
Output: /content/drive/MyDrive/RLVR_Research/checkpoints/exp2_n1_cbgrpo_0.5B_smoke

Next: Run evaluation cell to test the model!
================================================================================
```

## 🔧 Troubleshooting

### "Cluster file not found"
→ **Fixed!** The cell now creates clusters automatically.

### "Out of memory"
→ Reduce `BATCH_SIZE` from 2 to 1
→ Or reduce `TRAINING_STEPS` from 100 to 50

### "Connection timeout"
→ Models download from HuggingFace. Check internet connection.
→ Or set `HF_TOKEN` in Colab secrets

### Training is slow
→ Normal! 100 steps with 4 generations per step = 400 forward passes
→ Expected: 30-60 minutes on Colab T4 GPU

## 📚 Next Steps

After training completes:

1. **Evaluate the model** - Use evaluation cell (coming soon)
2. **Compare with baseline** - Compare to standard GRPO
3. **Scale up** - Set `USE_SMOKE_TIER = False` for 800 steps
4. **Try different models** - Change `MODEL_NAME` to larger models

## 🎉 You're Ready!

Copy `CELL_CBGRPO_ALL_IN_ONE.txt` to Colab and run it. That's it! 🚀
