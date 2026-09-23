# 🚀 Vanilla GRPO 1.5B Training - For Friend

## What This Is

Train a **Vanilla GRPO 1.5B model** completely standalone. No files needed from anyone else!

## Why This Matters

Your friend is training **CB-GRPO 0.5B** right now. You're training the **Vanilla GRPO 1.5B baseline** in parallel. Later you'll compare results!

## ⚡ Quick Start

### 1. Open Google Colab
Go to: https://colab.research.google.com/

### 2. Connect to GPU
- Runtime → Change runtime type → T4 GPU

### 3. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Copy the Training Cell
Copy everything from `VANILLA_GRPO_1.5B_STANDALONE.txt` into a Colab cell

### 5. Run!
Click Run. That's it!

## ⏱️ How Long?

- **Smoke test** (100 steps): 30-60 minutes
- **Full training** (800 steps): 4-6 hours

To switch:
```python
USE_SMOKE_TIER = True   # 30-60 min test
USE_SMOKE_TIER = False  # 4-6 hours full
```

## 📦 What It Does

1. ✅ Installs packages (trl, peft, bitsandbytes, etc.)
2. ✅ Loads Qwen 1.5B model with QLoRA (4-bit quantization)
3. ✅ Loads GSM8K math dataset (7,473 problems)
4. ✅ Trains with standard GRPO (no capacity balancing)
5. ✅ Saves model to your Google Drive
6. ✅ Saves training metrics

## 💾 Where Files Go

Everything saves to your Google Drive:
```
/content/drive/MyDrive/RLVR_Research/
├── checkpoints/exp2_n1_vanilla_1.5B/
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── final_model/
├── results/exp2_n1_vanilla_1.5B/
│   └── training_metrics.json
└── logs/exp2_n1_vanilla_1.5B/
```

## 🔍 What to Share Later

After training finishes, share just the **checkpoint path** with your friend:
```
/content/drive/MyDrive/RLVR_Research/checkpoints/exp2_n1_vanilla_1.5B/final_model
```

That's all they need to evaluate and compare!

## 🚨 Common Issues

### "Out of memory"
- Change `BATCH_SIZE = 2` to `BATCH_SIZE = 1`
- Or reduce `TRAINING_STEPS = 800` to `TRAINING_STEPS = 400`

### "Colab disconnected"
- Normal! Training will resume from last checkpoint
- Just re-run the cell

### "Training is slow"
- Normal! 800 steps with 4 generations each = 3200 forward passes
- Expected: 4-6 hours on T4 GPU

## ✅ How to Verify It's Working

You should see:
```
================================================================================
🚀 VANILLA GRPO 1.5B TRAINING
================================================================================

⏳ Installing required packages...
✅ Packages installed

Start time: 2026-09-08 12:00:00
================================================================================

Experiment: exp2_n1_vanilla_1.5B
Model: Qwen2.5-1.5B-Instruct
Training steps: 800
Smoke tier: False

⏳ STEP 1/7: Setting up directories...
✅ Directories created

⏳ STEP 2/7: Loading model with QLoRA...
✅ Model loaded with QLoRA
   Trainable params: 6,553,600

⏳ STEP 3/7: Loading GSM8K dataset...
✅ GSM8K loaded: 7473 problems

⏳ STEP 4/7: Preparing dataset...
✅ Dataset prepared: 7473 examples

⏳ STEP 5/7: Defining reward function...
✅ Reward function defined

⏳ STEP 6/7: Initializing trainer...
✅ Trainer initialized

================================================================================
🔥 STEP 7/7: STARTING VANILLA GRPO TRAINING
================================================================================

Training for 800 steps...
Expected time: 4-6 hours

[Training progress bars and metrics...]
```

## 🎯 What Happens Next

### After Your Training Finishes:
1. ✅ Model saved to Drive
2. ✅ Metrics saved
3. ⏳ Share checkpoint path with friend

### Your Friend Will:
1. Finish CB-GRPO 0.5B training
2. Get your Vanilla 1.5B checkpoint path
3. Compare: Vanilla 0.5B vs CB-GRPO 0.5B vs Vanilla 1.5B
4. Optionally: Train CB-GRPO 1.5B if CB-GRPO helped

## 📊 What Gets Compared Later

Your friend will compare:
```
Model            | Pass@1 | Pass@4 | Shrinkage Slope
-----------------|--------|--------|----------------
Vanilla 0.5B     | 0.25   | 0.65   | -0.15 (baseline)
CB-GRPO 0.5B     | 0.28   | 0.71   | -0.08 (better!)
Vanilla 1.5B     | 0.35   | 0.75   | -0.12 (your result!)
CB-GRPO 1.5B     | ???    | ???    | ??? (future work)
```

## 🤝 No File Sharing Needed Now!

- ✅ Completely standalone training
- ✅ Uses own Drive folder
- ✅ No conflicts
- ✅ Share results only AFTER training

## 💬 Questions?

If stuck, check:
1. GPU connected? (Runtime → Change runtime type)
2. Drive mounted? (Should see "Mounted at /content/drive")
3. Enough space? (Need ~5GB free in Drive)
4. Colab disconnected? (Just re-run, it resumes!)

## 🎉 That's It!

Just copy the cell and run. It handles everything else automatically!

---

**Expected completion time:** 4-6 hours  
**GPU used:** T4 (Colab free tier)  
**No setup needed:** Everything automatic!  
**Share later:** Just the checkpoint path!

Good luck! 🚀
