# 🔥 Parallel Training Experiments for Your Friend

## Current Status - What's Done

✅ **You (Already Completed):** Vanilla GRPO 0.5B
- Experiment: `exp2_n1_vanilla_0.5B`
- Status: COMPLETE ✅
- Results: Pass@1=25%, Pass@4=65%
- Saved: Baseline for comparison

✅ **You (Currently Running):** CB-GRPO 0.5B  
- Experiment: `exp2_n1_cbgrpo_0.5B`
- Status: Running smoke test (100 steps) → then full 800 steps
- Time: ~30-60 min smoke, then 2-4 hours full
- GPU: Colab T4

---

## 🚀 What Your Friend Can Train in Parallel (Needs GPU)

Since you already have Vanilla 0.5B ✅ and are training CB-GRPO 0.5B, here's what's left:

---

### Option 1: Vanilla GRPO 1.5B ⭐⭐⭐ **HIGHEST PRIORITY**

**What:** Train 1.5B model with standard GRPO (no capacity balancing)
**Why:** Test baseline at larger scale - needed for N1 analysis at both scales
**Time:** 4-6 hours (800 steps)
**GPU:** T4 (works but uses more VRAM ~10-12GB)

#### Why This is Next:
- ✅ You already have Vanilla 0.5B (done)
- ✅ You're running CB-GRPO 0.5B (current)
- ⭐ Need Vanilla 1.5B to compare CB-GRPO 1.5B later
- Part of research: "Does boundary shrinkage scale with model size?"

#### Configuration:
```python
EXPERIMENT_NAME = "exp2_n1_vanilla_0.5B"
MODEL_SIZE = "0.5B"
GATE_TYPE = "VanillaGate"  # ← No capacity balancing
USE_SMOKE_TIER = False  # 800 steps
TRAINING_STEPS = 800
```

#### What Changes:
```python
# In CELL_CBGRPO_ALL_IN_ONE.txt, change:
N_CLUSTERS = 16  # Keep this (still need clusters for dataset)

# But change the trainer to NOT use capacity balancing:
# Instead of CBGRPOTrainer, use standard GRPOTrainer
from trl import GRPOTrainer  # Standard GRPO

trainer = GRPOTrainer(  # ← NOT CBGRPOTrainer
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=[compute_reward_for_grpo],
)
# NO CB-GRPO parameters (no ema_alpha, decay_factor, etc.)
```

#### Output:
- Trained model: `checkpoints/exp2_n1_vanilla_0.5B/`
- Metrics: `results/exp2_n1_vanilla_0.5B/training_metrics.json`

#### How to Compare:
After both trainings finish, you compare:
```python
vanilla_slope = compute_shrinkage_slope(vanilla_pass_at_k, base_pass_at_k)
cbgrpo_slope = compute_shrinkage_slope(cbgrpo_pass_at_k, base_pass_at_k)

improvement = cbgrpo_slope - vanilla_slope  # Should be POSITIVE if CB-GRPO helps!
```

---

### Option 2: CB-GRPO 1.5B ⭐⭐

**What:** Train 1.5B model with CB-GRPO (same as your 0.5B but larger)
**Why:** Test if capacity balancing helps at larger scale
**Time:** 4-6 hours (800 steps)
**GPU:** T4
**When:** After CB-GRPO 0.5B finishes and shows CB-GRPO helps

#### Configuration:
```python
EXPERIMENT_NAME = "exp2_n1_vanilla_1.5B"
MODEL_SIZE = "1.5B"  # ← Larger model
GATE_TYPE = "VanillaGate"
USE_SMOKE_TIER = False
TRAINING_STEPS = 800
```

#### Changes from 0.5B:
```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # ← 1.5B instead of 0.5B
# Everything else same as Vanilla 0.5B
```

#### Why This Helps:
- Tests if boundary shrinkage is worse at larger scale
- Provides 1.5B baseline for future CB-GRPO 1.5B comparison
- Part of your research question: "Does the paradox scale with model size?"

---

### Option 3: Base Model Evaluation (Exp0) ⭐⭐ 

**What:** Evaluate base Qwen 1.5B (no training) for baseline
**Why:** Need this to compute shrinkage slope for 1.5B experiments
**Time:** 1-2 hours (just inference, no training)
**GPU:** T4

#### What to Do:
```python
# Load base model (no training)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Evaluate Pass@k with n=32
evaluate_pass_at_k(model, gsm8k_test, n=32)

# Save: exp0_1.5B_baseline.json
```

#### Why This Helps:
- You need Exp0 baseline to compute Δ Pass@k
- Can be done quickly (just inference)
- Doesn't require training

#### Configuration:
```python
EXPERIMENT_NAME = "exp2_n1_cbgrpo_1.5B"
MODEL_SIZE = "1.5B"
GATE_TYPE = "CBGRPOGate"
USE_SMOKE_TIER = False
TRAINING_STEPS = 800

# CB-GRPO parameters (same as 0.5B)
N_CLUSTERS = 16
EMA_ALPHA = 0.01
DECAY_FACTOR = 0.9
THETA_THRESHOLD = 1.2
```

#### Why Wait:
- Should wait for Vanilla 1.5B baseline first
- Or wait to see if CB-GRPO helps at 0.5B scale
- Lower priority than Vanilla baselines

---

### Option 4: Different Reward Modes 🔬

**What:** Train with different reward configurations
**Why:** Test hypothesis about reward sparsity
**Time:** 2-4 hours each
**GPU:** T4

#### Experiments:
1. **Negative-Only Rewards**
   ```python
   # In reward function:
   # Current: correct=+1.0, incorrect=0.0
   # Change to: correct=0.0, incorrect=-1.0
   ```

2. **Different Format Weights**
   ```python
   # Current: format=0.4, correctness=0.6
   # Test: format=0.2, correctness=0.8
   # Test: format=0.6, correctness=0.4
   ```

#### Why This Helps:
- Tests if reward structure affects boundary shrinkage
- Part of RLVR design space exploration
- Lower priority than baselines

---

### Option 5: Different Clustering (K=8 or K=32) 🔬

**What:** Train CB-GRPO with different number of clusters
**Why:** Test if cluster granularity matters
**Time:** 2-4 hours each
**GPU:** T4

#### Experiments:
1. **Fewer Clusters (K=8)**
   ```python
   N_CLUSTERS = 8  # Instead of 16
   # Broader problem groupings
   ```

2. **More Clusters (K=32)**
   ```python
   N_CLUSTERS = 32  # Instead of 16
   # Finer-grained problem groupings
   ```

#### Why This Helps:
- Tests sensitivity to clustering hyperparameter
- Part of CB-GRPO design validation
- Lower priority

---

## 📋 Updated Training Priority

### Current Status:
- ✅ Vanilla GRPO 0.5B - DONE
- ⏳ CB-GRPO 0.5B - RUNNING (you)
- ❌ Vanilla GRPO 1.5B - NOT STARTED
- ❌ CB-GRPO 1.5B - NOT STARTED

### Recommended Next Steps:

**While you're training CB-GRPO 0.5B:**
1. ⭐⭐⭐ **Friend:** Vanilla GRPO 1.5B (4-6 hours)
   - Runs in parallel with your CB-GRPO 0.5B
   - Provides 1.5B baseline for future experiments

**After CB-GRPO 0.5B finishes:**
2. **Evaluate and compare:** CB-GRPO 0.5B vs Vanilla 0.5B
   - Compute Δ slope
   - If CB-GRPO helps → proceed to 1.5B scale
   - If CB-GRPO doesn't help → analyze why

**If CB-GRPO helped at 0.5B:**
3. **You or Friend:** CB-GRPO 1.5B (4-6 hours)
   - Test if CB-GRPO scales to larger models
   - Compare with Vanilla 1.5B baseline (from step 1)

---

## 🎯 Why Vanilla GRPO 0.5B is Most Important

### Without Vanilla Baseline:
❌ Can't prove CB-GRPO helps
❌ Can't compute Δ slope = slope_cbgrpo - slope_vanilla
❌ Can't claim novelty (you need comparison!)
❌ Paper reviewers will ask: "How do you know CB-GRPO is better?"

### With Vanilla Baseline:
✅ Prove CB-GRPO reduces boundary shrinkage
✅ Quantify improvement: "CB-GRPO improves shrinkage slope by X%"
✅ Validate N1 novelty claim
✅ Answer reviewer questions with data

---

## 🔄 How to Set Up for Your Friend

### Step 1: Copy Your Working Cell
Your friend should copy `CELL_CBGRPO_ALL_IN_ONE.txt` and make ONE change:

```python
# CHANGE THIS SECTION (around line 340-360):

# OLD (CB-GRPO):
from trl import GRPOTrainer, GRPOConfig

class CBGRPOTrainer(GRPOTrainer):
    # ... CB-GRPO capacity balancing code ...

trainer = CBGRPOTrainer(
    n_clusters=N_CLUSTERS,
    ema_alpha=EMA_ALPHA,
    # ... CB-GRPO parameters ...
)

# NEW (Vanilla GRPO):
from trl import GRPOTrainer, GRPOConfig

# NO custom trainer class needed!

trainer = GRPOTrainer(  # ← Standard GRPO
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=[compute_reward_for_grpo],
)
```

### Step 2: Change Experiment Name
```python
# At top of cell:
EXPERIMENT_NAME = "exp2_n1_vanilla_0.5B"  # ← Changed from cbgrpo to vanilla
```

### Step 3: Run!
That's it! Everything else stays the same:
- ✅ Same clustering
- ✅ Same reward function
- ✅ Same dataset
- ✅ Same training steps (800)
- ✅ Same model (0.5B)

**Only difference:** No capacity balancing in the trainer

---

## 📊 What Results to Compare

After both trainings finish, compare these metrics:

### 1. Pass@k Results
```
Model         | Pass@1 | Pass@4 | Pass@16 | Pass@32
--------------|--------|--------|---------|--------
Vanilla 0.5B  | 0.45   | 0.67   | 0.82    | 0.87
CB-GRPO 0.5B  | 0.47   | 0.71   | 0.85    | 0.90
```

### 2. Shrinkage Slope
```
Model         | Slope   | Improvement
--------------|---------|------------
Vanilla 0.5B  | -0.15   | baseline
CB-GRPO 0.5B  | -0.08   | +47% better!
```

### 3. Transition Matrix
```
Category            | Vanilla | CB-GRPO
--------------------|---------|--------
Kept Correct        | 350     | 380
Lost Capability     | 50      | 20  ← CB-GRPO retains more!
Gained Capability   | 100     | 120
Kept Incorrect      | 500     | 480
```

### 4. Training Metrics
```
Metric              | Vanilla | CB-GRPO
--------------------|---------|--------
Final Loss          | 1.23    | 1.18
Format Success Rate | 0.85    | 0.87
Correctness Rate    | 0.42    | 0.46
```

---

## ⏱️ Timeline

| Time | You (Account 1) | Friend (Account 2) |
|------|-----------------|-------------------|
| **Now** | CB-GRPO 0.5B smoke (30-60 min) | Setup Vanilla GRPO 0.5B |
| **Hour 0-4** | CB-GRPO 0.5B full (800 steps) | Vanilla GRPO 0.5B (800 steps) |
| **Hour 4-5** | Evaluate CB-GRPO results | Evaluate Vanilla results |
| **Hour 5-6** | Compare and compute Δ slope | Share results |
| **Done!** | Paper results ready! 🎉 | |

**Both trainings run in parallel = Same total time as running them sequentially!** 🚀

---

## 📁 Files to Share with Friend

1. **Copy of your working cell:**
   - `CELL_CBGRPO_ALL_IN_ONE.txt`

2. **Modification instructions:**
   - Remove `CBGRPOTrainer` class
   - Use standard `GRPOTrainer`
   - Change experiment name

3. **Same cluster file:**
   - They can reuse your `cluster_assignments.pkl` from Drive
   - Or run clustering cell themselves (5 min)

4. **Troubleshooting:**
   - `ERRORS_FIXED.md` - All errors already solved!

---

## 🎯 Bottom Line

**Most Important Task for Friend:** Vanilla GRPO 0.5B ⭐⭐⭐

**Tell your friend:**
> "Can you train standard GRPO (without capacity balancing) on 0.5B model for 800 steps? I'm training CB-GRPO right now, and I need the vanilla baseline to compare against. Just use the same cell but remove the CBGRPOTrainer class and use standard GRPOTrainer instead. Change experiment name to 'exp2_n1_vanilla_0.5B'. Should take 2-4 hours."

**Why this is perfect:**
- ✅ Same time investment as your training
- ✅ Runs in parallel (no waiting)
- ✅ Provides critical baseline for comparison
- ✅ Required for your N1 novelty claim
- ✅ Uses same GPU (T4), same dataset, same reward
- ✅ Just one simple code change

**After both finish:**
You compare results and have your N1 novelty validated with data! 🎉

---

## 🤝 Collaboration Flow

```
You                           Friend
│                            │
├─ CB-GRPO 0.5B training    ├─ Vanilla GRPO 0.5B training
│  (with capacity balance)   │  (no capacity balance)
│                            │
├─ Save: cbgrpo_results.json ├─ Save: vanilla_results.json
│                            │
└─────────┬──────────────────┘
          │
          ├─ Compare results
          ├─ Compute Δ slope
          ├─ Generate paper figures
          └─ Write paper! 🎉
```

Both trainings are **completely independent** - no conflicts, no waiting! 🚀
