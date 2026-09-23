# ✅ Parameters Verification - Friend's Vanilla 1.5B

## Training Parameters Comparison

| Parameter | Your Vanilla 0.5B (CELL_21) | Friend's Vanilla 1.5B | Status |
|-----------|----------------------------|---------------------|--------|
| **Model** | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | ✅ Different (intended) |
| **Training Steps (full)** | 800 | 800 | ✅ SAME |
| **Training Steps (smoke)** | 100 | 100 | ✅ SAME |
| **Checkpoint Interval (full)** | 200 | 200 | ✅ SAME |
| **Checkpoint Interval (smoke)** | 50 | 50 | ✅ SAME |
| **Batch Size** | 4 | 4 | ✅ SAME |
| **Gradient Accumulation** | 1 | 1 | ✅ SAME |
| **Learning Rate** | 5e-5 | 5e-5 | ✅ SAME |
| **LR Scheduler** | cosine | cosine | ✅ SAME |
| **Warmup Steps** | 10% of steps | 10% of steps | ✅ SAME |
| **Beta (KL penalty)** | 0.1 | 0.1 | ✅ SAME |
| **Rollouts per Prompt** | 4 | 4 | ✅ SAME |
| **Max New Tokens** | 512 | 512 | ✅ SAME |
| **Temperature** | 0.7 | 0.7 | ✅ SAME |
| **Top P** | 0.95 | 0.95 | ✅ SAME |
| **Gradient Checkpointing** | True | True | ✅ SAME |
| **BF16** | True | True | ✅ SAME |
| **Save Total Limit** | 3 | 3 | ✅ SAME |
| **Random Seed** | 42 | 42 | ✅ SAME |

---

## Training Timeline

### Full Training (USE_SMOKE_TIER = False)
```python
TRAINING_STEPS = 800
CHECKPOINT_INTERVAL = 200

# Checkpoints saved at:
- Step 200
- Step 400
- Step 600
- Step 800 (final)

# Plus:
- checkpoint-last (latest)
- final_model (at end)

# Total: 6 checkpoints max (keeps last 3 + final)
```

### Smoke Test (USE_SMOKE_TIER = True)
```python
TRAINING_STEPS = 100
CHECKPOINT_INTERVAL = 50

# Checkpoints saved at:
- Step 50
- Step 100 (final)

# Total: 3 checkpoints
```

---

## Expected Outputs

### File Structure (Full Training):
```
/content/drive/MyDrive/RLVR_Research/
└── checkpoints/
    └── exp2_n1_vanilla_1.5B/
        ├── checkpoint-200/
        │   ├── model.safetensors
        │   ├── adapter_config.json
        │   └── ...
        ├── checkpoint-400/
        ├── checkpoint-600/
        ├── checkpoint-800/
        └── final_model/
            ├── model.safetensors
            ├── adapter_config.json
            └── ...
```

### Results:
```
/content/drive/MyDrive/RLVR_Research/
└── results/
    └── exp2_n1_vanilla_1.5B/
        └── training_metrics.json
```

---

## Checkpoint Sizes (Approximate)

| Model | Checkpoint Size | Total for 4 checkpoints |
|-------|----------------|------------------------|
| 0.5B | ~30-50 MB | ~150 MB |
| 1.5B | ~80-120 MB | ~400 MB |

**Note:** These are LoRA adapters only (not full model), so quite small!

---

## Training Time Estimates

### 0.5B Model:
- **Steps:** 800
- **Time:** 2-4 hours
- **VRAM:** ~6-8 GB

### 1.5B Model (Friend):
- **Steps:** 800
- **Time:** 4-6 hours
- **VRAM:** ~10-12 GB

**Why longer?** 
- Larger model = more parameters
- More VRAM usage = slightly slower
- Same steps, but each step takes longer

---

## Effective Training Computation

```python
# Per step computation:
Batch size: 4
Gradient accumulation: 1
Rollouts per prompt: 4

# Effective:
Problems per step = 4 * 1 = 4
Solutions per step = 4 * 4 = 16
Total forward passes per step = 16

# Full training:
Total steps = 800
Total problems = 800 * 4 = 3,200
Total solutions = 800 * 16 = 12,800
```

---

## Reward Function Parameters

| Component | Weight | Check |
|-----------|--------|-------|
| Format (<reasoning> tags) | 0.2 | ✅ |
| Format (<answer> tags) | 0.2 | ✅ |
| Correctness (answer match) | 0.8 | ✅ |
| **Total Max** | **1.2** | ✅ |

**Note:** Total can exceed 1.0 (format + correctness both succeed)

---

## LoRA Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| r (rank) | 16 | Low-rank dimension |
| lora_alpha | 32 | Scaling factor |
| Target modules | q_proj, v_proj, k_proj, o_proj | Attention layers |
| lora_dropout | 0.05 | Regularization |
| bias | none | No bias training |

**Trainable Parameters:**
- 0.5B model: ~2.1M parameters (0.4% of full model)
- 1.5B model: ~6.5M parameters (0.4% of full model)

---

## Quantization Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| Load in 4-bit | True | Memory efficiency |
| Quant type | nf4 | Normal Float 4-bit |
| Compute dtype | float16 | Computation precision |
| Double quant | True | Additional compression |

**Memory Savings:**
- Full 1.5B model: ~3 GB
- 4-bit quantized: ~800 MB
- With LoRA: ~1 GB total

---

## Key Differences from Your CB-GRPO

### Your CB-GRPO 0.5B:
```python
# Has custom trainer
class CBGRPOTrainer(GRPOTrainer):
    def __init__(self, n_clusters=16, ema_alpha=0.01, ...):
        # Capacity balancing logic
        
# Tracks cluster spend
self.spend_ema = torch.ones(n_clusters)

# Applies gating
weights = compute_capacity_weights(cluster_ids)
loss = loss * weights.mean()
```

### Friend's Vanilla 1.5B:
```python
# Standard trainer (no custom class)
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=compute_reward_for_grpo,
)

# NO cluster tracking
# NO capacity balancing
# NO gating
```

**This is the baseline!** Standard GRPO without modifications.

---

## What Friend Needs to Share After Training

### Minimum:
1. **Checkpoint path:**
   ```
   /content/drive/MyDrive/RLVR_Research/checkpoints/exp2_n1_vanilla_1.5B/final_model
   ```

2. **Training metrics:**
   ```
   /content/drive/MyDrive/RLVR_Research/results/exp2_n1_vanilla_1.5B/training_metrics.json
   ```

### Nice to Have:
3. Final loss value
4. Final reward value
5. Training time (total hours)

---

## Verification Checklist for Friend

After training completes, verify:

- [ ] Final checkpoint exists at `checkpoints/exp2_n1_vanilla_1.5B/final_model/`
- [ ] Metrics JSON saved at `results/exp2_n1_vanilla_1.5B/training_metrics.json`
- [ ] 4 checkpoints exist (200, 400, 600, 800) or kept last 3
- [ ] No errors in final output
- [ ] GPU memory was released (cleanup ran)

If all checked ✅, training successful!

---

## Quick Reference

### To Start:
```python
USE_SMOKE_TIER = False  # For full 800 steps
# or
USE_SMOKE_TIER = True   # For quick 100-step test
```

### Expected Runtime:
- Smoke (100 steps): 30-60 minutes
- Full (800 steps): 4-6 hours

### Expected VRAM:
- Allocated: ~10-12 GB
- Reserved: ~12-14 GB
- Total T4 GPU: 16 GB
- **Safe!** ✅

### Checkpoints:
- Every 200 steps (full mode)
- Every 50 steps (smoke mode)
- Keeps last 3 + final

---

## ✅ Final Verdict

All parameters are **correctly configured** for:
1. Matching your Vanilla 0.5B training (proven to work)
2. Scaled to 1.5B model (only change)
3. Same training regimen (800 steps, checkpoints every 200)
4. Same hyperparameters (batch size, learning rate, etc.)

**Ready to send to friend!** 🚀
