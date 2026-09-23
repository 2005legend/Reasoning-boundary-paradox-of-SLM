# CB-GRPO Training - Errors Fixed Log

## Session: 2026-09-08
**Goal:** Train CB-GRPO model on Google Colab with Qwen 0.5B + QLoRA

---

## Error 1: Missing Cluster File
**Error:**
```
FileNotFoundError: Run clustering cell first!
❌ Cluster file not found: /content/drive/MyDrive/RLVR_Research/cluster_assignments.pkl
```

**Root Cause:** Cell required pre-existing cluster assignments

**Fix:** Added auto-clustering logic to STEP 1
```python
if cluster_file.exists():
    # Load existing clusters
else:
    # Create clusters automatically with KMeans + sentence-transformers
```

**Result:** ✅ Cell now creates clusters on first run, loads on subsequent runs

---

## Error 2: Missing `trl` Package
**Error:**
```
ModuleNotFoundError: No module named 'trl'
```

**Root Cause:** Missing TRL (Transformers Reinforcement Learning) library

**Fix:** Added to package installation list
```python
packages_to_install = [
    "bitsandbytes>=0.46.1",
    "accelerate",
    "sentence-transformers",
    "scikit-learn",
    "trl",           # ← ADDED
    "peft",          # ← ADDED
    "datasets",      # ← ADDED
]
```

**Result:** ✅ All required packages now install automatically

---

## Error 3: Missing `reward_funcs` Parameter
**Error:**
```
TypeError: compute_reward_for_grpo() missing 1 required positional argument: 'samples'
```

**Root Cause:** Trainer initialized without reward function

**Fix:** Added reward function and passed to trainer
```python
def compute_reward_for_grpo(prompts, completions, **kwargs):
    # Reward based on format + answer extraction
    return rewards

trainer = CBGRPOTrainer(
    # ... other params ...
    reward_funcs=[compute_reward_for_grpo],  # ← ADDED
)
```

**Result:** ✅ Reward function now properly integrated

---

## Error 4: Wrong Parameter Names (TRL Compatibility)
**Error:**
```
TypeError: GRPOTrainer.__init__() got an unexpected keyword argument 'tokenizer'
TypeError: unexpected keyword argument 'warmup_ratio'
TypeError: unexpected keyword argument 'num_generation_per_prompt'
```

**Root Cause:** Parameter names changed in TRL 0.12+

**Fix:** Updated to new parameter names
```python
# OLD (TRL < 0.12)          # NEW (TRL >= 0.12)
tokenizer=tokenizer         → processing_class=tokenizer
warmup_ratio=0.1           → warmup_steps=10
num_generation_per_prompt=4 → num_generations=4
```

**Result:** ✅ Compatible with latest TRL version

---

## Error 5: Missing `num_items_in_batch` Parameter
**Error:**
```
TypeError: CBGRPOTrainer.compute_loss() got an unexpected keyword argument 'num_items_in_batch'
```

**Root Cause:** Transformers 4.48+ passes `num_items_in_batch` to `compute_loss()`

**Fix:** Updated compute_loss signature
```python
# OLD
def compute_loss(self, model, inputs, return_outputs=False):

# NEW
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
```

**Result:** ✅ Compatible with Transformers 4.48+

---

## Error 6: GRPOTrainer Does Not Support return_outputs
**Error:**
```
ValueError: The GRPOTrainer does not support returning outputs
```

**Root Cause:** GRPOTrainer explicitly forbids `return_outputs=True`

**Fix:** Changed compute_loss to not request outputs
```python
# OLD
outputs = super().compute_loss(model, inputs, return_outputs=True, ...)
loss = outputs[0] if isinstance(outputs, tuple) else outputs
return (loss, outputs[1]) if return_outputs else loss

# NEW
loss = super().compute_loss(model, inputs, return_outputs=False, ...)
# ... CB-GRPO logic ...
return loss
```

**Result:** ✅ Works with GRPOTrainer's constraints

---

## Error 7: Gradient Scaler with BFloat16
**Error:**
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

**Root Cause:** Gradient scaler (from fp16/bf16) incompatible with 4-bit quantization

**Fix:** Disabled mixed precision training
```python
training_args = GRPOConfig(
    # ... other params ...
    bf16=False,  # ← Disable bfloat16
    fp16=False,  # ← Disable fp16 (no gradient scaler with 4-bit)
)
```

**Result:** ✅ Training runs in FP32 (standard for QLoRA)

---

## Final Working Configuration

### Packages Required:
```python
bitsandbytes>=0.46.1  # QLoRA quantization
accelerate            # Distributed training
sentence-transformers # Prompt embeddings
scikit-learn         # KMeans clustering
trl                  # GRPO trainer
peft                 # LoRA adapters
datasets             # HuggingFace datasets
```

### Key Parameters:
```python
# Training Args
processing_class=tokenizer  # NOT tokenizer=
warmup_steps=10            # NOT warmup_ratio=
num_generations=4          # NOT num_generation_per_prompt=
bf16=False                 # Disable mixed precision
fp16=False                 # Disable gradient scaler

# Trainer Init
reward_funcs=[compute_reward_for_grpo]  # Required

# compute_loss signature
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
    # ... CB-GRPO logic ...
    return loss  # Just loss, no outputs tuple
```

---

## Timeline

| Time | Issue | Status |
|------|-------|--------|
| 05:16 | FileNotFoundError (clusters) | ✅ Fixed |
| 05:21 | ModuleNotFoundError (trl) | ✅ Fixed |
| 05:26 | Missing reward function | ✅ Fixed |
| 05:31 | Wrong parameter names | ✅ Fixed |
| 05:46 | num_items_in_batch signature | ✅ Fixed |
| 05:46 | GRPOTrainer return_outputs | ✅ Fixed |
| 06:15 | Gradient scaler BFloat16 | ✅ Fixed |
| 06:20 | **🎉 TRAINING STARTED** | ✅ **SUCCESS** |

---

## Lessons Learned

1. **TRL/Transformers Compatibility:** Always check parameter names when upgrading libraries
2. **QLoRA + Mixed Precision:** 4-bit quantization doesn't work with gradient scaler (fp16/bf16)
3. **GRPOTrainer Constraints:** Can't use `return_outputs=True` in compute_loss
4. **Auto-Clustering:** Better to auto-create dependencies than fail with FileNotFoundError
5. **Smoke Tests:** 100-step smoke test catches errors before committing to long training runs

---

## Quick Reference: Common TRL/GRPO Errors

### ✅ DO:
- Use `processing_class=tokenizer`
- Use `warmup_steps` (integer)
- Use `num_generations` (integer)
- Set `bf16=False, fp16=False` for QLoRA
- Accept `num_items_in_batch` in compute_loss
- Call parent compute_loss with `return_outputs=False`
- Pass `reward_funcs=[func]` to trainer

### ❌ DON'T:
- Use `tokenizer=tokenizer` (old parameter name)
- Use `warmup_ratio` (old parameter name)
- Use `num_generation_per_prompt` (old parameter name)
- Use fp16/bf16 with 4-bit quantization
- Request `return_outputs=True` from GRPOTrainer
- Return outputs tuple from compute_loss in GRPO

---

## File Status: ✅ READY

**File:** `CELL_CBGRPO_ALL_IN_ONE.txt`

**Features:**
- ✅ Auto-creates clusters on first run
- ✅ All packages auto-install
- ✅ Reward function integrated
- ✅ Compatible with TRL 0.12+
- ✅ Compatible with Transformers 4.48+
- ✅ Works with 4-bit QLoRA
- ✅ Smoke test mode (100 steps)
- ✅ Full training mode (800 steps)

**Usage:**
1. Copy to Colab
2. Mount Google Drive
3. Run cell
4. Wait 30-60 min (smoke) or 2-4 hours (full)

---

**Status:** ALL ERRORS FIXED ✅  
**Date:** 2026-09-08  
**Model:** Qwen/Qwen2.5-0.5B-Instruct  
**Method:** CB-GRPO with QLoRA  
**Result:** Training successfully started 🎉
