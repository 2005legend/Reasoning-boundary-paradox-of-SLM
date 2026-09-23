# CB-GRPO Implementation Fixes - Summary

## Date: 2026-09-09

## Critical Issues Fixed

### ✅ Priority 1: Correctness Reward (FIXED)
**Problem:** Reward function gave 0.6 points just for having `<answer>` tags, regardless of correctness.

**Fix:** Now checks extracted answer against `ground_truth`:
```python
if abs(float(extracted) - float(ground_truth[i])) < 1e-4:
    reward += 0.8  # Actual correctness signal
```

**Impact:** Training now optimizes for correct answers, not just format.

---

### ✅ Priority 2: CB-GRPO Mechanism (FIXED - Sampler Approach)
**Problem:** Reward-scaling approach gets canceled out by GRPO's group-relative normalization.

**Why it fails:**
- GRPO normalizes within each group: `advantage_i = (reward_i - mean) / std`
- Scaling all rewards in a group by constant `w` cancels out: `(w·r - w·mean) / (w·std) = (r - mean) / std`
- Since all 4 generations share the same cluster_id (same prompt), they all get the same weight `w`
- Result: CB-GRPO had no effect!

**Fix:** Use sampler-based throttling instead:
```python
class CBGRPOSampler:
    """Throttles how often over-spent clusters get sampled."""
    def __iter__(self):
        # Compute sampling probability based on spend
        probs = decay ** max(0, (spend[c] / mean_spend) - theta)
        # Sample indices according to probs
        return iter(sampled_indices)
```

**Impact:** Now actually throttles over-represented clusters by controlling **which prompts enter training**, not by scaling their rewards.

---

### ✅ Priority 3: Implementation Approach (FIXED)
**Problem:** Previous `CBGRPOTrainer.compute_loss()` override was:
1. Fighting TRL's internal APIs (fragile)
2. Operating on aggregated batch loss (loses per-sample info)
3. Applied same weight to entire batch (no per-cluster differentiation)

**Fix:** 
1. **Spend tracking:** `CBGRPOSpendTracker` updates `spend_ema` in reward function
2. **Throttling:** `CBGRPOSampler` controls sampling frequency via `get_train_dataloader()`
3. **Integration:** `CBGRPOTrainer` only overrides stable, documented `get_train_dataloader()` method

**Impact:** Robust implementation using only public TRL APIs.

---

## Implementation Summary

### Components:

1. **Reward Function** (with correctness check)
   ```python
   def compute_reward_for_grpo(prompts, completions, ground_truth=None, **kwargs):
       # Format: 0.1 + 0.1 = 0.2
       # Correctness: 0.8
       # Total: 1.0
   ```

2. **Spend Tracker** (updates spend_ema)
   ```python
   class CBGRPOSpendTracker:
       def __call__(self, ...):
           rewards = base_reward_fn(...)
           # Update spend_ema based on reward magnitude
           for r, p in zip(rewards, problems):
               spend_ema[cluster_id] = ema_alpha * |r| + (1 - ema_alpha) * spend_ema[cluster_id]
           return rewards  # Unmodified
   ```

3. **Custom Sampler** (throttles over-spent clusters)
   ```python
   class CBGRPOSampler(torch.utils.data.Sampler):
       def __iter__(self):
           probs = [decay ** max(0, (spend[c]/mean - theta)) for c in cluster_ids]
           return iter(sample_with_replacement(probs))
   ```

4. **Trainer** (uses custom sampler)
   ```python
   class CBGRPOTrainer(GRPOTrainer):
       def get_train_dataloader(self):
           return DataLoader(..., sampler=CBGRPOSampler(...))
   ```

---

## What Changed from Previous Run

| Aspect | Old (Invalid) | New (Correct) |
|--------|---------------|---------------|
| **Correctness** | No check (0.6 for any number) | Checks ground_truth (0.8 only if correct) |
| **CB-GRPO mechanism** | Reward scaling (canceled by GRPO) | Sampling frequency throttling |
| **Implementation** | compute_loss override | get_train_dataloader override |
| **API stability** | Private TRL internals | Public documented API |

---

## Action Required

**Current running training (step 415/800):**
- ❌ Has NO correctness reward
- ❌ CB-GRPO mechanism is no-op (reward scaling canceled)
- ❌ Only learning format, not solving math

**Decision:** STOP and restart with fixed code

**New training will:**
- ✅ Optimize for correctness (80% of reward)
- ✅ Actually throttle over-represented clusters
- ✅ Produce scientifically valid results

---

## Files Updated

- `02_notebook_cells/CELL_CBGRPO_ALL_IN_ONE.txt` - All fixes applied
- `07_for_friend/VANILLA_GRPO_1.5B_STANDALONE.txt` - Needs correctness fix too

---

## Next Steps

1. **Stop current Kaggle training**
2. **Copy updated CELL_CBGRPO_ALL_IN_ONE.txt to Kaggle**
3. **Run from step 0** (fresh start)
4. **Monitor for CB-GRPO sampler logs** (every 50 steps)
5. **Verify correctness rewards are non-zero** (should see 0.8 rewards)

---

## Verification Checklist

Before trusting results:
- [ ] See CB-GRPO logs: `[CB-GRPO] step=50 spend range 0.xxx-0.xxx`
- [ ] Loss is non-zero and increasing (GRPO expected behavior)
- [ ] Rewards include 0.8 values (correctness achieved)
- [ ] Training completes 800 steps without crashes
- [ ] Final model checkpoints saved properly

---

## Timeline Impact

- **Time invested so far:** ~17 hours (invalid training)
- **Time required for valid run:** ~10-12 hours (full 800 steps)
- **Total research time:** ~27-29 hours for CB-GRPO 0.5B

Worth it for scientifically valid results! 🚀
