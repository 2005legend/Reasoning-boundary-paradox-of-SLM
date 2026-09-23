# What's Different: Vanilla GRPO vs CB-GRPO

## Quick Comparison

| Feature | Your Friend's (Vanilla 1.5B) | You (CB-GRPO 0.5B) |
|---------|----------------------------|-------------------|
| **Model** | Qwen 1.5B | Qwen 0.5B |
| **Method** | Standard GRPO | CB-GRPO (with capacity balancing) |
| **Clusters** | No clustering needed | 16 clusters for balancing |
| **Trainer** | `GRPOTrainer` | `CBGRPOTrainer` (custom) |
| **Time** | 4-6 hours | 2-4 hours |
| **VRAM** | ~10-12GB | ~6-8GB |

## Code Differences

### Your Friend's Code (Vanilla):
```python
# Simple - no clustering
train_data.append({
    "prompt": prompt,
    "problem": problem,
    "ground_truth": answer,
    # No cluster_id
})

# Standard trainer
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=[compute_reward],
    # No CB-GRPO parameters
)
```

### Your Code (CB-GRPO):
```python
# Includes clustering
train_data.append({
    "prompt": prompt,
    "problem": problem,
    "ground_truth": answer,
    "cluster_id": cluster_id,  # ← Added
})

# Custom trainer with capacity balancing
class CBGRPOTrainer(GRPOTrainer):
    def __init__(self, n_clusters, ema_alpha, ...):
        # Capacity balancing logic
        self.spend_ema = torch.ones(n_clusters)
        # ...
    
    def compute_loss(self, ...):
        # Apply capacity-aware reweighting
        # ...

trainer = CBGRPOTrainer(
    n_clusters=16,           # ← CB-GRPO param
    ema_alpha=0.01,          # ← CB-GRPO param
    decay_factor=0.9,        # ← CB-GRPO param
    theta_threshold=1.2,     # ← CB-GRPO param
    balance_window=50,       # ← CB-GRPO param
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=[compute_reward],
)
```

## Why Two Different Approaches?

### Vanilla GRPO (Friend):
- **Goal:** Establish baseline performance
- **Method:** Standard reinforcement learning (GRPO)
- **Expected:** Will show boundary shrinkage paradox
- **Use:** Compare against to prove CB-GRPO helps

### CB-GRPO (You):
- **Goal:** Mitigate boundary shrinkage paradox
- **Method:** Capacity-balanced gradient reweighting
- **Expected:** Will reduce boundary shrinkage
- **Use:** Prove the novel algorithm works

## What You're Testing

```
Research Question:
"Does capacity balancing (CB-GRPO) reduce boundary shrinkage
compared to standard GRPO?"

Vanilla GRPO 0.5B  →  Shows the problem exists (baseline)
CB-GRPO 0.5B       →  Shows CB-GRPO helps (novel method)
Vanilla GRPO 1.5B  →  Shows problem at larger scale (friend)
CB-GRPO 1.5B       →  Shows CB-GRPO helps at scale (future)
```

## Standalone = No Dependencies

### Your Friend's Cell:
- ✅ Self-contained
- ✅ No clustering files needed
- ✅ No sharing required until after
- ✅ Own Drive folder
- ✅ Own experiment name

### Your Cell:
- ✅ Also self-contained
- ✅ Auto-creates clusters
- ✅ Own Drive folder
- ✅ Own experiment name

**No conflicts! Both can run simultaneously!** 🚀

## After Both Finish

You'll share:
1. Checkpoint paths
2. Training metrics (JSON files)
3. Run evaluation together
4. Compare results
5. Write paper!

But **during training**: Completely independent! ✅
