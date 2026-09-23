# 🚀 Train Vanilla GRPO 1.5B - Quick Setup

Hey! Can you help train a baseline model while I train CB-GRPO? This is completely standalone - you don't need any files from me!

## What to Do

### 1. Open this file:
`VANILLA_GRPO_1.5B_STANDALONE.txt`

### 2. Copy everything in that file

### 3. In Google Colab:
- Click "New notebook"
- Change runtime to T4 GPU (Runtime → Change runtime type)
- Mount Drive (it will show a code snippet)
- Paste the code
- Click Run!

## How Long?

- **Quick test:** 30-60 minutes (100 steps)
- **Full training:** 4-6 hours (800 steps)

Change this line to switch:
```python
USE_SMOKE_TIER = True   # Quick test
USE_SMOKE_TIER = False  # Full training
```

## What It Does

Trains a 1.5B model with standard GRPO on math problems. Everything is automatic:
- ✅ Installs packages
- ✅ Loads model
- ✅ Loads dataset
- ✅ Trains
- ✅ Saves to your Drive

## No Setup Needed!

- ❌ No files to download
- ❌ No configuration
- ❌ No dependencies on my work
- ✅ Completely standalone!

## What to Share After

After it finishes, just send me:
1. The checkpoint path (prints at the end)
2. The `training_metrics.json` file

That's it! Later we'll evaluate and compare results.

## Why This Helps

I'm training **CB-GRPO 0.5B** (novel method)  
You're training **Vanilla GRPO 1.5B** (baseline)

Both run at the same time = no waiting! 🚀

After both finish, we compare:
- Does CB-GRPO help at 0.5B? (my result vs existing baseline)
- Does the problem exist at 1.5B? (your result)
- Should we train CB-GRPO 1.5B next? (if CB-GRPO helped)

## Files for You

1. **`VANILLA_GRPO_1.5B_STANDALONE.txt`** ← Copy this to Colab
2. **`README_FOR_FRIEND.md`** ← Detailed instructions
3. **`WHATS_DIFFERENT.md`** ← What makes this different from mine

## Questions?

Check `README_FOR_FRIEND.md` for:
- Troubleshooting
- What to expect
- Common issues

## TL;DR

```
1. Copy VANILLA_GRPO_1.5B_STANDALONE.txt
2. Paste in Colab
3. Connect GPU
4. Run!
5. Wait 4-6 hours
6. Share checkpoint path
```

Thanks! 🎉

---

**Your training:** CB-GRPO 0.5B (novel method)  
**Friend's training:** Vanilla GRPO 1.5B (baseline)  
**Timeline:** Both run in parallel (4-6 hours each)  
**After:** Compare results and optionally train CB-GRPO 1.5B!
