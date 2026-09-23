# CB-GRPO Evaluation Guide

## 📁 File Structure Needed

Upload your trained models to Google Drive in this structure:

```
/content/drive/MyDrive/
├── exp2_n1_vanilla_0.5B/          # Note: nested folder from archive
│   └── final_model/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── ...
├── cbgrpo_final_model/
│   └── final_model/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
└── results/
    └── (eval results will be saved here)
```

**Note:** Base model (Qwen2.5-0.5B-Instruct) is loaded directly from HuggingFace - no upload needed!

## 🚀 Steps to Run Evaluation

### Step 1: Upload Models to Google Drive

1. **Upload vanilla model (already extracted locally):**
   - Source: `c:\Users\USER\sidaarth\reasoning boundry paradox of SLM\exp2_n1_vanilla_0.5B-20260911T061946Z-1-001\exp2_n1_vanilla_0.5B\`
   - Destination: `/content/drive/MyDrive/exp2_n1_vanilla_0.5B/`
   - Upload the ENTIRE folder (includes `final_model`, `checkpoint-800`, etc.)

2. **Verify CB-GRPO model is uploaded:**
   - Should already be at: `/content/drive/MyDrive/cbgrpo_final_model/final_model/`

### Step 2: Run Eval in Colab

1. Open new Colab notebook
2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Copy entire contents of `CELL_EVAL_PASS_AT_K.txt`
4. Paste into Colab cell
5. Run it!

## ⏱️ Expected Time

- **3 models × 30 problems × 16 samples × 2 datasets = 2,880 generation calls**
- **Total time: ~2-3 hours on T4 GPU**

## 📊 What You'll Get

### Output in Colab:
```
================================================================================
📊 FINAL COMPARISON: Base → Vanilla GRPO → CB-GRPO
================================================================================

GSM8K (in-distribution):
  Pass@k    base_0.5B      vanilla_0.5B   cbgrpo_0.5B    
  ----------------------------------------------------------------------
  1              0.001          0.002          0.002
  4              0.004          0.008          0.008
  8              0.008          0.017          0.017
  16             0.015          0.033          0.033

  Improvements over Base:
  Pass@k    Vanilla Δ      CB-GRPO Δ      
  ----------------------------------------------------------------------
  1             +0.001         +0.001
  4             +0.004         +0.004
  8             +0.009         +0.009
  16            +0.018         +0.018

MATH-500 (out-of-distribution):
  Pass@k    base_0.5B      vanilla_0.5B   cbgrpo_0.5B    
  ----------------------------------------------------------------------
  1              0.002          0.004          0.004
  4              0.008          0.017          0.017
  8              0.015          0.033          0.033
  16             0.030          0.067          0.067

  Improvements over Base:
  Pass@k    Vanilla Δ      CB-GRPO Δ      
  ----------------------------------------------------------------------
  1             +0.002         +0.002
  4             +0.009         +0.009
  8             +0.018         +0.018
  16            +0.037         +0.037
```

### Saved to Drive:
- `/content/drive/MyDrive/results/eval_base_0.5B.json`
- `/content/drive/MyDrive/results/eval_vanilla_0.5B.json`
- `/content/drive/MyDrive/results/eval_cbgrpo_0.5B.json`

## 🔄 Comparing with Vanilla 1.5B (Later)

Once your friend runs the vanilla 1.5B training, add it to the comparison:

```python
CHECKPOINTS = {
    "base_0.5B": None,
    "vanilla_0.5B": "/content/drive/MyDrive/vanilla_0.5B_final_model/final_model",
    "cbgrpo_0.5B": "/content/drive/MyDrive/cbgrpo_final_model/final_model",
    "vanilla_1.5B": "/content/drive/MyDrive/vanilla_1.5B_final_model/final_model",  # ADD THIS
}
```

## 📈 What to Look For

**Your hypothesis:** CB-GRPO should show:
- ✅ **Similar Pass@1** to vanilla (accuracy doesn't degrade)
- ✅ **Better Pass@16/Pass@1 ratio** (better exploration/diversity)
- ✅ **Smaller capacity imbalance** (balanced learning across problem types)

## ⚠️ Troubleshooting

**"CUDA out of memory"**
- Reduce `N_SAMPLES` from 16 to 8
- Reduce `N_EVAL_PROBLEMS` from 30 to 20

**"Model not found"**
- Check path in `CHECKPOINTS` dict matches your Drive structure
- Verify folder name is exactly `final_model` (case-sensitive)

**"No GPU visible"**
- Runtime → Change runtime type → T4 GPU
- Wait for GPU to allocate
