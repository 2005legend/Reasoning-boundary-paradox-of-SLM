"""
🚀 REAL TRAINING EXECUTION CELL
Copy this entire cell into your Colab notebook as Cell 20 or 21

This will ACTUALLY train models. Start with SMOKE TEST!
"""

# ==============================================================================
# 🚀 TRAINING EXECUTION - UNCOMMENT THE OPTION YOU WANT
# ==============================================================================

import torch
import gc
from datetime import datetime

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print("=" * 80)
print("🚀 READY FOR REAL TRAINING")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ==============================================================================
# OPTION 1: SMOKE TEST (RECOMMENDED FIRST) - 30-60 minutes
# ==============================================================================
print("\n" + "=" * 80)
print("🧪 SMOKE TEST: Quick Pipeline Validation (30-60 min)")
print("=" * 80)
print("Uncomment the lines below to run...")
print()

# Load 0.5B model
model, tokenizer = load_quantized_model("0.5B")

# Create smoke config
smoke_config = ExperimentConfig.from_compute_tier(
    exp_name="smoke_test_vanilla_0.5B",
    model_size="0.5B",
    gate_type="VanillaGate",
    compute_tier="smoke",  # 100 steps only
    batch_size=8
)

print(f"\n⏳ Training for {smoke_config.training_steps} steps...")
print(f"Dataset: {len(gsm8k_train['problems'][:500])} problems (subset)")

# Prepare dataset for TRL
from datasets import Dataset
train_subset = [
    {"prompt": p, "answer": a}
    for p, a in zip(
        gsm8k_train['problems'][:500],
        gsm8k_train['answers'][:500]
    )
]
train_dataset = Dataset.from_list(train_subset)

# Setup GRPO Trainer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

training_args = GRPOConfig(
    output_dir=f"/content/drive/MyDrive/RLVR_Research/checkpoints/{smoke_config.exp_name}",
    num_train_epochs=1,
    per_device_train_batch_size=smoke_config.batch_size,
    gradient_accumulation_steps=smoke_config.gradient_accumulation_steps,
    max_steps=smoke_config.training_steps,
    learning_rate=smoke_config.learning_rate,
    save_steps=50,
    logging_steps=10,
    beta=0.1,  # KL penalty
    remove_unused_columns=False,
)

peft_config = LoraConfig(
    r=smoke_config.lora_r,
    lora_alpha=smoke_config.lora_alpha,
    target_modules=smoke_config.lora_target_modules,
    task_type="CAUSAL_LM",
    bias="none"
)

print("\n⏳ Initializing GRPO Trainer...")

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

print("✅ Trainer initialized!")
print("\n" + "=" * 80)
print("🔥 STARTING TRAINING")
print("=" * 80)

# TRAIN!
trainer.train()

print("\n" + "=" * 80)
print("✅ SMOKE TEST COMPLETE!")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Next steps:")
print("  1. Check logs for any errors")
print("  2. Verify checkpoints in Google Drive")
print("  3. If successful, try OPTION 2 (standard tier)")


# ==============================================================================
# OPTION 2: STANDARD TIER - Full 0.5B Training (2-4 hours)
# ==============================================================================
# Uncomment below for full 800-step training:

# print("\n" + "=" * 80)
# print("🎯 STANDARD TRAINING: Vanilla GRPO 0.5B")
# print("=" * 80)
# 
# model, tokenizer = load_quantized_model("0.5B")
# 
# config = get_config("exp2_n1_vanilla_0.5B")
# print(f"\nTraining steps: {config.training_steps}")
# print(f"Expected time: 2-4 hours")
# 
# # Prepare full dataset
# train_full = [
#     {"prompt": p, "answer": a}
#     for p, a in zip(gsm8k_train['problems'], gsm8k_train['answers'])
# ]
# train_dataset = Dataset.from_list(train_full)
# 
# # Setup trainer (same as above but with config.training_steps)
# training_args = GRPOConfig(
#     output_dir=f"/content/drive/MyDrive/RLVR_Research/checkpoints/{config.exp_name}",
#     max_steps=config.training_steps,
#     per_device_train_batch_size=config.batch_size,
#     # ... (copy from above)
# )
# 
# trainer = GRPOTrainer(model=model, args=training_args, train_dataset=train_dataset, peft_config=peft_config, tokenizer=tokenizer)
# trainer.train()
# 
# print("\n✅ STANDARD TRAINING COMPLETE!")


# ==============================================================================
# OPTION 3: CB-GRPO - Your Novel Algorithm! (2-4 hours)
# ==============================================================================
# Uncomment for CB-GRPO experiment:

# print("\n" + "=" * 80)
# print("🌟 CB-GRPO TRAINING (Your Novel Algorithm!)")
# print("=" * 80)
# 
# # Ensure clustering is done
# if 'cluster_assignments' not in globals():
#     print("\n⏳ Running clustering...")
#     cluster_assignments = cluster_training_prompts(
#         prompts=gsm8k_train['problems'],
#         n_clusters=16
#     )
# 
# model, tokenizer = load_quantized_model("0.5B")
# config = get_config("exp2_n1_cbgrpo_0.5B")
# 
# # Create CB-GRPO gate
# gate = create_gate("CBGRPOGate", cluster_assignments=cluster_assignments, batch_size=config.batch_size)
# 
# # Use gate to sample batches (integrate with TRL's data collator)
# # ... CB-GRPO specific training logic ...
# 
# print("\n✅ CB-GRPO TRAINING COMPLETE!")

