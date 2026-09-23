# ═══════════════════════════════════════════════════════════════════
# §7b LAUNCH — edit MODEL_SIZE / GATE_TYPE / SEED between runs
# Re-run ONLY this cell for each new seed/gate (no need to re-run §0-§5)
#
# BEFORE your first launch after applying the score_sequences memory fix:
# re-run the §7b "write script" cell so train_script.py picks it up --
# editing the notebook cell alone does not touch the file on disk.
# ═══════════════════════════════════════════════════════════════════

import os
import subprocess

# ---- editable per-run settings ----
MODEL_SIZE   = "0.5B"      # "0.5B" or "1.5B"
GATE_TYPE    = "vanilla"   # "vanilla" | "cb_grpo" | "o_self" | "h_cb_grpo"
                           # NOTE: o_self / h_cb_grpo require MODEL_SIZE = "1.5B" (Req 33.4)
COMPUTE_TIER = "standard"  # "smoke" | "standard" | "final"
SEED         = 0

# ---- reduces CUDA memory fragmentation (the OOM's own error message
#      suggests this); set before the subprocess launches, not after ----
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cmd = [
    "accelerate", "launch",
    "--config_file", "/kaggle/working/accelerate_config.yaml",
    "/kaggle/working/train_script.py",
    "--model_size",   MODEL_SIZE,
    "--gate_type",    GATE_TYPE,
    "--compute_tier", COMPUTE_TIER,
    "--seed",         str(SEED),
]

print(f"Launching: {' '.join(cmd)}")
result = subprocess.run(cmd, check=False, env=env)

if result.returncode != 0:
    print(f"Training exited with code {result.returncode}")
    print("If this was an OOM again: confirm the score_sequences fix actually made it into "
          "train_script.py (grep for 'cross_entropy' in the file -- if it's not there, the "
          "write-script cell wasn't re-run before this launch), then next lower "
          "GenerationConfig.max_new_tokens (512 -> 384) or "
          "OptimConfig.per_device_prompts_per_step (4 -> 2) in the config cell as a further margin.")
else:
    print("Training complete.")

# ── Future runs: change the variables above and re-run only this cell ──
# seed 1:   SEED = 1
# seed 2:   SEED = 2
# cb_grpo:  GATE_TYPE = "cb_grpo"
# 1.5B:     MODEL_SIZE = "1.5B"   (also unlocks o_self, h_cb_grpo)
