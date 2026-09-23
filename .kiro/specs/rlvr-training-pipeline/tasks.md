# Implementation Plan: RLVR Training Pipeline

## Overview

This implementation plan creates a Jupyter notebook-based research pipeline for investigating the "Reasoning Boundary Paradox" in Small Language Models (SLMs) through Reinforcement Learning with Verifiable Rewards (RLVR). The system operates within Google Colab free-tier constraints (T4 GPU, 16GB VRAM) and produces results suitable for IEEE conference paper submission.

**Implementation Language:** Python 3.10+ (Jupyter Notebook)

**Target Environment:** Google Colab (free tier, T4 GPU)

**Single Notebook:** `rlvr_training_pipeline.ipynb`

## Task Prioritization

- **Core Mandatory (MVP):** Tasks 1-7, 9-13 (N1 analysis at both scales)
- **N2 Baselines:** Task 14 (required for N2 novelty)
- **Optional Enhancement:** Tasks 8, 15, 16, 18, 19 (time permitting)

## Tasks

### Task 1: Environment Setup and Configuration System

**Objective:** Initialize the Colab runtime environment, mount Google Drive for persistence, verify GPU availability, and create a flexible configuration system for experiments.

**Dependencies:** None (first task)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 1-3)

**Sub-tasks:**

- [x] 1.1 Create Google Drive mounting cell
  - Mount Google Drive to `/content/drive`
  - Create directory structure: `checkpoints/`, `results/`, `figures/`, `logs/`
  - Verify write permissions
  - _Requirements: 1.1, 19.1, 19.2, 19.3, 19.4_

- [x] 1.2 Install required dependencies
  - Install unsloth for QLoRA optimization
  - Install transformers, trl, sentence-transformers
  - Install sympy for answer normalization
  - Install plotly and matplotlib for visualization
  - Display package versions for reproducibility
  - _Requirements: 1.2, 29.4_

- [x] 1.3 GPU verification and VRAM check
  - Verify CUDA availability
  - Display GPU model name (expect T4)
  - Check available VRAM (minimum 14GB free)
  - Display warning if VRAM < 14GB
  - _Requirements: 1.3, 1.5, 28.1_

- [x] 1.4 Set random seeds for reproducibility
  - Set torch.manual_seed, np.random.seed, random.seed
  - Save seed value to experiment metadata
  - _Requirements: 1.6, 29.1, 29.2_

- [x] 1.5 Implement ExperimentConfig dataclass
  - Define complete configuration schema (exp_name, model_size, compute_tier, gate_type, etc.)
  - Implement validation logic (model size, compute tier, gate type compatibility)
  - Implement factory method `from_compute_tier()` for tier-based config creation
  - Create config dictionaries for Exp0, Exp1, Exp2, Exp3
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 33.1-33.8_

- [x] 1.6 Create configuration validation function
  - Validate model size is 0.5B, 1.5B, or 3B
  - Validate compute tier is smoke/standard/final
  - Validate gate type compatibility with model size (0.5B cannot use O-SELF, Static SELF, Adaptive Rollout)
  - Display clear error messages with correction guidance
  - _Requirements: 2.6, 2.7, 33.1-33.8_

**Checkpoint:** Environment initialized, GPU verified, configuration system ready

---

### Task 2: Dataset Pipeline (GSM8K Loading, XML Formatting, Clustering)

**Objective:** Load GSM8K dataset, reformat prompts to XML instruction format, cluster training prompts by semantic similarity for CB-GRPO, and create evaluation dataset loaders.

**Dependencies:** Task 1 (requires configuration system)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 4-7)

**Sub-tasks:**

- [x] 2.1 Implement GSM8K dataset loader
  - Load openai/gsm8k train split (7473 problems)
  - Load openai/gsm8k test split (1319 problems)
  - Parse ground truth answers from dataset format
  - Display sample count and example problem
  - _Requirements: 4.1, 4.2, 4.5, 4.8_

- [-] 2.2 Implement XML prompt formatter
  - Format prompts with Qwen2.5 instruction template
  - Include XML structure expectation in prompt: `<reasoning>...</reasoning>` and `<answer>\boxed{...}</answer>`
  - Create example prompt showing expected format
  - _Requirements: 4.6, 34.1_

- [-] 2.3 Implement prompt clustering for CB-GRPO
  - Embed all training prompts using sentence-transformers (all-MiniLM-L6-v2)
  - Cluster embeddings using KMeans with n_clusters=16
  - Assign cluster ID to each training prompt
  - Save cluster assignments to Drive (cluster_assignments.pkl)
  - Load existing assignments if available (avoid recomputation)
  - Display cluster size distribution
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [-] 2.4 Load additional evaluation datasets
  - Load madrylab/gsm8k-platinum (1319 problems)
  - Load HuggingFaceH4/MATH-500 (500 problems)
  - Format all evaluation datasets with XML template
  - _Requirements: 4.3, 4.4, 4.6_

- [x] 2.5 Implement dataset sample inspection utilities
  - Function to display N random training prompts
  - Function to display N random test problems with ground truth
  - Function to display specific problem by ID
  - Function to display cluster members for specific cluster ID
  - _Requirements: 34.1, 34.2, 34.4, 34.6_

- [x] 2.6 Implement error handling for dataset loading
  - Retry logic (up to 3 attempts) for network failures
  - Display descriptive error messages with dataset name
  - Suggest checking internet connection on repeated failures
  - _Requirements: 4.7, 27.3_

**Checkpoint:** Datasets loaded, prompts formatted in XML, clustering complete, sample inspection verified

---

### Task 3: Multi-Size Model Initialization (QLoRA with 4-bit Quantization)

**Objective:** Implement model loading with 4-bit NF4 quantization and QLoRA for both 0.5B and 1.5B Qwen2.5 models, ensuring VRAM usage stays within T4 constraints.

**Dependencies:** Task 1 (requires configuration system)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 8-10)

**Sub-tasks:**

- [-] 3.1 Implement quantized model loader function
  - Load Qwen2.5-0.5B-Instruct with 4-bit NF4 quantization
  - Load Qwen2.5-1.5B-Instruct with 4-bit NF4 quantization
  - Configure double quantization for additional memory savings
  - Apply gradient checkpointing to reduce activation memory
  - _Requirements: 3.1, 3.2, 3.6_

- [x] 3.2 Implement QLoRA adapter application
  - Apply LoRA with rank=16, alpha=32
  - Target modules: q_proj, v_proj, k_proj, o_proj
  - Display trainable parameter count after LoRA application
  - _Requirements: 3.3_

- [x] 3.3 Implement VRAM monitoring and safety checks
  - Display VRAM consumption after model loading
  - Raise error if VRAM exceeds 14GB before training
  - Verify minimum 2GB free VRAM headroom
  - _Requirements: 3.4, 3.5, 28.1, 28.4_

- [x] 3.4 Verify tokenizer compatibility with XML format
  - Test tokenizer on sample XML-formatted prompt
  - Verify special tokens don't conflict with XML tags
  - Display tokenized example for verification
  - _Requirements: 3.7_

- [x] 3.5 Implement memory management utilities
  - Function to clear GPU cache between operations
  - Function to display current VRAM usage
  - Function to unload model from GPU if needed
  - _Requirements: 28.3, 28.6_

- [x] 3.6 Implement error handling for model loading failures
  - Detect VRAM allocation failures
  - Display current VRAM usage on failure
  - Suggest reducing model size or batch size
  - _Requirements: 27.1_

**Checkpoint:** Both 0.5B and 1.5B models load successfully within VRAM constraints, QLoRA applied, trainable parameters verified

---

### Task 4: Reward System (Format Validation + Correctness Checking with Parser/Printer)

**Objective:** Implement dual reward system with format validation (XML structure) and correctness checking (answer equivalence), including parser and pretty-printer with correctness properties.

**Dependencies:** Task 1 (requires configuration)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 11-14)

**Sub-tasks:**

- [x] 4.1 Implement XML parser for model outputs
  - Extract text within `<reasoning>...</reasoning>` tags
  - Extract text within `<answer>...</answer>` tags
  - Extract content within `\boxed{...}` in answer section
  - Return None for malformed or missing tags
  - Handle nested tags (extract outermost)
  - Handle multi-line content within tags
  - Strip leading/trailing whitespace from extracted content
  - _Requirements: 36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7_

- [x] 4.2 Write property test for parser extraction (Properties 1-3)
  - **Property 1: Parser extracts reasoning content**
  - **Property 2: Parser extracts answer content**
  - **Property 3: Parser extracts boxed content**
  - **Validates: Requirements 36.1, 36.2, 36.3, 36.7**
  - Use Hypothesis to generate random strings with varying tag structures
  - Test with special characters, whitespace patterns, nested tags
  - Minimum 100 iterations per property

- [x] 4.3 Implement XML pretty-printer for output generation
  - Format reasoning text within `<reasoning>` tags
  - Format answer text within `<answer>` tags
  - Format numeric answer within `\boxed{...}` in answer section
  - Insert newlines between tags for readability
  - Escape special characters within tag content
  - _Requirements: 37.1, 37.2, 37.3, 37.4, 37.5_

- [x] 4.4 Write property test for round-trip preservation (Property 5)
  - **Property 5: Round-trip preservation - parse(print(parse(text))) ≡ parse(text)**
  - **Validates: Requirements 37.6, 38.1, 38.2**
  - Use Hypothesis to generate random valid output structures (reasoning, answer, boxed)
  - Verify parsing → printing → parsing produces equivalent components
  - Test with special characters, multi-line content, edge cases
  - Minimum 100 iterations
  - This is the PRIMARY correctness property for parser/printer

- [x] 4.5 Implement format reward computation
  - Check for `<reasoning>` opening and closing tags
  - Check for `<answer>` opening and closing tags
  - Check for `\boxed{...}` within answer tags
  - Assign reward 1.0 if all format requirements met, 0.0 otherwise
  - Execute format checking on CPU (avoid GPU memory overhead)
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

- [x] 4.6 Implement correctness reward computation
  - Extract numeric answer from `\boxed{...}` content
  - Normalize extracted answer using sympy parsing
  - Normalize ground truth answer using sympy parsing
  - Compare normalized answers using sympy equivalence
  - Support positive-only mode: correct=1.0, incorrect=0.0
  - Support negative-only mode: correct=0.0, incorrect=-1.0
  - Handle extraction/parsing failures gracefully (assign 0.0)
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10_

- [x] 4.7 Implement reward aggregator
  - Combine format reward and correctness reward (sum)
  - Log mean reward per batch
  - Log format success rate per batch
  - Log correctness success rate per batch
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [x] 4.8 Implement error handling for reward computation
  - Catch parsing errors and assign zero reward
  - Log problem ID when reward computation fails
  - Continue processing remaining samples
  - _Requirements: 27.5_

- [x] 4.9 Implement smoke tier validation for parser/printer
  - Run all property tests during smoke tier initialization
  - Log any property failures with counterexample
  - Display success message if all properties pass
  - _Requirements: 38.3_

**Checkpoint:** Format and correctness rewards working correctly, parser/printer round-trip property verified, property tests passing

---

### Task 5: Gate Architecture (Abstract Base + 5 Implementations)

**Objective:** Implement abstract Gate interface and five concrete implementations: VanillaGate (no filtering), CBGRPOGate (cluster-balanced), OSELFGate (online solve-rate), StaticSELFGate (precomputed solve-rate), AdaptiveRolloutGate (variance-based).

**Dependencies:** Task 2 (requires cluster assignments), Task 4 (requires reward computation)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 15-20)

**Sub-tasks:**

- [x] 5.1 Implement abstract Gate base class
  - Define interface: `weight(batch, trainer_state) -> torch.Tensor`
  - Define interface: `get_state_dict() -> dict` for checkpointing
  - Define interface: `load_state_dict(state_dict)` for resume
  - Define interface: `get_statistics() -> dict` for logging
  - _Requirements: Design Section 5_

- [x] 5.2 Implement VanillaGate
  - Return uniform weights (all 1.0)
  - No state tracking needed
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 5.3 Implement CBGRPOGate (cluster-balanced gradient gating)
  - Initialize gradient-mass accumulator per cluster (16 clusters)
  - Track cumulative gradient norm per cluster after each step
  - Compute spend ratio relative to mean cluster spend
  - Apply soft gating: weight = decay^(ratio - theta) when ratio > theta
  - Use theta=1.5, decay=0.98, EMA alpha=0.05
  - Log per-cluster gradient-mass every 50 steps
  - Return gate statistics (spend distribution, mean/std spend)
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 5.4 Implement OSELFGate (online solve-rate filtering)
  - Compute online EMA solve-rate per problem
  - Update EMA after each rollout evaluation
  - Reduce sampling probability when solve-rate exceeds threshold (0.7)
  - Apply exponential decay: weight = 0.95^(solve_rate - threshold) when above threshold
  - Log gate statistics (filtered count, mean solve-rate) every 50 steps
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 5.5 Implement StaticSELFGate (precomputed solve-rate filtering)
  - Precompute solve-rates using base model (Pass@4 proxy)
  - Save precomputed solve-rates to disk (static_solve_rates.pkl)
  - Load precomputed solve-rates if available
  - Exclude problems with solve-rate > threshold (0.7)
  - Display filtered problem count before training
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 5.6 Implement AdaptiveRolloutGate (variance-based filtering)
  - Compute reward variance per problem using online estimator
  - Update variance estimate after each rollout evaluation
  - Reduce sampling probability when variance < threshold (0.1)
  - Apply exponential decay: weight = 0.95^(threshold - variance) when below threshold
  - Log gate statistics (mean variance, filtered count) every 50 steps
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 5.7 Implement gate factory function
  - Accept gate_type string and gate_params dict
  - Return instantiated gate object
  - Validate gate type and parameters
  - _Requirements: 2.4_

**Checkpoint:** All 5 gates implemented, abstract interface consistent, gate statistics logging working

---

### Task 6: Pass@k Evaluation Suite (Unbiased Estimator + Shrinkage Slope + Transition Matrix)

**Objective:** Implement Pass@k evaluation using unbiased estimator, shrinkage slope computation via log-linear regression, and transition matrix computation for capability tracking.

**Dependencies:** Task 3 (requires model), Task 4 (requires reward system)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 21-25)

**Sub-tasks:**

- [x] 6.1 Implement rollout generator for evaluation
  - Generate n solutions per problem sequentially (not batched)
  - Use temperature=0.7, top_p=0.95, max_new_tokens=512
  - Implement retry logic (up to 3 attempts) for generation failures
  - _Requirements: 31.1, 31.2, 31.3, 31.4, 31.5, 31.7_

- [x] 6.2 Implement Pass@k unbiased estimator
  - Generate n solutions per test problem (n=10 default, configurable)
  - Evaluate correctness for all generated solutions
  - Compute Pass@k for k in [1, 2, 3, 5, 10] using formula: Pass@k = 1 - C(n-c, k) / C(n, k)
  - Handle edge case: n - c < k implies Pass@k = 1.0
  - Return PassAtKResult dataclass with results
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 15.10_

- [x] 6.3 Implement Pass@k evaluation on multiple datasets
  - Evaluate on openai/gsm8k test split
  - Evaluate on madrylab/gsm8k-platinum
  - Evaluate on HuggingFaceH4/MATH-500
  - Save results to JSON files per dataset
  - Display results in formatted tables
  - _Requirements: 15.5, 15.6, 15.7, 15.8, 15.9_

- [x] 6.4 Implement shrinkage slope computation
  - Compute Δ Pass@k = Pass@k_RL - Pass@k_base for each k
  - Fit linear regression of log(k) vs. Δ Pass@k
  - Extract slope coefficient as shrinkage slope metric
  - Compute R² for regression fit quality
  - Return slope, intercept, r_squared
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_

- [x] 6.5 Implement bootstrap confidence interval for shrinkage slope
  - Use bootstrap resampling (1000 iterations)
  - Compute 95% confidence interval for slope
  - Return (lower_bound, upper_bound)
  - _Requirements: 16.7_

- [x] 6.6 Implement transition matrix computation
  - Classify each problem: kept_correct, lost_capability, gained_capability, kept_incorrect
  - Compute counts and percentages for each category
  - Store problem IDs for each category
  - Display formatted transition matrix table
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7_

- [x] 6.7 Implement error handling for evaluation failures
  - Save partial results if evaluation fails mid-execution
  - Skip individual problems that fail and continue
  - Display warning when Pass@k values are invalid
  - _Requirements: 27.4, 16.7_

**Checkpoint:** Pass@k evaluation working on all datasets, shrinkage slope computation verified, transition matrix computation working

---

### Task 7: Experiment 0 - Base Model Evaluation (0.5B + 1.5B Baselines)

**Objective:** Evaluate base Qwen2.5 models (0.5B and 1.5B) without training to establish baseline Pass@k metrics for N1 analysis.

**Dependencies:** Task 2 (datasets), Task 3 (models), Task 6 (evaluation)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 26-28)

**Sub-tasks:**

- [x] 7.1 Implement Exp0 configuration
  - Create Exp0 config for 0.5B model
  - Create Exp0 config for 1.5B model
  - Set eval_num_samples=32 for higher statistical power
  - Set eval_k_values=[1, 4, 16, 32]
  - _Requirements: 2.1, 2.3_

- [x] 7.2 Run base model evaluation for 0.5B
  - Load Qwen2.5-0.5B-Instruct (no training)
  - Evaluate Pass@k on GSM8K test, GSM8K-platinum, MATH-500
  - Generate n=32 solutions per problem
  - Save results to `results/exp0/0.5B/pass_at_k_base.json`
  - Display results in formatted table
  - _Requirements: 20.1, 20.3, 20.5, 20.6, 20.7_

- [x] 7.3 Run base model evaluation for 1.5B
  - Load Qwen2.5-1.5B-Instruct (no training)
  - Evaluate Pass@k on GSM8K test, GSM8K-platinum, MATH-500
  - Generate n=32 solutions per problem
  - Save results to `results/exp0/1.5B/pass_at_k_base.json`
  - Display results in formatted table
  - _Requirements: 20.2, 20.4, 20.5, 20.6, 20.7_

- [x] 7.4 Implement pretty-printing for base model results
  - Format Pass@k results as aligned tables
  - Display results for all evaluation datasets
  - Save summary report to markdown
  - _Requirements: 32.1, 32.2, 32.7_

**Checkpoint:** Base model Pass@k metrics established for both 0.5B and 1.5B, baseline results saved for comparison

---

### Task 8: Experiment 1 - SFT Warmup Training (Optional, 1.5B Only)

**Objective:** Perform supervised fine-tuning warmup on 1.5B model to test whether SFT initialization improves RLVR training outcomes.

**Dependencies:** Task 2 (datasets), Task 3 (model), Task 4 (format checking)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 29-30)

**Sub-tasks:**

- [x] 8.1 Implement SFT training configuration
  - Create Exp1 config for 1.5B model
  - Set training_steps based on compute tier (500 for standard, 1000 for final)
  - Use standard supervised learning loss (cross-entropy)
  - _Requirements: 2.1, 21.3_

- [x] 8.2 Prepare SFT training data
  - Format ground truth solutions as training targets
  - Use XML format for consistency: `<reasoning>...<answer>\boxed{...}</answer>`
  - Create supervised dataset with prompt-solution pairs
  - _Requirements: 21.2_

- [x] 8.3 Implement SFT training loop
  - Train 1.5B model using supervised learning
  - Log loss every 10 steps
  - Save SFT checkpoint separately (`sft_1.5B_step_N.pt`)
  - _Requirements: 21.1, 21.3, 21.4_

- [x] 8.4 Evaluate SFT model before RLVR
  - Run Pass@k evaluation on SFT checkpoint
  - Save results to `results/exp1/1.5B/pass_at_k_sft.json`
  - Compare format compliance rate vs. base model
  - _Requirements: 21.5_

- [x] 8.5 Optionally use SFT checkpoint as RLVR initialization
  - Provide option to load SFT checkpoint before RLVR training
  - Test whether SFT warmup improves RLVR convergence
  - _Requirements: 21.6_

**Checkpoint:** SFT warmup training complete, SFT checkpoint saved, optional initialization for RLVR experiments ready

---

### Task 9: Checkpoint System and Resume Robustness Test

**Objective:** Implement comprehensive checkpoint saving/loading with integrity verification, resume protocol for Colab disconnects, and validate resume robustness.

**Dependencies:** Task 3 (models), Task 5 (gates)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 31-34)

**Sub-tasks:**

- [x] 9.1 Implement TrainingState dataclass
  - Define complete training state schema (step, epoch, model_state, optimizer_state, etc.)
  - Include gate state, metrics history, RNG states
  - Include metadata (timestamp, hostname, GPU name)
  - _Requirements: Design Section 7_

- [x] 9.2 Implement CheckpointManager class
  - Implement save_checkpoint() with three types: last, best, step
  - Compute SHA256 hash for integrity verification
  - Save hash to `.sha256` file alongside checkpoint
  - Implement load_checkpoint() with integrity verification
  - Implement find_latest_checkpoint() to detect resume candidates
  - Implement cleanup_old_checkpoints() to maintain only N most recent
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.9, 14.10_

- [x] 9.3 Implement ResumeManager class
  - Implement prompt_resume() to display checkpoint info and ask user
  - Implement restore_training_state() to load model, optimizer, LR scheduler, gate, and RNG states
  - Verify checkpoint integrity before loading (SHA256 check)
  - Fallback to previous checkpoint if corruption detected
  - _Requirements: 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7_

- [x] 9.4 Implement checkpoint interval logic
  - Save checkpoint every 150 steps for runs >4 hours
  - Save checkpoint every 200 steps for runs <4 hours
  - Always maintain last.pt and best.pt
  - Update best.pt when Pass@1 improves
  - _Requirements: 14.2, 14.6_

- [x] 9.5 Implement robust error handling for checkpoint operations
  - Display descriptive errors for checkpoint load failures
  - Suggest starting from scratch if no valid checkpoint exists
  - Verify checkpoint path exists before loading
  - _Requirements: 27.2, 27.6_

- [x] 9.6 Test resume robustness (smoke tier)
  - Start training with VanillaGate at 0.5B (smoke tier, 100 steps)
  - Manually interrupt at step 25 (simulate disconnect)
  - Restart notebook and verify checkpoint detection
  - Confirm resume and verify training continues from step 25
  - Verify metrics history is preserved
  - Verify RNG states produce deterministic results
  - Complete remaining 75 steps
  - _Requirements: 30.1-30.7_

**Checkpoint:** Checkpoint system fully implemented, integrity verification working, resume after disconnect verified in smoke tier

---

### Task 10: Experiment 2 N1 - Vanilla GRPO at 0.5B (Standard Tier)

**Objective:** Train 0.5B model using standard GRPO (VanillaGate) for 800 steps to measure baseline boundary-shrinkage severity at small scale.

**Dependencies:** Task 2 (datasets), Task 3 (model), Task 4 (rewards), Task 5 (VanillaGate), Task 6 (evaluation), Task 9 (checkpointing)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 35-37)

**Sub-tasks:**

- [x] 10.1 Implement GRPO training loop integration
  - Integrate TRL's GRPOTrainer with custom gate system
  - Implement advantage computation with gate weighting hook
  - Configure GRPO parameters (group_size=4, temperature=0.7, top_p=0.95)
  - Use paged_adamw_8bit optimizer for memory efficiency
  - _Requirements: Design Section 4.1_

- [x] 10.2 Implement training metrics tracking
  - Track loss, mean_reward, format_success_rate, correctness_success_rate
  - Log metrics every 10 steps
  - Store metrics history for visualization
  - _Requirements: 6.5, 6.6_

- [x] 10.3 Configure Exp2 N1 Vanilla 0.5B
  - Create config: exp_name="exp2_n1_vanilla_0.5B", model_size="0.5B", gate_type="VanillaGate", compute_tier="standard"
  - Set training_steps=800 (standard tier)
  - Set batch_size=4, gradient_accumulation_steps=1
  - Set learning_rate=5e-5, warmup_ratio=0.1
  - Set reward_mode="positive-only"
  - _Requirements: 2.1, 2.2, 2.3, 22.1_

- [x] 10.4 Run training for 0.5B Vanilla GRPO
  - Generate G=4 rollouts per training prompt per step
  - Compute format and correctness rewards
  - Apply GRPO gradient updates (no gating)
  - Log training metrics every 10 steps
  - Save checkpoint every 200 steps
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 10.5 Run Pass@k evaluation every 150 steps
  - Evaluate on GSM8K test split
  - Generate n=10 solutions, compute Pass@k for k in [1,2,3,5,10]
  - Save intermediate results to JSON
  - Update best.pt checkpoint if Pass@1 improves
  - _Requirements: 6.6, 6.7_

- [x] 10.6 Run final Pass@k evaluation (n=32, k=[1,4,16,32])
  - Use higher n=32 for publication-quality results
  - Evaluate on all three datasets (GSM8K test, platinum, MATH-500)
  - Compute shrinkage slope by comparing with Exp0 baseline
  - Compute transition matrix
  - Save final results to `results/exp2_n1_vanilla_0.5B/`
  - _Requirements: 22.1, 22.6, 22.7_

- [x] 10.7 Implement training visualization
  - Plot loss curve, mean reward curve
  - Plot format success rate, correctness success rate
  - Update plots every 50 steps inline in notebook
  - _Requirements: 18.1, 18.2, 18.3, 18.4_

**Checkpoint:** 0.5B Vanilla GRPO training complete (800 steps), Pass@k evaluated, shrinkage slope computed, baseline for 0.5B established

---

### Task 11: Experiment 2 N1 - Vanilla GRPO at 1.5B (Standard→Final Upgrade)

**Objective:** Train 1.5B model using standard GRPO (VanillaGate) for 800 steps initially, with option to upgrade to final tier (1800 steps) after N2 baselines complete.

**Dependencies:** Task 2 (datasets), Task 3 (model), Task 4 (rewards), Task 5 (VanillaGate), Task 6 (evaluation), Task 9 (checkpointing)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 38-40)

**Sub-tasks:**

- [x] 11.1 Configure Exp2 N1 Vanilla 1.5B (standard tier)
  - Create config: exp_name="exp2_n1_vanilla_1.5B", model_size="1.5B", gate_type="VanillaGate", compute_tier="standard"
  - Set training_steps=800 (standard tier initially)
  - Set batch_size=4, gradient_accumulation_steps=1
  - Monitor VRAM usage (expect ~10-12GB)
  - _Requirements: 2.1, 2.2, 2.3, 22.2_

- [x] 11.2 Run training for 1.5B Vanilla GRPO (standard tier)
  - Generate G=4 rollouts per training prompt per step
  - Apply GRPO gradient updates (no gating)
  - Log training metrics every 10 steps
  - Save checkpoint every 200 steps
  - Run Pass@k evaluation every 150 steps
  - _Requirements: 6.1-6.7, 22.2_

- [x] 11.3 Run intermediate Pass@k evaluation at step 800
  - Evaluate on GSM8K test (n=10)
  - Compute preliminary shrinkage slope
  - Compare with 0.5B results to verify scale effect
  - _Requirements: 22.6, 22.7_

- [x] 11.4 Implement resume capability for final tier upgrade
  - After N2 baselines complete, offer to upgrade to final tier (1800 steps)
  - Resume from step 800 checkpoint
  - Continue training for additional 1000 steps
  - Maintain checkpoint history across resume cycles
  - _Requirements: 30.3, 30.4_

- [x] 11.5 Run final Pass@k evaluation (n=32, k=[1,4,16,32])
  - Evaluate at step 800 (standard tier endpoint)
  - Optionally evaluate at step 1800 (final tier endpoint)
  - Compute shrinkage slope by comparing with Exp0 1.5B baseline
  - Compute transition matrix
  - Save results to `results/exp2_n1_vanilla_1.5B/`
  - _Requirements: 22.2, 22.6, 22.7_

**Checkpoint:** 1.5B Vanilla GRPO training complete (800 steps standard tier), Pass@k evaluated, resume capability verified, ready for final tier upgrade

---

### Task 12: Experiment 2 N1 - CB-GRPO at 0.5B (Standard Tier, Cluster Spend Logging)

**Objective:** Train 0.5B model using CB-GRPO (cluster-balanced gating) for 800 steps to test whether gradient-mass gating reduces boundary shrinkage at small scale.

**Dependencies:** Task 2 (clustering), Task 3 (model), Task 4 (rewards), Task 5 (CBGRPOGate), Task 6 (evaluation), Task 9 (checkpointing)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 41-43)

**Sub-tasks:**

- [x] 12.1 Configure Exp2 N1 CB-GRPO 0.5B
  - Create config: exp_name="exp2_n1_cbgrpo_0.5B", model_size="0.5B", gate_type="CBGRPOGate", compute_tier="standard"
  - Set gate_params: theta=1.5, decay=0.98, ema_alpha=0.05
  - Set training_steps=800 (standard tier)
  - _Requirements: 2.1, 2.3, 2.4, 22.3_

- [x] 12.2 Run training for 0.5B CB-GRPO
  - Generate G=4 rollouts per training prompt per step
  - Compute gradient-mass per sample (|advantage|)
  - Update cluster spend accumulators
  - Apply soft gating based on cluster over-budget ratio
  - Log per-cluster gradient-mass every 50 steps
  - Save checkpoint every 200 steps (include gate state)
  - _Requirements: 7.1-7.7, 22.3_

- [x] 12.3 Log CB-GRPO specific metrics
  - Display mean and std of cluster spend every 50 steps
  - Track spend distribution over time
  - Save cluster spend history to JSON every 200 steps
  - _Requirements: 7.5, 7.6, 24.1, 24.6_

- [x] 12.4 Run Pass@k evaluation every 150 steps
  - Evaluate on GSM8K test (n=10)
  - Compare Pass@k trajectory with Vanilla GRPO
  - Save intermediate results
  - _Requirements: 6.6, 6.7_

- [x] 12.5 Run final Pass@k evaluation (n=32, k=[1,4,16,32])
  - Evaluate on all three datasets
  - Compute shrinkage slope
  - Compute Δ slope = slope_cbgrpo - slope_vanilla (expect positive if CB-GRPO mitigates)
  - Compute transition matrix
  - Save results to `results/exp2_n1_cbgrpo_0.5B/`
  - _Requirements: 22.3, 22.6, 22.7_

- [x] 12.6 Generate cluster spend visualization
  - Plot histogram comparing cluster spend: Vanilla (uniform expected) vs. CB-GRPO (non-uniform)
  - Compute Gini coefficient for spend distributions
  - Save visualization to `figures/exp2_n1/cluster_spend_0.5B.png`
  - _Requirements: 7.7, 24.1, 24.2, 24.3, 24.4_

**Checkpoint:** 0.5B CB-GRPO training complete, cluster spend logged, shrinkage slope compared with Vanilla, spend visualization generated

---

### Task 13: Experiment 2 N1 - CB-GRPO at 1.5B (Standard→Final Upgrade, Cluster Spend Analysis)

**Objective:** Train 1.5B model using CB-GRPO for 800 steps initially (standard tier), with option to upgrade to final tier (1800 steps) for publication-quality results. Generate comprehensive cluster spend analysis for N3 novelty.

**Dependencies:** Task 2 (clustering), Task 3 (model), Task 4 (rewards), Task 5 (CBGRPOGate), Task 6 (evaluation), Task 9 (checkpointing)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 44-46)

**Sub-tasks:**

- [x] 13.1 Configure Exp2 N1 CB-GRPO 1.5B (standard tier)
  - Create config: exp_name="exp2_n1_cbgrpo_1.5B", model_size="1.5B", gate_type="CBGRPOGate", compute_tier="standard"
  - Set gate_params: theta=1.5, decay=0.98, ema_alpha=0.05
  - Set training_steps=800 (standard tier initially)
  - _Requirements: 2.1, 2.3, 2.4, 22.4_

- [x] 13.2 Run training for 1.5B CB-GRPO (standard tier)
  - Apply CB-GRPO gating with cluster spend tracking
  - Log per-cluster gradient-mass every 50 steps
  - Save checkpoint every 200 steps (include gate state with spend history)
  - Run Pass@k evaluation every 150 steps
  - _Requirements: 7.1-7.7, 22.4_

- [x] 13.3 Track cluster spend time series
  - Save per-cluster gradient-mass at every checkpoint
  - Enable post-experiment analysis of when clusters saturate
  - _Requirements: 24.6_

- [x] 13.4 Run intermediate Pass@k evaluation at step 800
  - Evaluate on GSM8K test (n=10)
  - Compute preliminary shrinkage slope
  - Compare with 1.5B Vanilla GRPO
  - _Requirements: 22.6, 22.7_

- [x] 13.5 Resume for final tier upgrade (optional)
  - After N2 baselines complete, upgrade to final tier (1800 steps)
  - Resume from step 800 checkpoint (preserves cluster spend state)
  - Continue training for additional 1000 steps
  - _Requirements: 30.3, 30.4_

- [x] 13.6 Run final Pass@k evaluation (n=32, k=[1,4,16,32])
  - Evaluate at step 800 and optionally at step 1800
  - Compute shrinkage slope
  - Compute Δ slope = slope_cbgrpo - slope_vanilla
  - Compute transition matrix
  - Save results to `results/exp2_n1_cbgrpo_1.5B/`
  - _Requirements: 22.4, 22.6, 22.7_

- [x] 13.7 Generate comprehensive CB-GRPO analysis (N3 novelty)
  - Plot side-by-side cluster spend histograms (Vanilla vs. CB-GRPO)
  - Compute Gini coefficient for both distributions
  - Plot cluster spend time series (top 5 clusters by final spend)
  - Generate table: CB-GRPO vs. Vanilla on shrinkage slope and Pass@k
  - Save all visualizations to `figures/exp2_n1/`
  - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.6_

**Checkpoint:** 1.5B CB-GRPO training complete (standard tier), cluster spend analysis generated, N3 novelty results ready, optional final tier upgrade capability verified

---

### Task 14: Experiment 2 N2 - Mitigation Baselines at 1.5B (O-SELF, Static SELF, Adaptive Rollout)

**Objective:** Test existing boundary-shrinkage mitigation techniques (O-SELF, Static SELF, Adaptive Rollout) at 1.5B scale to validate N2 novelty (transferability analysis).

**Dependencies:** Task 2 (datasets), Task 3 (model), Task 4 (rewards), Task 5 (all gates), Task 6 (evaluation), Task 9 (checkpointing)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 47-52)

**Sub-tasks:**

- [x] 14.1 Configure Exp2 N2 O-SELF 1.5B
  - Create config: exp_name="exp2_n2_oself_1.5B", model_size="1.5B", gate_type="OSELFGate", compute_tier="standard"
  - Set gate_params: threshold=0.7, decay=0.95, ema_alpha=0.1
  - Set training_steps=800
  - _Requirements: 2.1, 2.3, 2.4, 23.1_

- [x] 14.2 Run training for 1.5B O-SELF
  - Apply OSELFGate with online solve-rate tracking
  - Log gate statistics (filtered count, mean solve-rate) every 50 steps
  - Save checkpoint every 200 steps
  - Run Pass@k evaluation every 150 steps
  - _Requirements: 8.1-8.5, 23.1_

- [x] 14.3 Configure Exp2 N2 Static SELF 1.5B
  - Create config: exp_name="exp2_n2_staticself_1.5B", model_size="1.5B", gate_type="StaticSELFGate", compute_tier="standard"
  - Set gate_params: threshold=0.7
  - Precompute solve-rates using 1.5B base model (Pass@4 as proxy)
  - Save precomputed solve-rates to disk
  - Display filtered problem count
  - _Requirements: 2.1, 2.3, 2.4, 23.2_

- [x] 14.4 Run training for 1.5B Static SELF
  - Apply StaticSELFGate with precomputed filtering
  - Training excludes problems with solve-rate > threshold
  - Save checkpoint every 200 steps
  - Run Pass@k evaluation every 150 steps
  - _Requirements: 9.1-9.5, 23.2_

- [x] 14.5 Configure Exp2 N2 Adaptive Rollout 1.5B
  - Create config: exp_name="exp2_n2_adaptive_1.5B", model_size="1.5B", gate_type="AdaptiveRolloutGate", compute_tier="standard"
  - Set gate_params: threshold=0.1, decay=0.95
  - Set training_steps=800
  - _Requirements: 2.1, 2.3, 2.4, 23.3_

- [x] 14.6 Run training for 1.5B Adaptive Rollout
  - Apply AdaptiveRolloutGate with variance-based filtering
  - Log gate statistics (mean variance, filtered count) every 50 steps
  - Save checkpoint every 200 steps
  - Run Pass@k evaluation every 150 steps
  - _Requirements: 10.1-10.5, 23.3_

- [x] 14.7 Run final Pass@k evaluation for all N2 baselines
  - Evaluate O-SELF, Static SELF, Adaptive Rollout at step 800
  - Use n=32, k=[1,4,16,32] for all three approaches
  - Compute shrinkage slope for each
  - Compute Δ slope relative to 1.5B Vanilla baseline
  - Save results to respective `results/exp2_n2_{gate}/` directories
  - _Requirements: 23.1-23.6_

- [x] 14.8 Generate N2 transferability comparison table
  - Table columns: Gate, Pass@1, Pass@4, Pass@16, Pass@32, Shrinkage Slope, Δ Slope vs. Vanilla
  - Include Vanilla, CB-GRPO, O-SELF, Static SELF, Adaptive Rollout
  - Highlight best Δ slope (most effective mitigation)
  - Save table to `results/exp2_n2/comparison_table.md`
  - _Requirements: 23.4, 23.5_

- [x] 14.9 Generate Pass@k curves for all mitigation approaches
  - Plot Pass@k curves for Vanilla, CB-GRPO, O-SELF, Static SELF, Adaptive Rollout
  - Include base model curve for reference
  - Save visualization to `figures/exp2_n2/pass_at_k_comparison.png`
  - _Requirements: 23.6_

**Checkpoint:** All N2 baseline mitigation techniques trained at 1.5B, shrinkage slopes computed, transferability comparison table generated, N2 novelty results complete

---

### Task 15: CB-GRPO Ablation Studies (Optional, 1.5B Only)

**Objective:** Ablate CB-GRPO hyperparameters (n_clusters, theta, decay) at 1.5B scale to understand sensitivity and optimize performance.

**Dependencies:** Task 2 (clustering), Task 3 (model), Task 4 (rewards), Task 5 (CBGRPOGate), Task 6 (evaluation), Task 9 (checkpointing)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 53-55)

**Sub-tasks:**

- [x] 15.1 Configure ablation experiments
  - Baseline: n_clusters=16, theta=1.5, decay=0.98
  - Ablation 1: n_clusters=8 (coarser clustering)
  - Ablation 2: n_clusters=32 (finer clustering)
  - Ablation 3: theta=1.0 (earlier gating)
  - Ablation 4: theta=2.0 (later gating)
  - Ablation 5: decay=0.95 (stronger gating)
  - Ablation 6: decay=0.99 (weaker gating)
  - Use smoke tier (100 steps) for quick iteration

- [x] 15.2 Run ablation experiments
  - Train 1.5B CB-GRPO with each ablation setting
  - Run Pass@k evaluation at end of smoke tier (n=10)
  - Record shrinkage slope for each ablation
  - Record cluster spend Gini coefficient

- [x] 15.3 Generate ablation sensitivity analysis
  - Plot shrinkage slope vs. n_clusters, theta, decay
  - Identify optimal hyperparameter settings
  - Save ablation results to `results/exp2_ablations/`
  - Generate summary table showing sensitivity

**Checkpoint:** CB-GRPO ablation experiments complete, hyperparameter sensitivity understood, optimal settings identified

---

### Task 16: Experiment 3 - Base→GRPO Trajectory Ablation (Stretch, 1.5B Only)

**Objective:** Save intermediate checkpoints during Vanilla GRPO training and evaluate Pass@k at each checkpoint to identify when boundary-shrinkage emerges.

**Dependencies:** Task 3 (model), Task 4 (rewards), Task 5 (VanillaGate), Task 6 (evaluation), Task 9 (checkpointing)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 56-57)

**Sub-tasks:**

- [x] 16.1 Configure Exp3 trajectory ablation
  - Create config: exp_name="exp3_trajectory_1.5B", model_size="1.5B", gate_type="VanillaGate", compute_tier="standard"
  - Enable intermediate checkpoint saving every 100 steps
  - Set training_steps=800

- [x] 16.2 Run training with intermediate checkpointing
  - Train 1.5B Vanilla GRPO
  - Save checkpoint at steps: 100, 200, 300, 400, 500, 600, 700, 800
  - Continue training as normal

- [x] 16.3 Evaluate Pass@k at each intermediate checkpoint
  - Load each checkpoint and run Pass@k evaluation (n=10)
  - Compute shrinkage slope at each checkpoint
  - Record Pass@k and slope progression

- [x] 16.4 Generate trajectory analysis
  - Plot shrinkage slope evolution over training steps
  - Identify step range where boundary-shrinkage begins
  - Plot Pass@k curves at checkpoints: 0 (base), 200, 400, 600, 800
  - Save trajectory visualization to `figures/exp3/shrinkage_trajectory.png`
  - _Requirements: 25.1-25.5_

**Checkpoint:** Intermediate checkpoints evaluated, shrinkage emergence timeline identified, trajectory visualization generated

---

### Task 17: Comprehensive Analysis and Visualization (N1/N2/N3 Results)

**Objective:** Generate all publication-quality figures, tables, and analysis for N1 (capacity-scaling), N2 (transferability), and N3 (CB-GRPO mechanism) novelties.

**Dependencies:** Tasks 7, 10-14 (all experimental results)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 58-62)

**Sub-tasks:**

- [x] 17.1 Generate N1 capacity-scaling analysis
  - Plot shrinkage slope vs. trainable parameter count (0.5B, 1.5B)
  - Include both Vanilla and CB-GRPO for each model size
  - Display with 95% confidence intervals
  - Save to `figures/n1_capacity_scaling.png`
  - _Requirements: 22.6, 22.7_

- [x] 17.2 Generate N1 Pass@k comparison plots
  - Plot Pass@k curves: Base 0.5B, Vanilla 0.5B, CB-GRPO 0.5B
  - Plot Pass@k curves: Base 1.5B, Vanilla 1.5B, CB-GRPO 1.5B
  - Use plotly for interactive exploration
  - Save to `figures/n1_pass_at_k_{model_size}.html`
  - _Requirements: 18.5, 22.7_

- [x] 17.3 Generate N2 transferability comparison
  - Plot shrinkage slope for all 1.5B approaches: Vanilla, CB-GRPO, O-SELF, Static SELF, Adaptive Rollout
  - Display Δ slope relative to Vanilla baseline
  - Highlight CB-GRPO performance
  - Save to `figures/n2_transferability.png`
  - _Requirements: 23.5, 23.6_

- [x] 17.4 Generate N3 CB-GRPO mechanism analysis
  - Side-by-side cluster spend histograms (Vanilla vs. CB-GRPO) for both 0.5B and 1.5B
  - Cluster spend time series for 1.5B CB-GRPO (top 5 clusters)
  - Comparison table: shrinkage slope and Pass@k metrics
  - Compute and display Gini coefficient for spend distributions
  - Save to `figures/n3_cbgrpo_analysis.png`
  - _Requirements: 24.1-24.6_

- [x] 17.5 Generate transition matrices for all experiments
  - Compute kept_correct, lost_capability, gained_capability, kept_incorrect for each experiment
  - Display percentages and problem counts
  - Highlight experiments with lowest lost_capability percentage
  - Save transition matrix visualizations to `figures/transition_matrices/`
  - _Requirements: 17.1-17.7_

- [x] 17.6 Generate summary reports for all experiments
  - Create markdown summary for each experiment with key metrics
  - Include Pass@k results, shrinkage slope, transition matrix
  - Include training time, VRAM usage, checkpoint count
  - Provide interpretation guidance (significant/mild/no shrinkage)
  - Save to `results/{exp_name}/summary.md`
  - _Requirements: 19.9, 32.7_

- [x] 17.7 Export all results to structured JSON
  - Save all Pass@k results, shrinkage slopes, transition matrices to JSON
  - Include experiment metadata (config, timestamps, environment)
  - Enable post-experiment analysis and replotting
  - _Requirements: 19.1-19.9, 35.1-35.10_

**Checkpoint:** All N1/N2/N3 figures generated, transition matrices computed, summary reports created, results exported to JSON for paper writing

---

### Task 18: Optional vLLM Retrofit for Generation Speedup (Post-Core)

**Objective:** Replace HuggingFace generate() with vLLM for faster rollout generation during training and evaluation, reducing experiment runtime.

**Dependencies:** All core tasks complete (Tasks 1-14, 17)

**Implementation File:** `rlvr_training_pipeline.ipynb` (Cell 63-64)

**Sub-tasks:**

- [x] 18.1 Install and configure vLLM
  - Install vLLM package
  - Configure vLLM engine for Qwen2.5 models
  - Test compatibility with 4-bit quantization

- [x] 18.2 Implement vLLM RolloutGenerator
  - Replace sequential generation with vLLM batched generation
  - Maintain same generation parameters (temperature, top_p, max_new_tokens)
  - Benchmark speedup vs. HuggingFace generation

- [x] 18.3 Validate equivalence
  - Verify rewards match between HuggingFace and vLLM generation
  - Run smoke tier experiment with vLLM to verify training works
  - Compare Pass@k results with baseline experiments

**Checkpoint:** vLLM integration complete, generation speedup verified, training equivalence validated

---

### Task 19: Optional Streamlit Dashboard for Interactive Visualization (Post-Core)

**Objective:** Create interactive Streamlit dashboard for exploring experiment results, comparing approaches, and generating custom visualizations.

**Dependencies:** All core tasks complete (Tasks 1-14, 17)

**Implementation File:** `streamlit_dashboard.py` (separate file)

**Sub-tasks:**

- [x] 19.1 Implement dashboard layout
  - Sidebar for experiment selection
  - Main panel for visualization display
  - Tabs for different analysis types (Pass@k, shrinkage, clusters, transition matrices)

- [x] 19.2 Implement experiment comparison view
  - Multi-select for experiments to compare
  - Side-by-side Pass@k curves
  - Shrinkage slope comparison bar chart
  - Transition matrix comparison

- [x] 19.3 Implement cluster spend explorer
  - Interactive histogram for cluster spend distribution
  - Time series plot with cluster selection
  - Drill-down to view cluster member prompts

- [x] 19.4 Implement training metrics explorer
  - Loss, reward, success rate curves
  - Zoom and pan controls
  - Checkpoint marker overlay

- [x] 19.5 Deploy dashboard to Streamlit Cloud
  - Configure deployment settings
  - Test public access
  - Share link for paper reviewers

**Checkpoint:** Streamlit dashboard deployed, interactive exploration enabled, reviewer-friendly interface available

---

## Notes

### Task Execution Order

**Phase 1: Core Infrastructure (Tasks 1-6)**
- These tasks establish the foundational system and can proceed linearly
- Task 6 (evaluation suite) can be developed in parallel with Task 5 (gates)

**Phase 2: Baseline Establishment (Task 7)**
- Must complete before any training experiments
- Provides reference Pass@k values for shrinkage slope computation

**Phase 3: Checkpoint Validation (Task 9)**
- Critical to validate before running long experiments
- Smoke tier test ensures resume capability works

**Phase 4: N1 Core Experiments (Tasks 10-13)**
- Task 10 and 12 (0.5B experiments) can run in parallel
- Task 11 and 13 (1.5B experiments) can run in parallel after 0.5B complete
- Standard tier (800 steps) sufficient for initial results
- Final tier upgrade (1800 steps) optional for publication quality

**Phase 5: N2 Baseline Experiments (Task 14)**
- Requires 1.5B model only
- Three experiments (O-SELF, Static SELF, Adaptive Rollout) can run sequentially
- Standard tier (800 steps) sufficient

**Phase 6: Analysis and Visualization (Task 17)**
- Requires all core experiment results (Tasks 7, 10-14)
- Generates publication-ready figures and tables

**Phase 7: Optional Enhancements (Tasks 8, 15, 16, 18, 19)**
- Task 8 (SFT warmup): Tests alternative initialization, 1.5B only
- Task 15 (CB-GRPO ablations): Hyperparameter sensitivity, smoke tier (100 steps)
- Task 16 (trajectory ablation): Identifies when shrinkage emerges, standard tier
- Task 18 (vLLM): Speedup for future experiments
- Task 19 (Streamlit): Interactive exploration for reviewers

### Testing Approach

**Property-Based Tests (PBT):**
- Applied to parser/printer components (Task 4.2, 4.4)
- Use Hypothesis library with minimum 100 iterations per property
- Critical for ensuring reward computation correctness
- Failures saved with minimal counterexamples for debugging

**Unit Tests:**
- Applied to reward computation, gate logic, checkpoint integrity
- Test specific examples and edge cases
- Complement property tests with concrete scenarios

**Integration Tests:**
- Smoke tier experiments (100 steps) validate end-to-end pipeline
- Resume robustness test (Task 9.6) validates checkpoint system
- Final experiments serve as ultimate integration validation

### VRAM Management Strategy

**Model Loading:**
- 0.5B: ~4-6GB VRAM
- 1.5B: ~8-10GB VRAM
- 3B: ~12-14GB VRAM (stretch goal, not core)

**Training:**
- Batch size=4 target, reduce to 2 or 1 if OOM
- Gradient accumulation compensates for small batch size
- Sequential rollout generation (not batched)
- Reward computation on CPU

**Evaluation:**
- Generate solutions sequentially, not in parallel
- For n=32, generate in chunks of 8-10
- Clear GPU cache between training and evaluation

### Colab Session Management

**Checkpoint Frequency:**
- Standard tier (800 steps): checkpoint every 200 steps → 4 checkpoints
- Final tier (1800 steps): checkpoint every 150 steps → 12 checkpoints
- Always maintain last.pt and best.pt

**Expected Session Handling:**
- Standard tier: completes in single 4-6 hour session (no disconnect expected)
- Final tier: requires 10-12 hours total, expect 1-2 disconnects
- Resume protocol handles disconnects transparently

**Google Drive Storage:**
- All checkpoints saved to Drive (persists across sessions)
- Cluster assignments computed once, reused across experiments
- Results and figures saved incrementally

### Experiment Runtime Estimates

**Smoke Tier (100 steps):**
- 0.5B: ~30 minutes
- 1.5B: ~45 minutes

**Standard Tier (800 steps):**
- 0.5B: ~4 hours
- 1.5B: ~6 hours

**Final Tier (1800 steps):**
- 0.5B: ~9 hours
- 1.5B: ~12 hours (with resume cycles)

**Total Core Experimental Time:**
- Base evaluations (Exp0): ~2 hours
- N1 experiments (4 runs × standard tier): ~20 hours
- N2 experiments (3 runs × standard tier): ~18 hours
- Resume validation (smoke tier): ~1 hour
- **Total: ~41 hours of GPU time**

**With Final Tier Upgrade:**
- N1 upgrade (2 runs × additional 1000 steps): ~15 hours
- **Total with upgrades: ~56 hours of GPU time**

### Property Test Configuration

**Library:** Hypothesis 6.x

**Iterations:** Minimum 100 per property (due to randomization)

**Properties:**
1. Property 1: Parser extracts reasoning content (Task 4.2)
2. Property 2: Parser extracts answer content (Task 4.2)
3. Property 3: Parser extracts boxed content (Task 4.2)
4. Property 4: Pretty-printer formats valid structure (Task 4.3)
5. Property 5: Round-trip preservation - **PRIMARY** (Task 4.4)

**Test Execution:**
- Run during smoke tier initialization (Task 4.9)
- Re-run before major experiments as sanity check
- Failures logged with minimal counterexamples

### Optional Test Tasks

While property-based tests are marked optional with `*` notation, the parser/printer correctness properties (Task 4.2, 4.4) are **highly recommended** because:
- Reward computation correctness is critical to all experiments
- Round-trip property (Property 5) validates the entire parser/printer system
- Property tests catch edge cases that unit tests miss
- Hypothesis minimizes counterexamples for easy debugging

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["1.5", "1.6", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "3.1"] },
    { "id": 3, "tasks": ["2.5", "2.6", "3.2", "3.3", "4.1"] },
    { "id": 4, "tasks": ["3.4", "3.5", "3.6", "4.2", "4.3"] },
    { "id": 5, "tasks": ["4.4", "4.5", "4.6"] },
    { "id": 6, "tasks": ["4.7", "4.8", "4.9", "5.1"] },
    { "id": 7, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6"] },
    { "id": 8, "tasks": ["5.7", "6.1"] },
    { "id": 9, "tasks": ["6.2", "6.3", "6.4"] },
    { "id": 10, "tasks": ["6.5", "6.6", "6.7"] },
    { "id": 11, "tasks": ["7.1", "9.1"] },
    { "id": 12, "tasks": ["7.2", "7.3", "9.2"] },
    { "id": 13, "tasks": ["7.4", "9.3", "9.4"] },
    { "id": 14, "tasks": ["9.5", "9.6"] },
    { "id": 15, "tasks": ["10.1", "10.2"] },
    { "id": 16, "tasks": ["10.3", "11.1"] },
    { "id": 17, "tasks": ["10.4", "10.5", "11.2", "12.1"] },
    { "id": 18, "tasks": ["10.6", "10.7", "11.3", "12.2", "12.3"] },
    { "id": 19, "tasks": ["11.4", "12.4", "13.1"] },
    { "id": 20, "tasks": ["11.5", "12.5", "12.6", "13.2", "13.3"] },
    { "id": 21, "tasks": ["13.4", "13.5", "14.1"] },
    { "id": 22, "tasks": ["13.6", "13.7", "14.2", "14.3"] },
    { "id": 23, "tasks": ["14.4", "14.5"] },
    { "id": 24, "tasks": ["14.6", "14.7"] },
    { "id": 25, "tasks": ["14.8", "14.9", "17.1"] },
    { "id": 26, "tasks": ["17.2", "17.3", "17.4"] },
    { "id": 27, "tasks": ["17.5", "17.6", "17.7"] }
  ]
}
```

**Note:** Optional tasks (8, 15, 16, 18, 19) are not included in the dependency graph as they are time-permitting enhancements.

---

## Novelty Upgrade Tasks (Requirements 39–43, from `novelty_upgrade_addendum.md`)

These tasks implement the upgrades from the novelty addendum. They are **mandatory** for the N4 headline contribution and **Tier 0** (free) for N5 diagnostics.

---

### Task 18: H-CB-GRPO — CompositeGate Implementation (Requirement 39)

**Objective:** Implement the hierarchical composite gate that combines cluster-level macro gating (CBGRPOGate) and per-prompt micro gating (OSELFGate), as the headline N4 contribution.

**Dependencies:** Task 5 (CBGRPOGate + OSELFGate), Task 1 (configuration)

**Implementation File:** `rlvr_training_pipeline.ipynb` (§3g gates module), `gates.py`

**Sub-tasks:**

- [x] 18.1 Implement `CompositeGate` class conforming to `Gate` interface (Req 39.1)
  - Accept `macro: CBGRPOGate` and `micro: OSELFGate` as child gates
  - Route `local_increments` / `apply_synced_increments` via namespaced prefixes (`macro__`, `micro__`)
  - Implement `state_dict` / `load_state_dict` with nested macro/micro sub-dicts
  - _Requirements: 39.1_

- [x] 18.2 Implement positive-mass-only spend accumulator in `CBGRPOGate` (Req 39.2)
  - Replace `|advantage|` with `max(advantage, 0)` as the spend metric
  - Preserve `use_positive_mass_only=False` flag for ablation (reproduces original behavior)
  - Verify: `spend_ema[cluster_with_negative_adv]` stays 0 under positive-mass mode
  - _Requirements: 39.2_

- [x] 18.3 Implement multiplicative weight combination (Req 39.3-39.4)
  - Default: `gate_weight = macro_weight * micro_weight`
  - _Requirements: 39.3, 39.4_

- [x] 18.4 Add configurable combination operator (Req 39.5)
  - Support `multiplicative`, `min`, `harmonic_mean` via `combine_op` config field
  - Implement `_combine(macro_w, micro_w, op)` helper function
  - _Requirements: 39.5_

- [x] 18.5 Implement `diagnostic_snapshot()` for per-step gate weight logging (Req 39.6)
  - Log `cluster_spend_ema`, `gini`, `prompt_solve_ema_mean`, `frac_solve_above_tau`
  - Called in training loop at each `log_interval` step
  - _Requirements: 39.6_

- [x] 18.6 Add H-CB-GRPO to gate factory and training loop (Req 39.7)
  - Add `'h_cb_grpo'` to `GateType` literal and `build_gate()` factory
  - Register in `ExperimentConfig` gate validation
  - Run standard-tier 1.5B training alongside Vanilla/CB-GRPO/O-SELF conditions
  - _Requirements: 39.7_

- [x] 18.7 Validate 0.5B + CompositeGate incompatibility (Req 39.8)
  - Add `'h_cb_grpo'` to `INCOMPATIBLE_GATES_AT_0_5B` set
  - Validate at config construction time with clear `ValueError` message
  - Include self-test asserting the rejection
  - _Requirements: 39.8_

**Checkpoint:** CompositeGate passes all self-tests; 0.5B+h_cb_grpo raises ValueError; training loop logs macro/micro/combined weights separately

---

### Task 19: Mechanism-Level Diagnostics (Requirement 40)

**Objective:** Implement the SELF-paper interference metrics (Δ⁺, ‖Δ‖) and token entropy diagnostics to causally connect cluster-spend imbalance to shrinkage, converting \"CB-GRPO helped a little\" into a mechanism-level finding.

**Dependencies:** Task 7 (base model eval), Task 18 (all gate conditions trained)

**Implementation File:** `rlvr_training_pipeline.ipynb` (§3f diagnostics, §11), `diagnostics.py`

**Sub-tasks:**

- [x] 19.1 Build probing dataset of base-model-correct completions (Req 40.1)
  - Sample G=4 base-model responses per training prompt before training
  - Retain only the first correct (prompt, completion) pair per prompt
  - Cache to `artifacts/probing/probing_set.json` (never recompute)
  - Target 300 (prompt, completion) pairs
  - _Requirements: 40.1_

- [x] 19.2 Implement `compute_probing_logprobs()` for checkpoint evaluation (Req 40.2)
  - Forward pass on probing set under current model weights (no gradient)
  - Returns per-example mean per-token log-probability of the correct completion
  - Call once per checkpoint; feed consecutive snapshots to `compute_interference()`
  - _Requirements: 40.2_

- [x] 19.3 Implement `compute_interference()` for Δ⁺ and ‖Δ‖ (Req 40.2-40.3)
  - `delta_plus`: mean(log π_after(y⁺|x) − log π_before(y⁺|x)) — negative = negative interference
  - `delta_sq`: mean((log π_after − log π_before)²) — influence magnitude
  - Validated with unit tests (no drift, negative interference, improvement cases)
  - _Requirements: 40.2, 40.3_

- [x] 19.4 Implement mean token-level entropy from rollout logits (Req 40.4)
  - `entropy_from_probs()`: Shannon entropy (nats) from softmax probabilities
  - `mean_token_entropy()`: average within-sequence first, then across batch (equal weight)
  - `token_entropy_from_logits_torch()`: thin wrapper computing softmax in torch
  - _Requirements: 40.4_

- [x] 19.5 Report Δ⁺, ‖Δ‖, entropy for all gate conditions (Req 40.5)
  - Run `eval_full.py --mode interference` (or §11 cell) on final checkpoint vs. base
  - Display interference snapshot alongside Pass@k and shrinkage-slope curves
  - _Requirements: 40.5_

- [x] 19.6 Implement `correlate_exploratory()` for Gini ↔ interference (Req 40.6)
  - Compute Pearson and Spearman between per-condition Gini and Δ⁺/shrinkage slope
  - _Requirements: 40.6_

- [x] 19.7 Label correlation as exploratory with n caveat (Req 40.7)
  - Embed note text in `CorrelationResult.note` field
  - Always print the caveat in the summary report — never report as hypothesis test
  - _Requirements: 40.7_

**Checkpoint:** `compute_interference()` self-tests pass; probing set built and cached; interference snapshots generated for all trained conditions

---

### Task 20: Evaluation Range and Ceiling-Effect Mitigation (Requirement 41)

**Objective:** Detect and flag datasets where the base model is already saturated (Pass@k > 90%), preventing misleading shrinkage-slope claims.

**Dependencies:** Task 7 (base model eval)

**Implementation File:** `rlvr_training_pipeline.ipynb` (§6, §3e evaluation module), `evaluation.py`

**Sub-tasks:**

- [x] 20.1 Pre-training ceiling check on GSM8K, GSM8K-platinum, MATH-500 (Req 41.1)
  - Evaluate base-model Pass@1 and Pass@10 on all three datasets before any RL
  - Cache results to `baseline_pass_at_k.json`
  - _Requirements: 41.1_

- [x] 20.2 Implement `check_ceiling_effect()` flagging function (Req 41.2)
  - Flag any k-value where base-model Pass@k ≥ 90% as ceiling-limited
  - When flagged, treat MATH-500 (least saturated) as primary shrinkage evidence source
  - Print explicit ceiling warning in the summary report
  - _Requirements: 41.2_

- [x] 20.3 Increase eval samples per problem from n=10 to n=30 (Req 41.3)
  - Set `GenerationConfig.eval_samples_per_problem = 30` (not 10)
  - Evaluate Pass@k for k ∈ [1, 2, 3, 5, 10, 20] — extends measurable range
  - _Requirements: 41.3_

- [x] 20.4 Report maximum reliable k in summary (Req 41.4)
  - Compute maximum k where standard error is below threshold given n=30
  - Print in summary report alongside the ceiling-effect table
  - _Requirements: 41.4_

**Checkpoint:** Ceiling flags computed and logged; eval samples raised to 30; MATH-500 used as primary source when GSM8K is saturated

---

### Task 21: Multi-Seed Protocol and Significance Testing (Requirement 42)

**Objective:** Run the core 4-condition comparison with ≥3 seeds and bootstrap a CI on Δslope to distinguish real effects from seed noise.

**Dependencies:** Task 18 (all gates implemented), Task 20 (Pass@k eval)

**Implementation File:** `rlvr_training_pipeline.ipynb` (§12), `evaluation.py`

**Sub-tasks:**

- [x] 21.1 Run core comparison with ≥3 seeds (Req 42.1)
  - Seeds 0, 1, 2 for Vanilla, CB-GRPO, O-SELF, H-CB-GRPO at 1.5B standard tier
  - Each seed writes to its own `seed{N}/` directory under the experiment root
  - _Requirements: 42.1_

- [x] 21.2 Prioritize seed count over total steps when budget constrained (Req 42.2)
  - Use standard tier (800 steps) × 3 seeds, NOT final tier (1800 steps) × 1 seed
  - Reserve final tier for one confirmatory run after multi-seed results identify winner
  - _Requirements: 42.2_

- [x] 21.3 Implement `bootstrap_delta_slope_ci()` across seeds (Req 42.3)
  - Bootstrap CI on `Δslope = slope_condition − slope_vanilla` across seed dimension
  - 5000 bootstrap resamples; returns 95% CI and `excludes_zero` flag
  - Raises `ValueError` if fewer than 2 seeds per condition
  - _Requirements: 42.3_

- [x] 21.4 Enforce CI-excludes-zero requirement for improvement claims (Req 42.4)
  - Print explicit verdict per condition in summary report
  - Never use the word "improvement" if CI includes zero
  - _Requirements: 42.4_

- [x] 21.5 Reserve final tier for confirmatory runs (Req 42.5)
  - Add a note to the experiment checklist (Appendix of notebook)
  - Config validation prints a warning if `tier=final` is selected before multi-seed std results exist
  - _Requirements: 42.5_

**Checkpoint:** 3+ seeds per condition trained; `bootstrap_delta_slope_ci()` self-tests pass; verdict printed per condition in §12

---

### Task 22: Clustering Basis Ablation — Difficulty-Aware (Requirement 43)

**Objective:** Test whether clustering by difficulty (embedding + solve-rate) changes CB-GRPO's effectiveness vs. semantic-only clustering, to determine if the "topic" framing is the right granularity.

**Dependencies:** Task 7 (base model solve rates), Task 18 (CB-GRPO gate)

**Implementation File:** `rlvr_training_pipeline.ipynb` (§13, §3d clustering module), `clustering.py`

**Sub-tasks:**

- [x] 22.1 Compute per-prompt base-model solve rates (Req 43.1)
  - Reuse G=4 rollouts from the probing-set generation (Task 19.1) — do NOT recompute
  - Cache to `artifacts/solve_rates.json`
  - Display distribution: mean, fraction easy (>0.9), fraction hard (<0.1)
  - _Requirements: 43.1_

- [x] 22.2 Implement difficulty-aware clustering feature construction (Req 43.2)
  - Concatenate `[sentence-embedding, difficulty_weight * standardized_solve_rate]`
  - `basis='difficulty_aware'` path in `cluster_embeddings()` function
  - Unit test: identical embeddings but bimodal solve-rates → difficulty-aware clustering recovers the split, semantic-only cannot
  - _Requirements: 43.2_

- [x] 22.3 Re-run CB-GRPO with difficulty-aware clusters (Req 43.3)
  - 1.5B, standard tier, seed 0 — same hyperparameters as semantic-only CB-GRPO
  - Set `CLUSTERING_BASIS = 'difficulty_aware'` in §5 config cell
  - Compare shrinkage slope and Δslope-CI against semantic-only clustering
  - _Requirements: 43.3_

- [x] 22.4 Report as a clearly labeled ablation table, not a new method (Req 43.4)
  - Section header: "Clustering Basis Ablation (Req 43 — diagnostic, not a new headline method)"
  - One table: semantic vs. difficulty-aware clustering, slope + Δslope-CI side-by-side
  - One-sentence interpretation: does the clustering basis matter for CB-GRPO?
  - _Requirements: 43.4_

**Checkpoint:** `cluster_embeddings(basis='difficulty_aware')` self-test passes; solve rates cached; ablation run and table produced in §13
