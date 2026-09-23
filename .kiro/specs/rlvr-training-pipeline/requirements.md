# Requirements Document: RLVR Training Pipeline for Reasoning Boundary Paradox Research

## Introduction

This document specifies requirements for a Jupyter notebook-based research pipeline investigating the "Reasoning Boundary Paradox" in Small Language Models (SLMs) trained via Reinforcement Learning with Verifiable Rewards (RLVR). The system enables empirical measurement of boundary-shrinkage severity across parameter scales (0.5B-1.5B), testing of existing mitigation techniques, and validation of a novel Capacity-Budgeted GRPO (CB-GRPO) algorithm.

The system must operate within Google Colab free-tier constraints (T4 GPU, 16GB VRAM, 12-hour sessions with 90-minute idle disconnect) and produce results suitable for IEEE conference paper submission.

### Research Objectives

- **N1:** Boundary-shrinkage detection across 0.5B/1.5B parameter scales.
- **N2:** Test whether existing mitigations (O-SELF, Static-SELF, Adaptive-Rollout) transfer to SLMs.
- **N3:** Capacity–shrinkage relationship analysis.
- **N4 (headline contribution):** Does topic-level (macro) and prompt-level (micro) gating address *distinct* components of reasoning-boundary shrinkage, such that combining them is super-additive relative to either alone?
- **N5 (diagnostic):** Does the SELF paper's interference metric (Δ⁺, ‖Δ‖) mediate the relationship between cluster-spend imbalance (Gini) and shrinkage slope?

## Glossary

- **System**: The complete Jupyter notebook-based training and evaluation pipeline
- **Notebook**: The single comprehensive Jupyter notebook containing all experiments
- **Base_Model**: Pretrained Qwen2.5 model (0.5B or 1.5B Instruct variant)
- **RL_Model**: Fine-tuned model after RLVR training
- **Gate**: Implementation class controlling prompt filtering/selection during training
- **Cluster**: Group of semantically similar training prompts
- **Gradient_Mass**: Cumulative gradient norm accumulated per cluster during training
- **Shrinkage_Slope**: Linear regression slope of log(k) vs. Δ Pass@k, measuring boundary-shrinkage severity
- **Pass@k**: Unbiased estimator measuring probability that at least one of k generated solutions is correct
- **Rollout**: Single forward pass generating G solutions per training prompt
- **Checkpoint**: Saved model state and optimizer state enabling training resumption
- **Smoke_Tier**: 50-100 training steps for pipeline validation (~30-60 minutes)
- **Standard_Tier**: 600-1000 training steps for main experimental results (~4-6 hours)
- **Final_Tier**: 1500-2000 training steps for publication-quality results (~8-10 hours with resume cycles)
- **Format_Reward**: Reward component for valid XML output structure
- **Correctness_Reward**: Reward component for mathematically correct answer
- **VRAM**: Video RAM available on GPU (16GB on T4)
- **QLoRA**: Quantized Low-Rank Adaptation fine-tuning technique
- **GSM8K**: Grade School Math 8K problem dataset
- **Resume_Checkpoint**: Checkpoint file enabling continuation after session disconnect
- **CompositeGate**: Gate combining cluster-level (macro) and prompt-level (micro) gating signals for H-CB-GRPO
- **Macro_Weight**: Cluster-level gate weight computed from positive-advantage EMA spend
- **Micro_Weight**: Prompt-level gate weight computed from per-prompt solve-rate EMA
- **Interference_Metric**: Δ⁺(π_θ, π_b) measuring expected change in log-probability of base-model-correct completions
- **Relative_Influence**: ‖Δ(π_θ, π_b)‖ measuring expected squared change in log-probability over all completions
- **Difficulty_Aware_Cluster**: Cluster formed using both semantic embedding and base-model solve-rate
- **Probing_Dataset**: Set of base-model responses used to compute interference metrics at checkpoints
- **Ceiling_Limited**: Dataset where base-model Pass@k exceeds 90%, limiting measurable shrinkage effect

## Requirements

### Requirement 1: Environment Initialization

**User Story:** As a researcher, I want the Notebook to initialize the runtime environment automatically, so that I can start experiments without manual setup.

#### Acceptance Criteria

1. THE Notebook SHALL mount Google Drive to persist checkpoints and results
2. WHEN dependencies are missing, THE System SHALL install required packages (unsloth, transformers, trl, sentence-transformers, sympy, plotly)
3. THE Notebook SHALL verify GPU availability and display device information
4. THE System SHALL create output directory structure (checkpoints/, results/, figures/) if not present
5. WHEN VRAM is below 14GB, THE System SHALL display a warning message
6. THE Notebook SHALL set random seeds for reproducibility (torch, numpy, random)

### Requirement 2: Configuration Management

**User Story:** As a researcher, I want to configure experiments through centralized parameters, so that I can run different experimental conditions without code modification.

#### Acceptance Criteria

1. THE System SHALL define configuration dictionaries for each experiment type (Exp0, Exp1, Exp2, Exp3)
2. THE System SHALL support compute tier selection (smoke, standard, final) via single parameter
3. THE System SHALL support model size selection (0.5B, 1.5B) via single parameter
4. THE System SHALL support gate type selection (VanillaGate, CBGRPOGate, OSELFGate, StaticSELFGate, AdaptiveRolloutGate) via single parameter
5. THE System SHALL support reward mode selection (positive-only, negative-only, hybrid) via single parameter
6. THE System SHALL validate configuration compatibility before execution
7. WHEN configuration is invalid, THE System SHALL display clear error messages with correction guidance

### Requirement 3: Model Loading and Quantization

**User Story:** As a researcher, I want to load Qwen2.5 models with memory-efficient quantization, so that training fits within T4 GPU VRAM constraints.

#### Acceptance Criteria

1. THE System SHALL load Qwen2.5-0.5B-Instruct using 4-bit NF4 quantization
2. THE System SHALL load Qwen2.5-1.5B-Instruct using 4-bit NF4 quantization
3. THE System SHALL apply QLoRA with rank=16 and alpha=32
4. WHEN model loading succeeds, THE System SHALL display VRAM consumption
5. WHEN VRAM consumption exceeds 14GB, THE System SHALL raise an error before training
6. THE System SHALL configure models for gradient checkpointing to reduce memory
7. FOR ALL loaded models, the System SHALL verify tokenizer compatibility with GSM8K format

### Requirement 4: Dataset Loading and Preprocessing

**User Story:** As a researcher, I want to load and preprocess mathematical reasoning datasets, so that training and evaluation data are properly formatted.

#### Acceptance Criteria

1. THE System SHALL load openai/gsm8k train split (7473 problems)
2. THE System SHALL load openai/gsm8k test split (1319 problems)
3. THE System SHALL load madrylab/gsm8k-platinum (1319 problems)
4. THE System SHALL load HuggingFaceH4/MATH-500 (500 problems)
5. THE System SHALL parse ground truth answers from dataset format
6. THE System SHALL format prompts with instruction template matching Base_Model chat format
7. WHEN dataset loading fails, THE System SHALL display descriptive error with dataset name
8. THE System SHALL display sample count and example prompt after loading

### Requirement 5: Prompt Clustering

**User Story:** As a researcher, I want training prompts clustered by semantic similarity, so that CB-GRPO can track gradient-mass per cluster.

#### Acceptance Criteria

1. THE System SHALL embed all training prompts using sentence-transformers (all-MiniLM-L6-v2)
2. THE System SHALL cluster embeddings using KMeans with n_clusters=16
3. THE System SHALL assign cluster ID to each training prompt
4. THE System SHALL save cluster assignments to disk
5. WHEN cluster assignments exist on disk, THE System SHALL load them instead of recomputing
6. THE System SHALL display cluster size distribution after clustering
7. FOR ALL training prompts, the System SHALL maintain cluster ID mapping throughout training

### Requirement 6: Vanilla GRPO Training

**User Story:** As a researcher, I want to train models using standard GRPO, so that I have baseline results for comparison.

#### Acceptance Criteria

1. WHEN VanillaGate is selected, THE System SHALL train without prompt filtering
2. THE System SHALL generate G=4 rollouts per training prompt per step
3. THE System SHALL compute rewards for all rollouts
4. THE System SHALL apply GRPO gradient updates using TRL's GRPOTrainer
5. THE System SHALL log training metrics (loss, reward, learning_rate) every 10 steps
6. THE System SHALL evaluate Pass@k on test set every 150 steps
7. WHEN training completes, THE System SHALL save final checkpoint
8. THE System SHALL train 0.5B model for configured tier duration
9. THE System SHALL train 1.5B model for configured tier duration

### Requirement 7: CB-GRPO Training with Gradient-Mass Gating

**User Story:** As a researcher, I want to train models using CB-GRPO with cluster-level gradient-mass tracking, so that I can test the N3 novelty contribution.

#### Acceptance Criteria

1. WHEN CBGRPOGate is selected, THE System SHALL initialize gradient-mass accumulator per cluster
2. THE System SHALL track cumulative gradient norm per cluster after each training step
3. WHEN cluster gradient-mass exceeds threshold, THE System SHALL reduce sampling probability for that cluster
4. THE System SHALL apply gating decisions before generating rollouts
5. THE System SHALL log per-cluster gradient-mass every 50 steps
6. THE System SHALL save cluster spend distribution to disk at checkpoint intervals
7. WHEN training completes, THE System SHALL generate histogram comparing cluster spend vs. VanillaGate
8. THE System SHALL train 0.5B model with CB-GRPO for configured tier duration
9. THE System SHALL train 1.5B model with CB-GRPO for configured tier duration

### Requirement 8: O-SELF Gating

**User Story:** As a researcher, I want to train models using Online SELF gating, so that I can test N2 mitigation transferability.

#### Acceptance Criteria

1. WHEN OSELFGate is selected, THE System SHALL compute online EMA solve-rate per problem
2. THE System SHALL update EMA solve-rate after each rollout evaluation
3. WHEN problem solve-rate exceeds threshold, THE System SHALL reduce sampling probability
4. THE System SHALL apply gating decisions before generating rollouts
5. THE System SHALL log gate statistics (filtered_count, mean_solve_rate) every 50 steps
6. THE System SHALL train 1.5B model only for configured tier duration

### Requirement 9: Static SELF Gating

**User Story:** As a researcher, I want to train models using Static SELF gating with precomputed solve-rates, so that I can test N2 mitigation transferability.

#### Acceptance Criteria

1. WHEN StaticSELFGate is selected, THE System SHALL precompute solve-rates using Base_Model
2. THE System SHALL save precomputed solve-rates to disk
3. WHEN precomputed solve-rates exist, THE System SHALL load them
4. WHEN problem solve-rate exceeds threshold, THE System SHALL exclude problem from training
5. THE System SHALL display filtered problem count before training
6. THE System SHALL train 1.5B model only for configured tier duration

### Requirement 10: Adaptive Rollout Gating

**User Story:** As a researcher, I want to train models using Adaptive Rollout gating with variance-based filtering, so that I can test N2 mitigation transferability.

#### Acceptance Criteria

1. WHEN AdaptiveRolloutGate is selected, THE System SHALL compute reward variance per problem
2. THE System SHALL update variance estimate after each rollout evaluation
3. WHEN reward variance is below threshold, THE System SHALL reduce sampling probability
4. THE System SHALL apply gating decisions before generating rollouts
5. THE System SHALL log gate statistics (mean_variance, filtered_count) every 50 steps
6. THE System SHALL train 1.5B model only for configured tier duration

### Requirement 11: Format Reward Computation

**User Story:** As a researcher, I want to compute format rewards based on XML structure validation, so that models learn to produce properly structured outputs.

#### Acceptance Criteria

1. THE System SHALL parse generated text for `<reasoning>` opening and closing tags
2. THE System SHALL parse generated text for `<answer>` opening and closing tags
3. THE System SHALL parse generated text for `\boxed{...}` within answer tags
4. WHEN all format requirements are met, THE System SHALL assign format reward of 1.0
5. WHEN any format requirement is missing, THE System SHALL assign format reward of 0.0
6. THE System SHALL execute format checking on CPU to avoid GPU memory overhead
7. FOR ALL generated rollouts, the System SHALL compute format rewards before correctness checking

### Requirement 12: Correctness Reward Computation

**User Story:** As a researcher, I want to compute correctness rewards by comparing extracted answers with ground truth, so that models learn mathematical accuracy.

#### Acceptance Criteria

1. THE System SHALL extract numeric answer from `\boxed{...}` content
2. THE System SHALL normalize extracted answer using sympy parsing
3. THE System SHALL normalize ground truth answer using sympy parsing
4. THE System SHALL compare normalized answers using sympy equivalence checking
5. WHEN reward mode is positive-only AND answers match, THE System SHALL assign correctness reward of 1.0
6. WHEN reward mode is positive-only AND answers differ, THE System SHALL assign correctness reward of 0.0
7. WHEN reward mode is negative-only AND answers match, THE System SHALL assign correctness reward of 0.0
8. WHEN reward mode is negative-only AND answers differ, THE System SHALL assign correctness reward of -1.0
9. WHEN extraction or parsing fails, THE System SHALL assign correctness reward of 0.0
10. FOR ALL generated rollouts, the System SHALL compute correctness rewards after format checking

### Requirement 13: Combined Reward Computation

**User Story:** As a researcher, I want to combine format and correctness rewards, so that training optimizes for both structure and accuracy.

#### Acceptance Criteria

1. THE System SHALL compute total reward as sum of format reward and correctness reward
2. THE System SHALL pass total rewards to GRPOTrainer for gradient computation
3. THE System SHALL log mean reward per batch every 10 steps
4. THE System SHALL log format success rate per batch every 10 steps
5. THE System SHALL log correctness success rate per batch every 10 steps

### Requirement 14: Checkpoint Management

**User Story:** As a researcher, I want automatic checkpoint saving with resume capability, so that training survives Colab session disconnects.

#### Acceptance Criteria

1. THE System SHALL save checkpoint containing model state, optimizer state, and training step
2. WHEN training duration exceeds 4 hours, THE System SHALL save checkpoint every 150 steps
3. WHEN training duration is below 4 hours, THE System SHALL save checkpoint every 200 steps
4. THE System SHALL save checkpoint with filename format `step_{step_num}.pt`
5. THE System SHALL maintain `last.pt` checkpoint pointing to most recent save
6. THE System SHALL maintain `best.pt` checkpoint pointing to highest eval Pass@1 score
7. WHEN resume checkpoint exists, THE System SHALL load model state, optimizer state, and training step
8. WHEN resume checkpoint exists, THE System SHALL continue training from loaded step
9. THE System SHALL save all Pass@k evaluation results as JSON files
10. THE System SHALL save cluster assignments, gate statistics, and gradient-mass logs to disk

### Requirement 15: Pass@k Evaluation

**User Story:** As a researcher, I want to evaluate Pass@k using the unbiased estimator, so that I can measure boundary-shrinkage accurately.

#### Acceptance Criteria

1. THE System SHALL generate n=10 solutions per test problem
2. THE System SHALL evaluate correctness for all generated solutions
3. THE System SHALL compute Pass@k for k in [1, 2, 3, 5, 10]
4. THE System SHALL use unbiased estimator formula: Pass@k = 1 - C(n-c, k) / C(n, k)
5. THE System SHALL compute Pass@k on openai/gsm8k test split
6. THE System SHALL compute Pass@k on madrylab/gsm8k-platinum
7. THE System SHALL compute Pass@k on HuggingFaceH4/MATH-500
8. THE System SHALL save Pass@k results to JSON file with experiment metadata
9. THE System SHALL display Pass@k values in formatted table
10. WHEN evaluation completes, THE System SHALL log evaluation duration

### Requirement 16: Shrinkage Slope Computation

**User Story:** As a researcher, I want to compute shrinkage slope from Pass@k curves, so that I can quantify boundary-shrinkage severity for N1 analysis.

#### Acceptance Criteria

1. THE System SHALL compute Δ Pass@k as difference between Base_Model and RL_Model for each k
2. THE System SHALL fit linear regression of log(k) vs. Δ Pass@k
3. THE System SHALL extract slope coefficient as shrinkage slope metric
4. THE System SHALL compute shrinkage slope for each evaluation dataset
5. THE System SHALL save shrinkage slope values to JSON file
6. THE System SHALL display shrinkage slope with 95% confidence interval
7. WHEN Pass@k values are invalid, THE System SHALL skip shrinkage slope computation and log warning

### Requirement 17: Transition Matrix Computation

**User Story:** As a researcher, I want to compute per-problem capability transition matrices, so that I can analyze which problems gained or lost correctness capability.

#### Acceptance Criteria

1. THE System SHALL classify each test problem as correct or incorrect for Base_Model
2. THE System SHALL classify each test problem as correct or incorrect for RL_Model
3. THE System SHALL compute transition counts: kept_correct, lost_capability, gained_capability, kept_incorrect
4. THE System SHALL compute transition matrix as percentage of total problems
5. THE System SHALL save transition matrix to JSON file with problem IDs per category
6. THE System SHALL display transition matrix as formatted table
7. THE System SHALL compute transition matrix for each evaluation dataset

### Requirement 18: Training Visualization

**User Story:** As a researcher, I want inline visualizations of training metrics, so that I can monitor training progress without post-processing.

#### Acceptance Criteria

1. THE System SHALL plot training loss curve updated every 50 steps
2. THE System SHALL plot mean reward curve updated every 50 steps
3. THE System SHALL plot format success rate curve updated every 50 steps
4. THE System SHALL plot correctness success rate curve updated every 50 steps
5. THE System SHALL plot Pass@k curves comparing Base_Model and RL_Model after each evaluation
6. WHEN CBGRPOGate is active, THE System SHALL plot per-cluster gradient-mass histogram every 200 steps
7. THE System SHALL use matplotlib or plotly for all visualizations
8. THE System SHALL display visualizations inline in Notebook output cells

### Requirement 19: Results Organization and Export

**User Story:** As a researcher, I want all results organized in structured directories, so that I can perform post-experiment analysis and generate paper figures.

#### Acceptance Criteria

1. THE System SHALL save all checkpoints to `checkpoints/{exp_name}/{model_size}/` directory
2. THE System SHALL save all Pass@k JSON results to `results/{exp_name}/{model_size}/` directory
3. THE System SHALL save all figures to `figures/{exp_name}/{model_size}/` directory
4. THE System SHALL save training logs to `logs/{exp_name}/{model_size}.log` file
5. THE System SHALL save experiment configuration to `results/{exp_name}/config.json`
6. THE System SHALL save shrinkage slope results to `results/{exp_name}/shrinkage_slopes.json`
7. THE System SHALL save transition matrices to `results/{exp_name}/transition_matrices.json`
8. WHEN CBGRPOGate is active, THE System SHALL save cluster spend distributions to `results/{exp_name}/cluster_spend.json`
9. THE System SHALL create summary report with all key metrics in `results/{exp_name}/summary.md`

### Requirement 20: Base Model Evaluation (Exp0)

**User Story:** As a researcher, I want to evaluate base models without training, so that I have baseline metrics for N1 analysis.

#### Acceptance Criteria

1. THE System SHALL load Qwen2.5-0.5B-Instruct base model
2. THE System SHALL load Qwen2.5-1.5B-Instruct base model
3. THE System SHALL evaluate Pass@k for 0.5B model on all evaluation datasets
4. THE System SHALL evaluate Pass@k for 1.5B model on all evaluation datasets
5. THE System SHALL save base model Pass@k results separately for use in comparison
6. THE System SHALL complete evaluation in inference-only mode without training
7. THE System SHALL display base model results in formatted tables

### Requirement 21: SFT Warmup Training (Exp1)

**User Story:** As a researcher, I want to perform supervised fine-tuning warmup, so that I can test whether SFT initialization improves RLVR training outcomes.

#### Acceptance Criteria

1. WHEN Exp1 is enabled, THE System SHALL train 1.5B model using supervised learning on GSM8K train split
2. THE System SHALL use ground truth solutions as SFT targets
3. THE System SHALL train for 500-1000 steps depending on compute tier
4. THE System SHALL save SFT checkpoint separately
5. THE System SHALL evaluate Pass@k on SFT model before RLVR training
6. THE System SHALL optionally use SFT checkpoint as initialization for subsequent RLVR training

### Requirement 22: N1 Capacity-Scaling Experiments (Exp2 Core)

**User Story:** As a researcher, I want to measure boundary-shrinkage severity at 0.5B and 1.5B scales, so that I can produce N1 novelty results.

#### Acceptance Criteria

1. THE System SHALL train 0.5B model using VanillaGate for standard tier minimum
2. THE System SHALL train 1.5B model using VanillaGate for standard tier minimum
3. THE System SHALL train 0.5B model using CBGRPOGate for standard tier minimum
4. THE System SHALL train 1.5B model using CBGRPOGate for standard tier minimum
5. THE System SHALL compute shrinkage slope for all four trained models
6. THE System SHALL generate plot showing shrinkage slope vs. trainable parameter count
7. THE System SHALL generate plot showing Pass@k curves per model size comparing Base_Model and RL_Model

### Requirement 23: N2 Mitigation Transferability Experiments (Exp2 Grid)

**User Story:** As a researcher, I want to test existing mitigation techniques at 1.5B scale, so that I can produce N2 novelty results.

#### Acceptance Criteria

1. THE System SHALL train 1.5B model using OSELFGate for standard tier
2. THE System SHALL train 1.5B model using StaticSELFGate for standard tier
3. THE System SHALL train 1.5B model using AdaptiveRolloutGate for standard tier
4. THE System SHALL compute shrinkage slope for all three mitigation approaches
5. THE System SHALL generate table showing Δslope comparing each mitigation to VanillaGate baseline
6. THE System SHALL generate plot showing Pass@k curves for all mitigation approaches

### Requirement 24: N3 CB-GRPO Analysis

**User Story:** As a researcher, I want to analyze CB-GRPO cluster spend distributions, so that I can produce N3 novelty results.

#### Acceptance Criteria

1. THE System SHALL compute cluster spend histogram for VanillaGate (uniform expected)
2. THE System SHALL compute cluster spend histogram for CBGRPOGate (non-uniform expected)
3. THE System SHALL generate side-by-side histogram plot comparing cluster spend distributions
4. THE System SHALL compute cluster spend Gini coefficient for both approaches
5. THE System SHALL generate table comparing CBGRPOGate vs. VanillaGate on shrinkage slope and Pass@k metrics
6. THE System SHALL save per-cluster gradient-mass time series for detailed analysis

### Requirement 25: Base-to-GRPO Ablation (Exp3)

**User Story:** As a researcher, I want to ablate intermediate training checkpoints, so that I can analyze when boundary-shrinkage emerges during training.

#### Acceptance Criteria

1. WHEN Exp3 is enabled, THE System SHALL save intermediate checkpoints every 100 steps
2. THE System SHALL evaluate Pass@k on each intermediate checkpoint
3. THE System SHALL compute shrinkage slope for each intermediate checkpoint
4. THE System SHALL generate plot showing shrinkage slope evolution over training steps
5. THE System SHALL identify step range where boundary-shrinkage begins

### Requirement 26: Logging and Verbosity Control

**User Story:** As a researcher, I want configurable logging verbosity, so that I can control output detail level for different experimental stages.

#### Acceptance Criteria

1. THE System SHALL support three verbosity levels: smoke, standard, verbose
2. WHEN verbosity is smoke, THE System SHALL log only critical milestones and errors
3. WHEN verbosity is standard, THE System SHALL log training metrics every 10 steps and evaluation results
4. WHEN verbosity is verbose, THE System SHALL log detailed per-rollout rewards and gate decisions
5. THE System SHALL write all logs to both console and log file
6. THE System SHALL include timestamp and experiment name in all log entries

### Requirement 27: Error Handling and Recovery

**User Story:** As a researcher, I want robust error handling, so that experiments fail gracefully with actionable error messages.

#### Acceptance Criteria

1. WHEN VRAM allocation fails, THE System SHALL display current VRAM usage and suggest reducing model size or batch size
2. WHEN checkpoint loading fails, THE System SHALL display checkpoint path and suggest starting from scratch
3. WHEN dataset loading fails, THE System SHALL display dataset name and suggest checking internet connection
4. WHEN evaluation fails mid-execution, THE System SHALL save partial results and continue
5. WHEN reward computation encounters parsing errors, THE System SHALL assign zero reward and log problem ID
6. WHEN training diverges (loss > 100), THE System SHALL pause training and display warning
7. THE System SHALL wrap all major operations in try-except blocks with descriptive error messages

### Requirement 28: Memory Management

**User Story:** As a researcher, I want automatic memory management, so that training stays within T4 VRAM limits.

#### Acceptance Criteria

1. THE System SHALL verify VRAM headroom before loading models (minimum 2GB free)
2. THE System SHALL apply gradient checkpointing for all models
3. THE System SHALL clear GPU cache between major operations
4. THE System SHALL monitor VRAM usage during training and log peaks
5. WHEN VRAM usage exceeds 14GB, THE System SHALL reduce batch size automatically if possible
6. THE System SHALL unload models from GPU when not in use
7. THE System SHALL display VRAM usage after each major operation

### Requirement 29: Reproducibility

**User Story:** As a researcher, I want deterministic experiment execution, so that results are reproducible across runs.

#### Acceptance Criteria

1. THE System SHALL set random seed for PyTorch, NumPy, and Python random module
2. THE System SHALL save random seed value to experiment configuration
3. THE System SHALL use deterministic algorithms when available
4. THE System SHALL save complete environment information (package versions, CUDA version, GPU model)
5. THE System SHALL save exact model checkpoint SHA hash
6. THE System SHALL document any non-deterministic operations in logs

### Requirement 30: Session Resume After Disconnect

**User Story:** As a researcher, I want training to resume cleanly after Colab disconnect, so that I can complete long experiments across multiple sessions.

#### Acceptance Criteria

1. WHEN Notebook restarts, THE System SHALL detect existing checkpoints in Google Drive
2. WHEN Resume_Checkpoint exists, THE System SHALL display checkpoint information (step, timestamp, metrics)
3. THE System SHALL prompt user to confirm resume or start fresh
4. WHEN user confirms resume, THE System SHALL load Resume_Checkpoint and continue from saved step
5. THE System SHALL verify checkpoint integrity before loading
6. WHEN checkpoint is corrupted, THE System SHALL display error and fallback to previous checkpoint
7. THE System SHALL merge logs from previous session with current session

### Requirement 31: Generation Backend

**User Story:** As a researcher, I want to generate multiple solutions per prompt efficiently, so that Pass@k evaluation completes within time constraints.

#### Acceptance Criteria

1. THE System SHALL use HuggingFace generate() with Unsloth kernel optimizations
2. THE System SHALL generate G=4 rollouts per prompt during training
3. THE System SHALL generate n=10 solutions per prompt during Pass@k evaluation
4. THE System SHALL configure temperature=0.7 and top_p=0.95 for sampling
5. THE System SHALL configure max_new_tokens=512 for generation
6. THE System SHALL batch generation requests when memory permits
7. WHEN generation fails or times out, THE System SHALL retry up to 3 times before skipping problem

### Requirement 32: Pretty Printer for Results

**User Story:** As a researcher, I want formatted display of results and metrics, so that I can quickly interpret experiment outcomes.

#### Acceptance Criteria

1. THE System SHALL format Pass@k results as aligned tables with dataset names and k values
2. THE System SHALL format shrinkage slope results with ± confidence intervals
3. THE System SHALL format transition matrices as percentage tables
4. THE System SHALL format cluster spend distributions as ASCII histograms when plots unavailable
5. THE System SHALL format training metrics with appropriate precision (loss: 4 decimals, rewards: 2 decimals)
6. THE System SHALL display experiment progress bars during long operations
7. THE System SHALL highlight key metrics in summary reports using markdown formatting

### Requirement 33: Configuration Validation

**User Story:** As a researcher, I want configuration validation before experiment execution, so that I catch errors early.

#### Acceptance Criteria

1. THE System SHALL validate model size is either 0.5B or 1.5B
2. THE System SHALL validate compute tier is one of: smoke, standard, final
3. THE System SHALL validate gate type is one of: VanillaGate, CBGRPOGate, OSELFGate, StaticSELFGate, AdaptiveRolloutGate
4. WHEN model is 0.5B AND gate is OSELFGate, THE System SHALL reject configuration
5. WHEN model is 0.5B AND gate is StaticSELFGate, THE System SHALL reject configuration
6. WHEN model is 0.5B AND gate is AdaptiveRolloutGate, THE System SHALL reject configuration
7. WHEN compute tier is smoke AND final tier results are requested, THE System SHALL display warning
8. THE System SHALL validate batch size produces acceptable VRAM usage for selected model size

### Requirement 34: Dataset Sample Inspection

**User Story:** As a researcher, I want to inspect dataset samples and model outputs, so that I can verify data quality and debug formatting issues.

#### Acceptance Criteria

1. THE System SHALL display 3 random training prompts after dataset loading
2. THE System SHALL display 3 random test problems with ground truth answers
3. THE System SHALL display example model generation with format and correctness rewards
4. THE System SHALL provide function to display specific problem by ID
5. THE System SHALL provide function to display all generations for a specific problem
6. THE System SHALL provide function to display cluster members for a specific cluster ID

### Requirement 35: Experiment Metadata Tracking

**User Story:** As a researcher, I want comprehensive metadata saved with each experiment, so that I can trace results back to exact experimental conditions.

#### Acceptance Criteria

1. THE System SHALL save experiment start timestamp
2. THE System SHALL save experiment end timestamp
3. THE System SHALL save model name, size, and quantization settings
4. THE System SHALL save gate type and gate-specific parameters
5. THE System SHALL save compute tier and actual training steps
6. THE System SHALL save random seed and environment information
7. THE System SHALL save hyperparameters (learning rate, batch size, rollout count)
8. THE System SHALL save dataset versions and sizes
9. THE System SHALL save evaluation dataset results separately
10. THE System SHALL save git commit hash if repository is git-tracked

## Parser and Serializer Requirements

### Requirement 36: Parse Generated Model Outputs

**User Story:** As a researcher, I want to parse model-generated reasoning and answers, so that I can extract structured information for reward computation.

#### Acceptance Criteria

1. WHEN a model output is provided, THE Parser SHALL extract text within `<reasoning>...</reasoning>` tags
2. WHEN a model output is provided, THE Parser SHALL extract text within `<answer>...</answer>` tags
3. WHEN a model output is provided, THE Parser SHALL extract content within `\boxed{...}` in the answer section
4. WHEN XML tags are malformed or missing, THE Parser SHALL return None for that component
5. WHEN nested tags are present, THE Parser SHALL extract outermost tag content
6. THE Parser SHALL handle multi-line content within tags
7. THE Parser SHALL strip leading/trailing whitespace from extracted content

### Requirement 37: Pretty Print Model Outputs

**User Story:** As a researcher, I want to format model outputs consistently, so that I can generate valid examples for debugging and analysis.

#### Acceptance Criteria

1. THE Pretty_Printer SHALL format reasoning text within `<reasoning>` tags
2. THE Pretty_Printer SHALL format answer text within `<answer>` tags
3. THE Pretty_Printer SHALL format numeric answer within `\boxed{...}` in answer section
4. THE Pretty_Printer SHALL insert newlines for readability between tags
5. THE Pretty_Printer SHALL escape special characters within tag content
6. THE Pretty_Printer SHALL produce output parseable by Parser

### Requirement 38: Round-Trip Property for Output Parsing

**User Story:** As a researcher, I want parsing and printing to be inverse operations, so that I can trust format validation correctness.

#### Acceptance Criteria

1. FOR ALL valid output structures, parsing then printing then parsing SHALL produce equivalent extracted components
2. THE System SHALL test round-trip property on sample outputs during initialization
3. WHEN round-trip property fails, THE System SHALL log failure and example that failed
4. THE System SHALL include round-trip property test as part of smoke tier validation

---

## Hierarchical Composite Gate Requirements (N4 Contribution)

### Requirement 39: Hierarchical Composite Gate (H-CB-GRPO)

**User Story:** As a researcher, I want a composite gate combining cluster-level and prompt-level signals, so that I can test whether macro and micro interventions address distinct components of boundary shrinkage (N4).

#### Acceptance Criteria

1. THE System SHALL implement `CompositeGate` conforming to the existing `Gate` interface, accepting two child `Gate` instances.
2. THE System SHALL compute `macro_weight` using cluster-level EMA spend restricted to positive-advantage mass (`max(advantage, 0)`), replacing the `|advantage|` accumulator used by the original `CBGRPOGate`.
3. THE System SHALL compute `micro_weight` by reusing the `greedy_solve_ema` state already implemented for `OSELFGate`.
4. THE System SHALL combine weights as `gate_weight = macro_weight * micro_weight`.
5. THE System SHALL support a configuration flag selecting the combination operator (`multiplicative`, `min`, `harmonic_mean`) for ablation purposes.
6. THE System SHALL log `macro_weight`, `micro_weight`, and `gate_weight` separately per step for diagnostic purposes.
7. THE System SHALL train the 1.5B model with `CompositeGate` for the standard tier, alongside the existing Vanilla/CB-GRPO/O-SELF conditions (Requirement 22/23).
8. THE System SHALL validate that `0.5B + CompositeGate` is rejected at configuration time, consistent with the existing incompatibility rule for `OSELFGate` at 0.5B (Requirement 33.4).

---

## Mechanism-Level Diagnostics Requirements (N5 Contribution)

### Requirement 40: Interference and Diversity Diagnostics

**User Story:** As a researcher, I want to compute the interference metrics from the base SELF paper on my own checkpoints, so that I can causally connect cluster-spend imbalance to shrinkage rather than just correlating them (N5).

#### Acceptance Criteria

1. THE System SHALL construct a probing dataset by sampling G=4 base-model responses per training prompt prior to training (reuses existing rollout-generation code; one-time cost).
2. THE System SHALL compute, at each saved checkpoint, the interference metric `Δ⁺(π_θ, π_b) = E[Δ log π_θ(y⁺|x)]` over the probing set, where `y⁺` are base-model-correct completions.
3. THE System SHALL compute the relative influence magnitude `‖Δ(π_θ, π_b)‖ = E[(Δ log π_θ(y|x))²]` over the same set.
4. THE System SHALL compute mean token-level entropy of the sampling distribution during rollout generation at each logged step (near-zero marginal cost — reuses logits already produced during generation).
5. THE System SHALL report `Δ⁺`, `‖Δ‖`, and entropy trends alongside Pass@k and shrinkage-slope curves for every gate condition (Vanilla, CB-GRPO, O-SELF, H-CB-GRPO).
6. THE System SHALL compute the Pearson/Spearman correlation between per-condition cluster-spend Gini coefficient (already required, Requirement 24.4) and `Δ⁺`/`‖Δ‖`, in addition to the existing Gini–shrinkage-slope comparison.
7. THE System SHALL report this correlation as descriptive/exploratory (given the small number of conditions, ~8–12 data points) rather than as a hypothesis test with a p-value, and SHALL state this limitation explicitly in the summary report.

---

## Measurement Rigor Requirements

### Requirement 41: Evaluation Range and Ceiling-Effect Mitigation

**User Story:** As a researcher, I want to detect and mitigate ceiling effects in Pass@k evaluation, so that shrinkage-slope measurements are meaningful and not floor/ceiling-limited.

#### Acceptance Criteria

1. THE System SHALL report Pass@1 and Pass@10 for the base model on GSM8K test, GSM8K-platinum, and MATH-500 *before* running any RL experiment, as a saturation check.
2. WHEN base-model Pass@10 on a dataset exceeds 90%, THE System SHALL flag that dataset as ceiling-limited for shrinkage-slope interpretation and SHALL treat MATH-500 (or the least-saturated dataset) as the primary evidence source for shrinkage claims, with GSM8K/platinum reported as secondary/robustness checks.
3. THE System SHALL increase `n` (solutions per problem) for Pass@k evaluation from 10 to the largest value tractable within the remaining Colab budget (target: n≥30 if feasible), to extend the measurable k range beyond k=10, since shrinkage in prior work is most visible at larger k.
4. THE System SHALL report, in the summary report, the maximum k at which Pass@k can be estimated without exceeding the standard error threshold implied by n (i.e., avoid reporting Pass@10 estimates that are effectively saturated at both base and RL model).

### Requirement 42: Multi-Seed Protocol and Significance Testing

**User Story:** As a researcher, I want multi-seed experiments with proper significance testing, so that I can distinguish real effects from noise in RLVR training results.

#### Acceptance Criteria

1. THE System SHALL run the core comparison (Vanilla vs. CB-GRPO vs. H-CB-GRPO, 1.5B, standard tier) with a minimum of 3 random seeds per condition.
2. WHEN Colab time budget is constrained, THE System SHALL prioritize seed count over total training steps for the core comparison (e.g., prefer 3 seeds × standard tier over 1 seed × final tier).
3. THE System SHALL extend the existing bootstrap confidence-interval machinery (design.md Algorithm 3) to compute a bootstrap confidence interval on `Δslope = slope_condition − slope_vanilla` across seeds, not just within a single run's k-values.
4. THE System SHALL report whether the 95% CI on `Δslope` excludes zero, and SHALL avoid describing a result as "improvement" if the CI includes zero.
5. THE Final tier (1500–2000 steps, single seed) SHALL be reserved for at most one confirmatory run per headline condition, run only after the multi-seed standard-tier comparison identifies which conditions are worth confirming.

### Requirement 43: Clustering Basis Ablation

**User Story:** As a researcher, I want to test whether CB-GRPO's cluster definition (semantic topic vs. difficulty) changes its effectiveness, so that I can determine whether the "topic" framing or a "difficulty" framing better matches the shrinkage mechanism.

#### Acceptance Criteria

1. THE System SHALL compute an additional per-prompt feature: base-model solve-rate (already required for `StaticSELFGate`, Requirement 9.1 — reuse, don't recompute).
2. THE System SHALL cluster prompts using KMeans on `[sentence-embedding, solve-rate]` (concatenated, solve-rate scaled) as an alternative to the existing embedding-only clustering, producing `difficulty-aware clusters`.
3. THE System SHALL re-run CB-GRPO (macro-only) with difficulty-aware clusters at 1.5B, standard tier, and compare shrinkage slope and Δslope-CI against the original semantic-only clustering.
4. THE System SHALL report this as a single, clearly labeled ablation table (not as a new headline method), since its purpose is diagnostic (does clustering basis matter) rather than a new algorithmic claim.

