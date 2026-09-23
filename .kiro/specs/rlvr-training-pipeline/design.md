# RLVR Training Pipeline Design

## Overview

The RLVR Training Pipeline is a research notebook system designed to investigate the **reasoning boundary paradox** in small language models (SLMs) through reinforcement learning from verifiable rewards (RLVR). The system enables controlled experiments comparing different gradient gating strategies to understand how RL fine-tuning affects model capability boundaries.

### Research Objectives

1. **N1: Boundary Shrinkage Detection** - Measure whether RL fine-tuning causes capability deterioration at higher Pass@k values
2. **N2: CB-GRPO Mitigation** - Evaluate whether cluster-balanced gradient gating reduces boundary shrinkage (tested as baselines, not competition)
3. **N3: Capacity-Shrinkage Relationship** - Determine if shrinkage severity correlates with trainable parameter count (must cite arXiv:2604.06298)
4. **N4: Hierarchical Gating (Headline Contribution)** - Does topic-level (macro) and prompt-level (micro) gating address *distinct* components of reasoning-boundary shrinkage, such that combining them is super-additive relative to either alone?
5. **N5: Interference Diagnostics** - Does the SELF paper's interference metric (Δ⁺, ‖Δ‖) mediate the relationship between cluster-spend imbalance (Gini) and shrinkage slope?

### Design Philosophy

This design prioritizes:
- **Colab-first optimization**: All components fit within T4 GPU constraints (16GB VRAM)
- **Robust checkpointing**: Automatic resume after Colab session disconnects
- **Reproducibility**: Complete state preservation for scientific rigor
- **Modularity**: Clean separation between training, evaluation, and gating logic
- **Correctness**: Property-based testing for parser/printer components
- **Measurement rigor**: Ceiling-effect detection, multi-seed protocols, significance testing

### System Scope

The pipeline includes:
- Model loading with 4-bit quantization and QLoRA
- Dataset processing for GSM8K math reasoning tasks
- Multiple gradient gating strategies (Vanilla, CB-GRPO, O-SELF, Static-SELF, Adaptive-Rollout, **H-CB-GRPO CompositeGate**)
- Dual reward system (format validation + correctness checking)
- GRPO training loop with TRL integration
- **Interference diagnostics system** (Δ⁺, ‖Δ‖, entropy tracking)
- **Probing dataset construction** for checkpoint-level analysis
- Pass@k evaluation with shrinkage slope computation
- **Ceiling-effect detection and mitigation** (dataset saturation checks)
- Visualization and results export
- **Multi-seed protocol** with significance testing

## Architecture

### High-Level System Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION PHASE                             │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Load Config → 2. Load Model (QLoRA) → 3. Load Dataset (GSM8K)      │
│  4. Cluster Prompts → 5. Initialize Gate → 6. Check for Checkpoint     │
│  7. [NEW] Ceiling Check: Verify base Pass@k saturation                  │
│  8. [NEW] Build Probing Dataset (G=4 base responses per prompt)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  1. Sample Batch (prompts + cluster IDs)                 │           │
│  └──────────────────────────────────────────────────────────┘           │
│                           ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  2. Generate Rollouts (G=4 solutions per prompt)         │           │
│  │     [NEW] Track token-level entropy during generation    │           │
│  └──────────────────────────────────────────────────────────┘           │
│                           ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  3. Compute Rewards (format + correctness)               │           │
│  └──────────────────────────────────────────────────────────┘           │
│                           ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  4. Compute GRPO Advantages                              │           │
│  └──────────────────────────────────────────────────────────┘           │
│                           ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  5. Apply Gate (Vanilla/CB-GRPO/O-SELF/H-CB-GRPO)       │           │
│  │     [NEW] H-CB-GRPO: macro_weight × micro_weight         │           │
│  └──────────────────────────────────────────────────────────┘           │
│                           ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  6. Compute Loss & Update Model                          │           │
│  └──────────────────────────────────────────────────────────┘           │
│                           ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  7. Log Metrics & Checkpoint                             │           │
│  │     [NEW] Compute interference Δ⁺, ‖Δ‖ at checkpoints    │           │
│  └──────────────────────────────────────────────────────────┘           │
│                                                                          │
│  Loop until training_steps reached or user interrupts                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PHASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  1. [NEW] Check ceiling effect: base Pass@10 per dataset                │
│  2. Load Test Dataset → 3. Generate n≥30 Solutions per Problem          │
│  4. Evaluate Correctness → 5. Compute Pass@k (k=1,2,3,5,10,max_k)      │
│  6. Compute Shrinkage Slope → 7. Generate Transition Matrix            │
│  8. [NEW] Compute interference metrics correlation analysis              │
│  9. Create Visualizations → 10. Export Results                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                            SYSTEM LAYERS                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    Configuration Layer                           │  │
│  │  - ExperimentConfig (training params, gate params, paths)        │  │
│  │  - Compute tier management (smoke/standard/final)                │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Model Layer                                  │  │
│  │  - Model loading with 4-bit quantization                         │  │
│  │  - QLoRA adapter application                                     │  │
│  │  - Rollout generation (sequential for VRAM safety)               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Data Pipeline Layer                          │  │
│  │  - Dataset loading (GSM8K)                                       │  │
│  │  - Prompt clustering (for CB-GRPO)                               │  │
│  │  - Batch sampling with cluster IDs                               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Gate Layer                                   │  │
│  │  - Abstract Gate interface                                       │  │
│  │  - VanillaGate (no filtering)                                    │  │
│  │  - CBGRPOGate (cluster-balanced gating)                          │  │
│  │  - OSELFGate, StaticSELFGate, AdaptiveRolloutGate               │  │
│  │  - CompositeGate (H-CB-GRPO: macro × micro combination)          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Reward Layer                                 │  │
│  │  - FormatRewardModel (XML tag validation)                        │  │
│  │  - CorrectnessRewardModel (answer checking)                      │  │
│  │  - RewardAggregator (combines format + correctness)              │  │
│  │  - Parser & Pretty-Printer (with correctness properties)         │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Training Layer                               │  │
│  │  - GRPOTrainingLoop (TRL integration)                            │  │
│  │  - RolloutGenerator                                              │  │
│  │  - Metrics tracking                                              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Checkpoint Layer                             │  │
│  │  - CheckpointManager (save/load/verify)                          │  │
│  │  - ResumeManager (handle Colab disconnects)                      │  │
│  │  - Integrity verification (SHA256)                               │  │
│  │  - Probing dataset persistence (for interference metrics)        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                Interference Diagnostics Layer [NEW]              │  │
│  │  - ProbingDatasetBuilder (G=4 base responses per prompt)         │  │
│  │  - InterferenceMetricComputer (Δ⁺, ‖Δ‖)                          │  │
│  │  - EntropyTracker (token-level entropy during generation)        │  │
│  │  - CorrelationAnalyzer (Gini vs. interference metrics)           │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Evaluation Layer                             │  │
│  │  - PassAtKEvaluator (unbiased estimator)                         │  │
│  │  - ShrinkageSlopeComputer (log-linear regression)                │  │
│  │  - TransitionMatrixComputer (capability tracking)                │  │
│  │  - CeilingDetector (saturation check, dataset flagging)          │  │
│  │  - MultiSeedAnalyzer (Δslope CI across seeds)                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Visualization Layer                          │  │
│  │  - TrainingVisualizer (loss, reward, success rates)              │  │
│  │  - CBGRPOVisualizer (cluster spend analysis)                     │  │
│  │  - ResultsManager (export to JSON, generate reports)             │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Strategy Pattern**: Gate implementations share common interface, enabling plug-and-play comparison
2. **Template Method**: Training loop structure is fixed, but gate application is customizable
3. **Observer Pattern**: Metrics tracking observes training progress without coupling to training logic
4. **Memento Pattern**: Checkpoint system captures complete training state for restoration
5. **Facade Pattern**: RewardAggregator provides simple interface to complex reward computation

### Data Flow: Training Step

```
Batch (prompts, cluster_ids)
         ↓
    [Rollout Generator]
         ↓
Solutions (4 per prompt)
         ↓
    [Reward Aggregator]
         ↓
Rewards (format + correctness)
         ↓
    [GRPO Trainer]
         ↓
Advantages (policy gradient estimates)
         ↓
    [Gate]
         ↓
Weighted Advantages (gated by cluster spend)
         ↓
    [Loss Computation]
         ↓
Gradients → Model Update
```

### Memory Management Strategy

**4-bit Quantization**:
- Model weights stored in NF4 format (~4x size reduction)
- Dynamic dequantization during forward pass

**QLoRA**:
- Only LoRA adapters trainable (~2-8M parameters)
- Base model frozen in 4-bit format

**Sequential Generation**:
- Generate one rollout at a time (not batched)
- Prevents VRAM spikes from parallel generation

**Gradient Checkpointing**:
- Recompute activations during backward pass
- Trades computation for memory

**CPU Offloading**:
- Reward computation on CPU (regex-based)
- Cluster assignments stored in RAM

## Components and Interfaces

### Training Loop

#### GRPO Training Integration

```python
class GRPOTrainingLoop:
    """
    Main training loop integrating TRL's GRPOTrainer with custom gating.
    
    Architecture:
    1. Sample batch from dataset
    2. Generate G rollouts per prompt
    3. Compute rewards (format + correctness)
    4. Apply gate to get sampling weights
    5. Compute GRPO advantages
    6. Gate advantages before loss computation
    7. Perform gradient update
    8. Log metrics and checkpoint
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        cluster_assignments: dict[str, int],
        gate: Gate,
        reward_aggregator: RewardAggregator,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.cluster_assignments = cluster_assignments
        self.gate = gate
        self.reward_aggregator = reward_aggregator
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0.0
        self.best_pass_at_1 = 0.0
        
        # Metrics tracking
        self.metrics_history = {
            "step": [],
            "loss": [],
            "mean_reward": [],
            "format_success_rate": [],
            "correctness_success_rate": [],
        }
        
        # Initialize GRPO trainer
        self.grpo_trainer = self._initialize_grpo_trainer()
    
    def _initialize_grpo_trainer(self) -> GRPOTrainer:
        """Initializes TRL's GRPOTrainer with custom configuration."""
        training_args = GRPOConfig(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=1,  # We control steps manually
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=self.config.log_interval,
            save_steps=self.config.checkpoint_interval,
            eval_steps=self.config.eval_interval,
            save_total_limit=3,
            bf16=True,  # Use bfloat16 for memory efficiency
            optim="paged_adamw_8bit",  # 8-bit paged optimizer
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            # GRPO-specific settings
            num_train_epochs=1,
            group_size=self.config.rollouts_per_prompt,  # G=4
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
        )
        
        trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
        )
        
        # Hook into advantage computation for gate application
        self._inject_gate_hook(trainer)
        
        return trainer
    
    def _inject_gate_hook(self, trainer: GRPOTrainer):
        """
        Injects gate into GRPO trainer's advantage computation.
        
        Strategy: Override the advantage computation method to multiply
        advantages by gate weights before loss computation.
        """
        original_compute_loss = trainer.compute_loss
        
        def gated_compute_loss(model, inputs, return_outputs=False):
            # Extract batch information for gate
            prompt_ids = inputs["prompt_ids"]
            cluster_ids = [self.cluster_assignments[pid] for pid in prompt_ids]
            
            # Compute original advantages
            loss, outputs = original_compute_loss(model, inputs, return_outputs=True)
            advantages = outputs.advantages
            
            # Create RolloutBatch for gate
            batch = RolloutBatch(
                prompt_ids=prompt_ids,
                prompts=inputs["prompts"],
                cluster_ids=cluster_ids,
                rollouts=inputs["rollouts"],
                rewards=inputs["rewards"],
                advantages=advantages,
            )
            
            # Apply gate
            gate_weights = self.gate.weight(
                batch, TrainerState(
                    step=self.current_step,
                    epoch=self.current_epoch,
                    total_steps=self.config.training_steps,
                )
            )
            
            # Gate advantages
            gated_advantages = advantages * gate_weights.to(advantages.device).unsqueeze(-1)
            outputs.advantages = gated_advantages
            
            # Recompute loss with gated advantages
            gated_loss = trainer._compute_loss_from_advantages(gated_advantages, outputs)
            
            if return_outputs:
                return gated_loss, outputs
            return gated_loss
        
        trainer.compute_loss = gated_compute_loss
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Resume if checkpoint provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        print(f"Starting training for {self.config.training_steps} steps")
        print(f"Gate: {self.gate}")
        print(f"Reward mode: {self.config.reward_mode}")
        
        # Training loop
        while self.current_step < self.config.training_steps:
            # Single training step
            outputs = self.grpo_trainer.training_step(self.model, None)
            loss = outputs.loss.item()
            
            # Extract metrics
            mean_reward = outputs.rewards.mean().item()
            format_success = (outputs.format_rewards > 0).float().mean().item()
            correctness_success = (outputs.correctness_rewards > 0).float().mean().item()
            
            # Update metrics
            self.current_step += 1
            self.current_epoch = self.current_step / len(self.train_dataset) * self.config.batch_size
            
            self.metrics_history["step"].append(self.current_step)
            self.metrics_history["loss"].append(loss)
            self.metrics_history["mean_reward"].append(mean_reward)
            self.metrics_history["format_success_rate"].append(format_success)
            self.metrics_history["correctness_success_rate"].append(correctness_success)
            
            # Logging
            if self.current_step % self.config.log_interval == 0:
                self._log_metrics(loss, mean_reward, format_success, correctness_success)
            
            # Checkpoint saving
            if self.current_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint("last")
            
            # Evaluation
            if self.current_step % self.config.eval_interval == 0:
                self._run_evaluation()
        
        print("Training complete!")
        self.save_checkpoint("final")
    
    def _log_metrics(
        self, loss: float, mean_reward: float, format_success: float, correctness_success: float
    ):
        """Logs training metrics."""
        print(
            f"Step {self.current_step}/{self.config.training_steps} | "
            f"Loss: {loss:.4f} | "
            f"Reward: {mean_reward:.2f} | "
            f"Format: {format_success:.2%} | "
            f"Correct: {correctness_success:.2%}"
        )
        
        # Gate-specific logging
        if isinstance(self.gate, CBGRPOGate):
            spend_dist = self.gate.get_spend_distribution()
            mean_spend = np.mean(list(spend_dist.values()))
            std_spend = np.std(list(spend_dist.values()))
            print(f"  CB-GRPO spend: mean={mean_spend:.2f}, std={std_spend:.2f}")
        
        elif isinstance(self.gate, OSELFGate):
            filtered_count = self.gate.get_filtered_count()
            print(f"  O-SELF filtered: {filtered_count} prompts")
        
        elif isinstance(self.gate, AdaptiveRolloutGate):
            mean_variance = self.gate.get_mean_variance()
            print(f"  Adaptive mean variance: {mean_variance:.4f}")
    
    def _run_evaluation(self):
        """Runs Pass@k evaluation on test sets."""
        print(f"\nRunning evaluation at step {self.current_step}...")
        # Evaluation logic implemented in Evaluator class (see Section 8)
        pass
    
    def save_checkpoint(self, checkpoint_name: str):
        """Saves training checkpoint."""
        # Checkpoint logic implemented in CheckpointManager (see Section 7)
        pass
    
    def load_checkpoint(self, checkpoint_path: str):
        """Loads training checkpoint."""
        # Checkpoint logic implemented in CheckpointManager (see Section 7)
        pass
```

#### Rollout Generation

```python
class RolloutGenerator:
    """
    Generates multiple solution rollouts for GRPO training.
    
    Strategy: Sequential generation to minimize VRAM usage.
    Generate one rollout at a time instead of batching all G rollouts.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ExperimentConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate_rollouts(
        self, prompts: list[str], num_rollouts: int = 4
    ) -> list[list[str]]:
        """
        Generates multiple rollouts per prompt.
        
        Args:
            prompts: List of prompt strings
            num_rollouts: Number of rollouts per prompt (G)
        
        Returns:
            List of lists: outer list has len(prompts), inner lists have num_rollouts
        """
        all_rollouts = []
        
        for prompt in prompts:
            prompt_rollouts = []
            
            # Generate rollouts sequentially for memory efficiency
            for _ in range(num_rollouts):
                generated_text = self._generate_single(prompt)
                prompt_rollouts.append(generated_text)
            
            all_rollouts.append(prompt_rollouts)
        
        return all_rollouts
    
    def _generate_single(self, prompt: str) -> str:
        """Generates a single solution for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the generated part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):]
        
        return generated_text
```

### Checkpoint System

#### Checkpoint Schema

```python
@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    
    # Training progress
    step: int
    epoch: float
    total_steps: int
    
    # Model and optimizer
    model_state_dict: dict
    optimizer_state_dict: dict
    lr_scheduler_state_dict: dict
    
    # Gate state
    gate_type: str
    gate_state_dict: dict
    
    # Best metrics
    best_pass_at_1: float
    best_checkpoint_step: int
    
    # Random state for reproducibility
    torch_rng_state: torch.Tensor
    numpy_rng_state: dict
    python_rng_state: tuple
    
    # Configuration
    config: ExperimentConfig
    
    # Metrics history
    metrics_history: dict
    
    # Metadata
    timestamp: str
    hostname: str
    gpu_name: str
```

#### Checkpoint Manager

```python
import hashlib
from datetime import datetime

class CheckpointManager:
    """
    Manages checkpoint saving, loading, and verification.
    
    Checkpoint types:
    - last.pt: Most recent checkpoint (for resume)
    - best.pt: Best checkpoint by Pass@1 (for final results)
    - step_{N}.pt: Incremental checkpoints every N steps
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.exp_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        training_state: TrainingState,
        checkpoint_type: str = "last",
    ) -> str:
        """
        Saves checkpoint to disk.
        
        Args:
            training_state: Complete training state
            checkpoint_type: "last", "best", or "step"
        
        Returns:
            Path to saved checkpoint
        """
        # Determine checkpoint filename
        if checkpoint_type == "last":
            filename = "last.pt"
        elif checkpoint_type == "best":
            filename = "best.pt"
        elif checkpoint_type == "step":
            filename = f"step_{training_state.step}.pt"
        else:
            raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint dictionary
        checkpoint = asdict(training_state)
        
        # Add metadata
        checkpoint["save_timestamp"] = datetime.now().isoformat()
        checkpoint["checkpoint_version"] = "1.0"
        
        # Save to disk
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        
        # Compute and save SHA256 hash for integrity verification
        checkpoint_hash = self._compute_file_hash(checkpoint_path)
        hash_path = checkpoint_path.with_suffix(".pt.sha256")
        with open(hash_path, "w") as f:
            f.write(checkpoint_hash)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> TrainingState:
        """
        Loads checkpoint from disk with integrity verification.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            TrainingState object
        
        Raises:
            RuntimeError: If checkpoint is corrupted or invalid
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Verify integrity if hash file exists
        hash_path = checkpoint_path.with_suffix(".pt.sha256")
        if hash_path.exists():
            with open(hash_path, "r") as f:
                expected_hash = f.read().strip()
            
            actual_hash = self._compute_file_hash(checkpoint_path)
            
            if expected_hash != actual_hash:
                raise RuntimeError(
                    f"Checkpoint integrity check failed for {checkpoint_path}. "
                    f"Expected hash: {expected_hash}, actual: {actual_hash}. "
                    f"File may be corrupted."
                )
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Convert back to TrainingState
        # (This requires manual reconstruction since we can't directly deserialize dataclass)
        training_state = TrainingState(
            step=checkpoint["step"],
            epoch=checkpoint["epoch"],
            total_steps=checkpoint["total_steps"],
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            lr_scheduler_state_dict=checkpoint["lr_scheduler_state_dict"],
            gate_type=checkpoint["gate_type"],
            gate_state_dict=checkpoint["gate_state_dict"],
            best_pass_at_1=checkpoint["best_pass_at_1"],
            best_checkpoint_step=checkpoint["best_checkpoint_step"],
            torch_rng_state=checkpoint["torch_rng_state"],
            numpy_rng_state=checkpoint["numpy_rng_state"],
            python_rng_state=checkpoint["python_rng_state"],
            config=ExperimentConfig(**checkpoint["config"]),
            metrics_history=checkpoint["metrics_history"],
            timestamp=checkpoint["timestamp"],
            hostname=checkpoint["hostname"],
            gpu_name=checkpoint["gpu_name"],
        )
        
        return training_state
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Finds most recent checkpoint in checkpoint directory.
        
        Returns:
            Path to last.pt if it exists, otherwise None
        """
        last_checkpoint = self.checkpoint_dir / "last.pt"
        if last_checkpoint.exists():
            return str(last_checkpoint)
        
        # Fallback: find most recent step_N.pt
        step_checkpoints = list(self.checkpoint_dir.glob("step_*.pt"))
        if step_checkpoints:
            # Sort by step number
            step_checkpoints.sort(
                key=lambda p: int(p.stem.split("_")[1])
            )
            return str(step_checkpoints[-1])
        
        return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Removes old incremental checkpoints, keeping only the N most recent.
        
        Preserves last.pt and best.pt.
        """
        step_checkpoints = list(self.checkpoint_dir.glob("step_*.pt"))
        if len(step_checkpoints) <= keep_last_n:
            return
        
        # Sort by step number
        step_checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        
        # Remove oldest checkpoints
        to_remove = step_checkpoints[:-keep_last_n]
        for checkpoint_path in to_remove:
            checkpoint_path.unlink()
            hash_path = checkpoint_path.with_suffix(".pt.sha256")
            if hash_path.exists():
                hash_path.unlink()
        
        print(f"Cleaned up {len(to_remove)} old checkpoints")
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Computes SHA256 hash of file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def display_checkpoint_info(self, checkpoint_path: str):
        """Displays checkpoint information for user confirmation."""
        training_state = self.load_checkpoint(checkpoint_path)
        
        print("\n" + "="*60)
        print("CHECKPOINT INFORMATION")
        print("="*60)
        print(f"Experiment: {training_state.config.exp_name}")
        print(f"Step: {training_state.step}/{training_state.total_steps}")
        print(f"Epoch: {training_state.epoch:.2f}")
        print(f"Best Pass@1: {training_state.best_pass_at_1:.4f} (step {training_state.best_checkpoint_step})")
        print(f"Gate: {training_state.gate_type}")
        print(f"Saved: {training_state.timestamp}")
        print("="*60 + "\n")
```

#### Resume Protocol

```python
class ResumeManager:
    """Manages training resumption after Colab disconnect."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    def prompt_resume(self) -> tuple[bool, Optional[str]]:
        """
        Prompts user to resume from latest checkpoint if available.
        
        Returns:
            (should_resume, checkpoint_path) tuple
        """
        latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
        
        if latest_checkpoint is None:
            print("No existing checkpoint found. Starting fresh.")
            return False, None
        
        # Display checkpoint information
        self.checkpoint_manager.display_checkpoint_info(latest_checkpoint)
        
        # Prompt user
        response = input("Resume from this checkpoint? [y/n]: ").strip().lower()
        
        if response == "y":
            return True, latest_checkpoint
        else:
            print("Starting fresh (existing checkpoint will not be overwritten)")
            return False, None
    
    def restore_training_state(
        self,
        checkpoint_path: str,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        gate: Gate,
    ) -> tuple[int, float, dict]:
        """
        Restores complete training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to restore state into
            optimizer: Optimizer to restore state into
            lr_scheduler: LR scheduler to restore state into
            gate: Gate to restore state into
        
        Returns:
            (step, epoch, metrics_history) tuple
        """
        training_state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Restore model state
        model.load_state_dict(training_state.model_state_dict)
        print("Model state restored")
        
        # Restore optimizer state
        optimizer.load_state_dict(training_state.optimizer_state_dict)
        print("Optimizer state restored")
        
        # Restore LR scheduler state
        lr_scheduler.load_state_dict(training_state.lr_scheduler_state_dict)
        print("LR scheduler state restored")
        
        # Restore gate state
        gate.load_state_dict(training_state.gate_state_dict)
        print(f"Gate state restored: {gate}")
        
        # Restore RNG states for reproducibility
        torch.set_rng_state(training_state.torch_rng_state)
        np.random.set_state(training_state.numpy_rng_state)
        random.setstate(training_state.python_rng_state)
        print("RNG states restored")
        
        return (
            training_state.step,
            training_state.epoch,
            training_state.metrics_history,
        )
```

### Evaluation System

#### Pass@k Evaluator

```python
from math import comb
from typing import Callable

class PassAtKEvaluator:
    """
    Evaluates Pass@k using the unbiased estimator.
    
    Pass@k = probability that at least one of k generated solutions is correct.
    
    Unbiased estimator formula:
    Pass@k = 1 - C(n-c, k) / C(n, k)
    
    where:
    - n = total samples generated per problem
    - c = number of correct samples
    - k = number of samples to consider
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ExperimentConfig,
        reward_aggregator: RewardAggregator,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_aggregator = reward_aggregator
        self.rollout_generator = RolloutGenerator(model, tokenizer, config)
    
    def evaluate_dataset(
        self,
        dataset: Dataset,
        dataset_name: str,
        num_samples: int = 10,
        k_values: list[int] = [1, 2, 3, 5, 10],
    ) -> PassAtKResult:
        """
        Evaluates Pass@k on a dataset.
        
        Args:
            dataset: Evaluation dataset with prompt_text and ground_truth_answer
            dataset_name: Name of dataset for logging
            num_samples: Number of solutions to generate per problem (n)
            k_values: List of k values to compute Pass@k for
        
        Returns:
            PassAtKResult with Pass@k values
        """
        print(f"Evaluating Pass@k on {dataset_name} ({len(dataset)} problems)...")
        
        # Generate solutions and evaluate correctness
        all_correctness = []
        
        for example in tqdm(dataset):
            prompt = example["prompt_text"]
            ground_truth = example["ground_truth_answer"]
            
            # Generate n solutions
            solutions = self.rollout_generator._generate_single(prompt)
            solutions = [solutions]  # Wrap in list
            
            # Generate remaining solutions
            for _ in range(num_samples - 1):
                solution = self.rollout_generator._generate_single(prompt)
                solutions.append(solution)
            
            # Evaluate correctness
            correctness = []
            for solution in solutions:
                rewards = self.reward_aggregator.compute_reward(solution, ground_truth)
                is_correct = rewards["correctness"] > 0.0
                correctness.append(is_correct)
            
            all_correctness.append(correctness)
        
        # Compute Pass@k for each k
        pass_at_k_values = []
        for k in k_values:
            pass_at_k = self._compute_pass_at_k(all_correctness, n=num_samples, k=k)
            pass_at_k_values.append(pass_at_k)
        
        # Create result object
        result = PassAtKResult(
            dataset_name=dataset_name,
            model_name=self.config.model_name,
            checkpoint_step=getattr(self, "current_step", 0),
            k_values=k_values,
            pass_at_k_values=pass_at_k_values,
            num_samples=num_samples,
            timestamp=datetime.now().isoformat(),
        )
        
        # Log results
        self._log_pass_at_k_results(result)
        
        return result
    
    @staticmethod
    def _compute_pass_at_k(
        all_correctness: list[list[bool]], n: int, k: int
    ) -> float:
        """
        Computes Pass@k using unbiased estimator.
        
        Formula: Pass@k = 1 - C(n-c, k) / C(n, k)
        
        where c = number of correct samples for each problem
        """
        if k > n:
            raise ValueError(f"k ({k}) cannot exceed n ({n})")
        
        total_pass = 0
        for correctness in all_correctness:
            c = sum(correctness)
            
            # If all k samples would be incorrect, Pass@k = 0
            if n - c < k:
                pass_prob = 1.0
            else:
                # Unbiased estimator
                pass_prob = 1.0 - comb(n - c, k) / comb(n, k)
            
            total_pass += pass_prob
        
        return total_pass / len(all_correctness)
    
    def _log_pass_at_k_results(self, result: PassAtKResult):
        """Logs Pass@k results in formatted table."""
        print(f"\n{result.dataset_name} Pass@k Results:")
        print("=" * 40)
        for k, pass_k in zip(result.k_values, result.pass_at_k_values):
            print(f"  Pass@{k:2d}: {pass_k:.4f} ({pass_k*100:.2f}%)")
        print("=" * 40 + "\n")


@dataclass
class PassAtKResult:
    """Results from Pass@k evaluation."""
    dataset_name: str
    model_name: str
    checkpoint_step: int
    k_values: list[int]
    pass_at_k_values: list[float]
    num_samples: int
    timestamp: str
    
    def to_dict(self) -> dict:
        """Converts to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "checkpoint_step": self.checkpoint_step,
            "pass_at_k": {
                f"k={k}": v for k, v in zip(self.k_values, self.pass_at_k_values)
            },
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
        }
```

#### Shrinkage Slope Computer

```python
class ShrinkageSlopeComputer:
    """
    Computes shrinkage slope from Pass@k curves.
    
    Shrinkage slope = linear regression slope of log(k) vs. Δ Pass@k
    
    where Δ Pass@k = Pass@k_RL - Pass@k_base
    
    Negative slope indicates boundary shrinkage (Pass@k deteriorates faster
    at higher k values).
    """
    
    @staticmethod
    def compute_slope(
        base_result: PassAtKResult,
        rl_result: PassAtKResult,
    ) -> tuple[float, float, float]:
        """
        Computes shrinkage slope from base and RL Pass@k results.
        
        Args:
            base_result: Pass@k results for base model
            rl_result: Pass@k results for RL-tuned model
        
        Returns:
            (slope, intercept, r_squared) tuple
        """
        # Ensure k_values match
        if base_result.k_values != rl_result.k_values:
            raise ValueError("k_values must match between base and RL results")
        
        k_values = base_result.k_values
        
        # Compute Δ Pass@k
        delta_pass_k = [
            rl_val - base_val
            for rl_val, base_val in zip(
                rl_result.pass_at_k_values, base_result.pass_at_k_values
            )
        ]
        
        # Log transform k
        log_k = np.log(k_values)
        
        # Linear regression
        slope, intercept = np.polyfit(log_k, delta_pass_k, 1)
        
        # Compute R^2
        predicted = slope * log_k + intercept
        ss_res = np.sum((delta_pass_k - predicted) ** 2)
        ss_tot = np.sum((delta_pass_k - np.mean(delta_pass_k)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return slope, intercept, r_squared
    
    @staticmethod
    def compute_confidence_interval(
        base_result: PassAtKResult,
        rl_result: PassAtKResult,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Computes confidence interval for shrinkage slope.
        
        Uses bootstrap resampling.
        
        Returns:
            (lower_bound, upper_bound) tuple
        """
        # Bootstrap parameters
        n_bootstrap = 1000
        slopes = []
        
        k_values = base_result.k_values
        log_k = np.log(k_values)
        
        # Compute Δ Pass@k
        delta_pass_k = np.array([
            rl_val - base_val
            for rl_val, base_val in zip(
                rl_result.pass_at_k_values, base_result.pass_at_k_values
            )
        ])
        
        # Bootstrap
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(len(k_values), size=len(k_values), replace=True)
            log_k_sample = log_k[indices]
            delta_sample = delta_pass_k[indices]
            
            # Fit slope
            slope, _ = np.polyfit(log_k_sample, delta_sample, 1)
            slopes.append(slope)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(slopes, lower_percentile)
        upper_bound = np.percentile(slopes, upper_percentile)
        
        return lower_bound, upper_bound
```

#### Transition Matrix Computer

```python
class TransitionMatrixComputer:
    """
    Computes per-problem capability transition matrix.
    
    Classifies each problem into one of four categories:
    1. kept_correct: Base model correct, RL model correct
    2. lost_capability: Base model correct, RL model incorrect
    3. gained_capability: Base model incorrect, RL model correct
    4. kept_incorrect: Base model incorrect, RL model incorrect
    """
    
    @staticmethod
    def compute_matrix(
        base_correctness: list[bool],
        rl_correctness: list[bool],
        problem_ids: list[str],
    ) -> dict:
        """
        Computes transition matrix.
        
        Args:
            base_correctness: List of correctness values for base model
            rl_correctness: List of correctness values for RL model
            problem_ids: List of problem IDs (for detailed tracking)
        
        Returns:
            Dictionary with counts, percentages, and problem ID lists
        """
        if len(base_correctness) != len(rl_correctness) != len(problem_ids):
            raise ValueError("All input lists must have same length")
        
        # Initialize categories
        kept_correct = []
        lost_capability = []
        gained_capability = []
        kept_incorrect = []
        
        # Classify each problem
        for i, (base, rl, pid) in enumerate(
            zip(base_correctness, rl_correctness, problem_ids)
        ):
            if base and rl:
                kept_correct.append(pid)
            elif base and not rl:
                lost_capability.append(pid)
            elif not base and rl:
                gained_capability.append(pid)
            else:  # not base and not rl
                kept_incorrect.append(pid)
        
        total = len(problem_ids)
        
        # Compute transition matrix
        matrix = {
            "kept_correct": {
                "count": len(kept_correct),
                "percentage": len(kept_correct) / total * 100,
                "problem_ids": kept_correct,
            },
            "lost_capability": {
                "count": len(lost_capability),
                "percentage": len(lost_capability) / total * 100,
                "problem_ids": lost_capability,
            },
            "gained_capability": {
                "count": len(gained_capability),
                "percentage": len(gained_capability) / total * 100,
                "problem_ids": gained_capability,
            },
            "kept_incorrect": {
                "count": len(kept_incorrect),
                "percentage": len(kept_incorrect) / total * 100,
                "problem_ids": kept_incorrect,
            },
            "total": total,
        }
        
        return matrix
    
    @staticmethod
    def display_matrix(matrix: dict):
        """Displays transition matrix in formatted table."""
        print("\nTransition Matrix:")
        print("=" * 60)
        print(f"  Kept Correct:      {matrix['kept_correct']['count']:4d} "
              f"({matrix['kept_correct']['percentage']:5.1f}%)")
        print(f"  Lost Capability:   {matrix['lost_capability']['count']:4d} "
              f"({matrix['lost_capability']['percentage']:5.1f}%)")
        print(f"  Gained Capability: {matrix['gained_capability']['count']:4d} "
              f"({matrix['gained_capability']['percentage']:5.1f}%)")
        print(f"  Kept Incorrect:    {matrix['kept_incorrect']['count']:4d} "
              f"({matrix['kept_incorrect']['percentage']:5.1f}%)")
        print("-" * 60)
        print(f"  Total:             {matrix['total']:4d} (100.0%)")
        print("=" * 60 + "\n")
```

Now I'll continue with the visualization system, algorithms, and complete the design document.

<function_calls>
<invoke name="report_progress">
<parameter name="message">Created comprehensive design sections for configuration, model management, dataset pipeline, gate architecture, reward system, training loop, checkpointing, and evaluation. Now completing visualization system and algorithms.

### Visualization System

#### Training Metrics Plotter

```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TrainingVisualizer:
    """
    Creates inline visualizations for training metrics.
    
    Uses matplotlib for static plots and plotly for interactive plots.
    """
    
    def __init__(self, metrics_history: dict):
        self.metrics_history = metrics_history
    
    def plot_training_curves(self, show_inline: bool = True):
        """
        Plots training loss, reward, format rate, and correctness rate.
        
        Creates a 2x2 grid of subplots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        steps = self.metrics_history["step"]
        
        # Loss curve
        axes[0, 0].plot(steps, self.metrics_history["loss"], linewidth=2)
        axes[0, 0].set_title("Training Loss", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean reward curve
        axes[0, 1].plot(steps, self.metrics_history["mean_reward"], linewidth=2, color="green")
        axes[0, 1].set_title("Mean Reward", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        
        # Format success rate
        axes[1, 0].plot(steps, self.metrics_history["format_success_rate"], linewidth=2, color="blue")
        axes[1, 0].set_title("Format Success Rate", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correctness success rate
        axes[1, 1].plot(steps, self.metrics_history["correctness_success_rate"], linewidth=2, color="orange")
        axes[1, 1].set_title("Correctness Success Rate", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Success Rate")
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show_inline:
            plt.show()
        
        return fig
    
    def plot_pass_at_k_comparison(
        self,
        base_result: PassAtKResult,
        rl_result: PassAtKResult,
        show_inline: bool = True,
    ):
        """
        Plots Pass@k curves comparing base model and RL-tuned model.
        """
        fig = go.Figure()
        
        # Base model curve
        fig.add_trace(go.Scatter(
            x=base_result.k_values,
            y=base_result.pass_at_k_values,
            mode="lines+markers",
            name="Base Model",
            line=dict(color="blue", width=3),
            marker=dict(size=8),
        ))
        
        # RL model curve
        fig.add_trace(go.Scatter(
            x=rl_result.k_values,
            y=rl_result.pass_at_k_values,
            mode="lines+markers",
            name="RL-tuned Model",
            line=dict(color="red", width=3),
            marker=dict(size=8),
        ))
        
        fig.update_layout(
            title=f"Pass@k Comparison - {base_result.dataset_name}",
            xaxis_title="k (number of samples)",
            yaxis_title="Pass@k",
            xaxis_type="log",
            yaxis_range=[0, 1.05],
            width=800,
            height=500,
            template="plotly_white",
            font=dict(size=12),
            hovermode="x unified",
        )
        
        if show_inline:
            fig.show()
        
        return fig
    
    def plot_shrinkage_slope_vs_capacity(
        self,
        model_sizes: list[str],
        trainable_params: list[int],
        shrinkage_slopes: list[float],
        confidence_intervals: list[tuple[float, float]],
        show_inline: bool = True,
    ):
        """
        Plots shrinkage slope vs. trainable parameter count.
        
        This is the key figure for N1 analysis.
        """
        fig = go.Figure()
        
        # Extract confidence interval bounds
        lower_bounds = [ci[0] for ci in confidence_intervals]
        upper_bounds = [ci[1] for ci in confidence_intervals]
        
        # Slope points with error bars
        fig.add_trace(go.Scatter(
            x=trainable_params,
            y=shrinkage_slopes,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[ub - s for ub, s in zip(upper_bounds, shrinkage_slopes)],
                arrayminus=[s - lb for lb, s in zip(lower_bounds, shrinkage_slopes)],
            ),
            mode="markers+lines",
            name="Shrinkage Slope",
            marker=dict(size=12, color="red"),
            line=dict(width=2, dash="dash"),
            text=model_sizes,
            hovertemplate="<b>%{text}</b><br>Params: %{x:,.0f}<br>Slope: %{y:.4f}<extra></extra>",
        ))
        
        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No shrinkage")
        
        fig.update_layout(
            title="Boundary Shrinkage Severity vs. Model Capacity",
            xaxis_title="Trainable Parameters (LoRA)",
            yaxis_title="Shrinkage Slope (Δ Pass@k / log(k))",
            xaxis_type="log",
            width=900,
            height=600,
            template="plotly_white",
            font=dict(size=12),
        )
        
        if show_inline:
            fig.show()
        
        return fig


class CBGRPOVisualizer:
    """Specialized visualizations for CB-GRPO gate."""
    
    @staticmethod
    def plot_cluster_spend_histogram(
        vanilla_spend: dict[int, float],
        cbgrpo_spend: dict[int, float],
        show_inline: bool = True,
    ):
        """
        Plots side-by-side histograms comparing cluster spend distributions.
        
        Shows how CB-GRPO flattens the spend distribution compared to
        vanilla GRPO (which is expected to be uniform).
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Vanilla GRPO", "CB-GRPO"),
            horizontal_spacing=0.12,
        )
        
        # Vanilla GRPO histogram
        cluster_ids_vanilla = sorted(vanilla_spend.keys())
        spend_values_vanilla = [vanilla_spend[cid] for cid in cluster_ids_vanilla]
        
        fig.add_trace(
            go.Bar(
                x=cluster_ids_vanilla,
                y=spend_values_vanilla,
                name="Vanilla GRPO",
                marker_color="lightblue",
            ),
            row=1, col=1
        )
        
        # CB-GRPO histogram
        cluster_ids_cbgrpo = sorted(cbgrpo_spend.keys())
        spend_values_cbgrpo = [cbgrpo_spend[cid] for cid in cluster_ids_cbgrpo]
        
        fig.add_trace(
            go.Bar(
                x=cluster_ids_cbgrpo,
                y=spend_values_cbgrpo,
                name="CB-GRPO",
                marker_color="lightcoral",
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Cluster ID", row=1, col=1)
        fig.update_xaxes(title_text="Cluster ID", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Gradient Mass", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Gradient Mass", row=1, col=2)
        
        fig.update_layout(
            title_text="Cluster Spend Distribution Comparison",
            width=1200,
            height=500,
            showlegend=False,
            template="plotly_white",
        )
        
        if show_inline:
            fig.show()
        
        return fig
    
    @staticmethod
    def plot_spend_time_series(
        spend_history: list[dict[int, float]],
        steps: list[int],
        selected_clusters: list[int] = None,
        show_inline: bool = True,
    ):
        """
        Plots gradient-mass accumulation over time for selected clusters.
        
        Shows which clusters consume capacity early vs. late in training.
        """
        if selected_clusters is None:
            # Select top 5 clusters by final spend
            final_spend = spend_history[-1]
            selected_clusters = sorted(
                final_spend.keys(),
                key=lambda cid: final_spend[cid],
                reverse=True
            )[:5]
        
        fig = go.Figure()
        
        for cluster_id in selected_clusters:
            spend_values = [
                spend_dict.get(cluster_id, 0.0) for spend_dict in spend_history
            ]
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=spend_values,
                mode="lines",
                name=f"Cluster {cluster_id}",
                line=dict(width=2),
            ))
        
        fig.update_layout(
            title="Gradient-Mass Accumulation Over Training",
            xaxis_title="Training Step",
            yaxis_title="Cumulative Gradient Mass",
            width=900,
            height=600,
            template="plotly_white",
            font=dict(size=12),
        )
        
        if show_inline:
            fig.show()
        
        return fig
```

### Results Organization and Export

#### Results Manager

```python
import json
from pathlib import Path

class ResultsManager:
    """
    Manages experiment results organization and export.
    
    Directory structure:
    results/
      {exp_name}/
        {model_size}/
          pass_at_k_step_{N}.json
          pass_at_k_final.json
        config.json
        shrinkage_slopes.json
        transition_matrices.json
        cluster_spend.json (CB-GRPO only)
        summary.md
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base_dir = Path("results") / config.exp_name / config.model_size
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_pass_at_k_result(self, result: PassAtKResult, checkpoint_step: int = None):
        """Saves Pass@k result to JSON."""
        if checkpoint_step is not None:
            filename = f"pass_at_k_step_{checkpoint_step}.json"
        else:
            filename = "pass_at_k_final.json"
        
        filepath = self.base_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"Saved Pass@k results to {filepath}")
    
    def save_shrinkage_slope(
        self,
        dataset_name: str,
        slope: float,
        confidence_interval: tuple[float, float],
        r_squared: float,
    ):
        """Saves shrinkage slope to JSON."""
        filepath = self.base_dir.parent / "shrinkage_slopes.json"
        
        # Load existing if present
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        # Update with new result
        data[self.config.model_size] = {
            "dataset": dataset_name,
            "slope": slope,
            "confidence_interval_95": {
                "lower": confidence_interval[0],
                "upper": confidence_interval[1],
            },
            "r_squared": r_squared,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved shrinkage slope to {filepath}")
    
    def save_transition_matrix(self, dataset_name: str, matrix: dict):
        """Saves transition matrix to JSON."""
        filepath = self.base_dir.parent / "transition_matrices.json"
        
        # Load existing if present
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        # Update with new result
        key = f"{self.config.model_size}_{dataset_name}"
        data[key] = matrix
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved transition matrix to {filepath}")
    
    def save_cluster_spend(self, spend_distribution: dict[int, float], step: int):
        """Saves CB-GRPO cluster spend distribution."""
        filepath = self.base_dir.parent / "cluster_spend.json"
        
        # Load existing if present
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        # Update with new result
        key = f"{self.config.model_size}_step_{step}"
        data[key] = {str(k): v for k, v in spend_distribution.items()}
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved cluster spend to {filepath}")
    
    def save_config(self):
        """Saves experiment configuration."""
        filepath = self.base_dir.parent / "config.json"
        
        config_dict = {
            "exp_name": self.config.exp_name,
            "model_size": self.config.model_size,
            "compute_tier": self.config.compute_tier,
            "gate_type": self.config.gate_type,
            "gate_params": self.config.gate_params,
            "reward_mode": self.config.reward_mode,
            "training_steps": self.config.training_steps,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "rollouts_per_prompt": self.config.rollouts_per_prompt,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_new_tokens": self.config.max_new_tokens,
            "random_seed": self.config.random_seed,
        }
        
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved config to {filepath}")
    
    def generate_summary_report(
        self,
        final_pass_at_k: PassAtKResult,
        shrinkage_slope: float,
        transition_matrix: dict,
        training_time_hours: float,
    ) -> str:
        """
        Generates summary markdown report.
        
        Returns:
            Path to generated summary.md
        """
        filepath = self.base_dir.parent / "summary.md"
        
        with open(filepath, "w") as f:
            f.write(f"# Experiment Summary: {self.config.exp_name}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Model**: Qwen2.5-{self.config.model_size}-Instruct\n")
            f.write(f"- **Gate**: {self.config.gate_type}\n")
            f.write(f"- **Compute Tier**: {self.config.compute_tier}\n")
            f.write(f"- **Training Steps**: {self.config.training_steps}\n")
            f.write(f"- **Training Time**: {training_time_hours:.2f} hours\n")
            f.write(f"- **Reward Mode**: {self.config.reward_mode}\n\n")
            
            f.write("## Key Results\n\n")
            f.write(f"### Pass@k ({final_pass_at_k.dataset_name})\n\n")
            f.write("| k | Pass@k | Percentage |\n")
            f.write("|---|--------|------------|\n")
            for k, val in zip(final_pass_at_k.k_values, final_pass_at_k.pass_at_k_values):
                f.write(f"| {k} | {val:.4f} | {val*100:.2f}% |\n")
            
            f.write(f"\n### Boundary Shrinkage\n\n")
            f.write(f"- **Shrinkage Slope**: {shrinkage_slope:.4f}\n")
            f.write(f"- Negative slope indicates boundary shrinkage severity\n\n")
            
            f.write("### Transition Matrix\n\n")
            f.write("| Category | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")
            f.write(f"| Kept Correct | {transition_matrix['kept_correct']['count']} | "
                   f"{transition_matrix['kept_correct']['percentage']:.1f}% |\n")
            f.write(f"| Lost Capability | {transition_matrix['lost_capability']['count']} | "
                   f"{transition_matrix['lost_capability']['percentage']:.1f}% |\n")
            f.write(f"| Gained Capability | {transition_matrix['gained_capability']['count']} | "
                   f"{transition_matrix['gained_capability']['percentage']:.1f}% |\n")
            f.write(f"| Kept Incorrect | {transition_matrix['kept_incorrect']['count']} | "
                   f"{transition_matrix['kept_incorrect']['percentage']:.1f}% |\n")
            
            f.write("\n## Interpretation\n\n")
            if shrinkage_slope < -0.05:
                f.write("⚠️ **Significant boundary shrinkage detected.** "
                       "The model's capability deteriorates rapidly as k increases, "
                       "indicating over-specialization on high-reward solutions.\n")
            elif shrinkage_slope < 0:
                f.write("⚠️ **Mild boundary shrinkage detected.** "
                       "Some capability loss at higher k values.\n")
            else:
                f.write("✓ **No boundary shrinkage.** "
                       "The model maintains broad capability across k values.\n")
        
        print(f"Generated summary report at {filepath}")
        return str(filepath)
```

## Algorithms

### Algorithm 1: CB-GRPO Gradient-Mass Gating

```
ALGORITHM: CB-GRPO Gradient-Mass Gating
INPUT: batch (prompts, cluster_ids, advantages), state (step, epoch)
OUTPUT: gate_weights (per-sample multipliers in [0, 1])

INITIALIZE:
  spend[c] ← 0 for all clusters c
  spend_ema[c] ← 0 for all clusters c
  theta ← 1.5  # over-budget threshold
  decay ← 0.98  # decay factor
  ema_alpha ← 0.05  # EMA smoothing

FOR each sample i in batch:
  cluster_id ← batch.cluster_ids[i]
  advantage ← batch.advantages[i]
  
  # 1. Compute gradient mass for this sample
  grad_mass ← |advantage|
  
  # 2. Update cluster spend
  spend[cluster_id] ← spend[cluster_id] + grad_mass
  spend_ema[cluster_id] ← ema_alpha * grad_mass + (1 - ema_alpha) * spend_ema[cluster_id]
  
  # 3. Compute spend ratio relative to mean
  mean_spend ← mean(spend_ema) + 1e-8
  ratio ← spend_ema[cluster_id] / mean_spend
  
  # 4. Apply soft gating
  IF ratio > theta:
    over_budget ← ratio - theta
    gate_weight[i] ← decay^over_budget  # exponential decay
  ELSE:
    gate_weight[i] ← 1.0
  END IF
END FOR

RETURN gate_weights
```

**Key Design Choices:**
- **Soft decay** instead of hard cutoff prevents permanent cluster exclusion
- **Cluster-level tracking** instead of prompt-level captures reasoning patterns
- **Cold-start safe** because ratio starts near 1.0 when all spend is zero
- **EMA smoothing** prevents noise from single-step spikes

### Algorithm 2: H-CB-GRPO Composite Gate (N4 Headline Contribution)

```
ALGORITHM: H-CB-GRPO Composite Gate
INPUT: batch (prompts, prompt_ids, cluster_ids, advantages), state (step, epoch)
OUTPUT: gate_weights (per-sample multipliers in [0, 1])

INITIALIZE:
  spend_pos_ema[c] ← 0 for all clusters c        # only positive-advantage mass
  greedy_solve_ema[p] ← 0 for all prompts p       # reuse from OSELFGate
  theta ← 1.5, decay ← 0.98, ema_alpha ← 0.05     # macro params (unchanged)
  tau_solve ← 0.7, lambda_self ← 0.3              # micro params (new)

FOR each sample i in batch:
  c ← cluster_ids[i];  p ← prompt_ids[i];  a ← advantages[i]

  # --- Macro: cluster budget on POSITIVE mass only ---
  pos_mass ← max(a, 0)
  spend_pos_ema[c] ← ema_alpha * pos_mass + (1 - ema_alpha) * spend_pos_ema[c]
  ratio ← spend_pos_ema[c] / (mean(spend_pos_ema) + 1e-8)
  IF ratio > theta:
    macro_w[i] ← decay ^ (ratio - theta)
  ELSE:
    macro_w[i] ← 1.0

  # --- Micro: per-prompt solve-rate gate (SELF-style, already built) ---
  IF greedy_solve_ema[p] > tau_solve:
    micro_w[i] ← lambda_self          # down-weight already-solved "winner" prompts
  ELSE:
    micro_w[i] ← 1.0

  # --- Combine ---
  gate_weight[i] ← macro_w[i] * micro_w[i]
END FOR

RETURN gate_weights
```

**Key Design Choices (H-CB-GRPO):**
- **Positive-mass-only spend** (`max(advantage, 0)`) distinguishes harmful winner-reinforcement from useful corrective gradient on hard clusters
- **Two-level gating** addresses both topic-level imbalance (macro) and per-prompt winner-take-all (micro)
- **Micro term reuse** leverages existing OSELFGate machinery—no new state tracking required
- **Soft combination** (`lambda_self` instead of hard cutoff) maintains differentiability
- **Configurable combination operator** supports ablation (multiplicative, min, harmonic_mean)

**Ablation Conditions (Requirement 22/23/39):**
1. Vanilla GRPO (no gating)
2. CB-GRPO (macro only, refined per positive-mass fix)
3. O-SELF (micro only)
4. H-CB-GRPO (macro × micro)

### Algorithm 3: Pass@k Unbiased Estimator

```
ALGORITHM: Pass@k Unbiased Estimator
INPUT: dataset, model, num_samples (n), k_values
OUTPUT: pass_at_k_values for each k

INITIALIZE:
  all_correctness ← empty list

FOR each problem in dataset:
  # 1. Generate n solutions
  solutions ← []
  FOR i = 1 to n:
    solution ← model.generate(problem.prompt)
    solutions.append(solution)
  END FOR
  
  # 2. Evaluate correctness
  correctness ← []
  FOR solution in solutions:
    is_correct ← evaluate_correctness(solution, problem.ground_truth)
    correctness.append(is_correct)
  END FOR
  
  all_correctness.append(correctness)
END FOR

# 3. Compute Pass@k for each k
pass_at_k_values ← []
FOR k in k_values:
  total_pass ← 0
  
  FOR correctness in all_correctness:
    c ← sum(correctness)  # number of correct solutions
    
    # Unbiased estimator: Pass@k = 1 - C(n-c, k) / C(n, k)
    IF n - c < k:
      pass_prob ← 1.0  # all samples would be correct
    ELSE:
      pass_prob ← 1.0 - C(n - c, k) / C(n, k)
    END IF
    
    total_pass ← total_pass + pass_prob
  END FOR
  
  pass_at_k ← total_pass / len(all_correctness)
  pass_at_k_values.append(pass_at_k)
END FOR

RETURN pass_at_k_values
```

**Formula Explanation:**
- `C(n, k)` is binomial coefficient "n choose k"
- `c` is number of correct solutions out of `n` total
- Formula gives probability that at least one of `k` randomly sampled solutions is correct
- Unbiased because it accounts for finite sampling

### Algorithm 3: Interference Diagnostics (N5 Diagnostic Contribution)

```
ALGORITHM: Interference Metrics Computation
INPUT: model π_θ (checkpoint), base model π_b, probing_dataset
OUTPUT: Δ⁺, ‖Δ‖, entropy_metrics

# --- 1. Probing Dataset Construction (one-time, pre-training) ---
probing_dataset ← []
FOR each prompt p in training_prompts:
  responses ← []
  FOR g = 1 to G:  # G=4 rollouts
    response ← π_b.generate(p)
    is_correct ← evaluate_correctness(response, ground_truth[p])
    responses.append({text: response, correct: is_correct, log_prob: π_b.log_prob(response|p)})
  END FOR
  probing_dataset.append({prompt: p, responses: responses})
END FOR

# --- 2. Compute Interference Metrics at Checkpoint ---
Δ⁺ ← 0  # Interference on base-correct completions
‖Δ‖ ← 0  # Relative influence magnitude
entropy_sum ← 0

FOR each item in probing_dataset:
  FOR each response in item.responses:
    # Δ log π for this completion
    Δ_log_prob ← π_θ.log_prob(response.text | item.prompt) - response.log_prob
    
    # Accumulate for base-correct completions (Δ⁺)
    IF response.correct:
      Δ⁺ ← Δ⁺ + Δ_log_prob
    
    # Accumulate squared change (‖Δ‖)
    ‖Δ‖ ← ‖Δ‖ + (Δ_log_prob)²
  END FOR
END FOR

# Normalize
Δ⁺ ← Δ⁺ / num_base_correct
‖Δ‖ ← sqrt(‖Δ‖ / total_responses)

# --- 3. Entropy Tracking (near-zero marginal cost, reuses generation logits) ---
entropy_metrics ← compute_mean_entropy(π_θ, generation_logits_cache)

RETURN Δ⁺, ‖Δ‖, entropy_metrics
```

**Interpretation:**
- **Δ⁺ (Interference)**: Expected change in log-probability of base-model-correct completions. Negative values indicate suppression of correct solution modes.
- **‖Δ‖ (Relative Influence)**: Expected squared change in log-probability over all completions. Measures overall distribution shift.
- **Entropy**: Token-level entropy during generation. Lower entropy indicates reduced exploration.

**Correlation Analysis (Requirement 40.6):**
```
# Compute Pearson/Spearman correlation between:
# - Cluster-spend Gini coefficient (Requirement 24.4)
# - Δ⁺ / ‖Δ‖ interference metrics

# Report as descriptive/exploratory (small N conditions ~8-12 data points)
# State limitation explicitly in summary report
```

**Validates: Requirements 40.1-40.7**

### Algorithm 4: Shrinkage Slope Computation

```
ALGORITHM: Shrinkage Slope Computation
INPUT: base_pass_at_k, rl_pass_at_k, k_values
OUTPUT: slope, intercept, r_squared, confidence_interval

# 1. Compute Δ Pass@k
delta_pass_k ← []
FOR i = 0 to len(k_values) - 1:
  delta ← rl_pass_at_k[i] - base_pass_at_k[i]
  delta_pass_k.append(delta)
END FOR

# 2. Log transform k values
log_k ← [log(k) for k in k_values]

# 3. Linear regression: delta_pass_k = slope * log_k + intercept
slope, intercept ← linear_regression(log_k, delta_pass_k)

# 4. Compute R²
predicted ← [slope * x + intercept for x in log_k]
ss_res ← sum((delta_pass_k[i] - predicted[i])² for i in range(len(k_values)))
ss_tot ← sum((delta_pass_k[i] - mean(delta_pass_k))² for i in range(len(k_values)))
r_squared ← 1 - ss_res / ss_tot

# 5. Bootstrap confidence interval
slopes ← []
FOR i = 1 to 1000:  # bootstrap iterations
  # Resample with replacement
  indices ← random_choice(len(k_values), size=len(k_values), replace=True)
  log_k_sample ← [log_k[j] for j in indices]
  delta_sample ← [delta_pass_k[j] for j in indices]
  
  # Fit slope
  slope_i, _ ← linear_regression(log_k_sample, delta_sample)
  slopes.append(slope_i)
END FOR

# Compute 95% CI
lower_bound ← percentile(slopes, 2.5)
upper_bound ← percentile(slopes, 97.5)
confidence_interval ← (lower_bound, upper_bound)

RETURN slope, intercept, r_squared, confidence_interval
```

**Interpretation:**
- **Negative slope**: Boundary shrinkage (Pass@k deteriorates faster at high k)
- **Slope near zero**: No shrinkage (proportional improvement across k)
- **Positive slope**: Boundary expansion (rare, indicates genuine capability gain)

### Algorithm 4: Resume from Checkpoint

```
ALGORITHM: Resume from Checkpoint
INPUT: checkpoint_dir, model, optimizer, lr_scheduler, gate
OUTPUT: restored (step, epoch, metrics_history)

# 1. Detect latest checkpoint
checkpoint_path ← find_latest_checkpoint(checkpoint_dir)

IF checkpoint_path is None:
  PRINT "No checkpoint found, starting fresh"
  RETURN (0, 0.0, empty_dict)
END IF

# 2. Display checkpoint information
checkpoint_state ← load_checkpoint(checkpoint_path)
DISPLAY:
  - Experiment name
  - Step: checkpoint_state.step / checkpoint_state.total_steps
  - Epoch: checkpoint_state.epoch
  - Best Pass@1: checkpoint_state.best_pass_at_1
  - Gate type: checkpoint_state.gate_type
  - Saved timestamp

# 3. Prompt user
user_response ← INPUT("Resume from this checkpoint? [y/n]")

IF user_response != "y":
  PRINT "Starting fresh"
  RETURN (0, 0.0, empty_dict)
END IF

# 4. Verify checkpoint integrity
expected_hash ← read_file(checkpoint_path + ".sha256")
actual_hash ← compute_sha256(checkpoint_path)

IF expected_hash != actual_hash:
  PRINT "ERROR: Checkpoint corrupted (hash mismatch)"
  
  # Try fallback to previous checkpoint
  previous_checkpoint ← find_previous_checkpoint(checkpoint_path)
  IF previous_checkpoint exists:
    PRINT "Attempting to load previous checkpoint"
    checkpoint_path ← previous_checkpoint
    checkpoint_state ← load_checkpoint(checkpoint_path)
  ELSE:
    RAISE RuntimeError("No valid checkpoint available")
  END IF
END IF

# 5. Restore state
model.load_state_dict(checkpoint_state.model_state_dict)
optimizer.load_state_dict(checkpoint_state.optimizer_state_dict)
lr_scheduler.load_state_dict(checkpoint_state.lr_scheduler_state_dict)
gate.load_state_dict(checkpoint_state.gate_state_dict)

# 6. Restore RNG states for reproducibility
torch.set_rng_state(checkpoint_state.torch_rng_state)
numpy.random.set_state(checkpoint_state.numpy_rng_state)
random.setstate(checkpoint_state.python_rng_state)

PRINT "Training state restored successfully"
PRINT "Resuming from step", checkpoint_state.step

RETURN (
  checkpoint_state.step,
  checkpoint_state.epoch,
  checkpoint_state.metrics_history
)
```

**Error Handling Strategy:**
- Checksum verification catches corrupted files
- Fallback to previous checkpoint provides resilience
- User prompt prevents accidental overwrites
- RNG restoration ensures reproducibility

## Data Models

### Core Data Structures

```python
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Experiment identity
    exp_name: str
    model_size: str  # "0.5B" or "1.5B"
    compute_tier: str  # "smoke", "standard", "final"
    
    # Gate configuration
    gate_type: str  # "VanillaGate", "CBGRPOGate", etc.
    gate_params: dict = field(default_factory=dict)
    
    # Training configuration
    training_steps: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # GRPO configuration
    rollouts_per_prompt: int = 4  # G
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    
    # Reward configuration
    reward_mode: str = "positive-only"  # or "negative-only", "hybrid"
    
    # QLoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Checkpoint configuration
    checkpoint_interval: int = 150
    eval_interval: int = 150
    log_interval: int = 10
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    figures_dir: str = "figures"
    logs_dir: str = "logs"
    
    # Reproducibility
    random_seed: int = 42
    
    # Evaluation configuration
    eval_num_samples: int = 10  # n for Pass@k
    eval_k_values: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Clustering configuration (for CB-GRPO)
    num_clusters: int = 16
    clustering_model: str = "all-MiniLM-L6-v2"
    
    def __post_init__(self):
        """Validates configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validates configuration consistency."""
        # Model size validation
        if self.model_size not in ["0.5B", "1.5B", "3B"]:
            raise ValueError(f"Invalid model_size: {self.model_size}")
        
        # Compute tier validation
        if self.compute_tier not in ["smoke", "standard", "final"]:
            raise ValueError(f"Invalid compute_tier: {self.compute_tier}")
        
        # Gate type validation
        valid_gates = [
            "VanillaGate", "CBGRPOGate", "OSELFGate",
            "StaticSELFGate", "AdaptiveRolloutGate"
        ]
        if self.gate_type not in valid_gates:
            raise ValueError(f"Invalid gate_type: {self.gate_type}")
        
        # Model size vs. gate compatibility
        if self.model_size == "0.5B":
            incompatible_gates = ["OSELFGate", "StaticSELFGate", "AdaptiveRolloutGate"]
            if self.gate_type in incompatible_gates:
                raise ValueError(
                    f"Gate {self.gate_type} is not compatible with 0.5B model "
                    f"(only tested at 1.5B scale)"
                )
        
        # Reward mode validation
        if self.reward_mode not in ["positive-only", "negative-only", "hybrid"]:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}")
    
    @classmethod
    def from_compute_tier(
        cls,
        exp_name: str,
        model_size: str,
        gate_type: str,
        compute_tier: str,
        **kwargs
    ) -> "ExperimentConfig":
        """
        Factory method to create config from compute tier.
        
        Automatically sets training_steps based on tier.
        """
        tier_steps = {
            "smoke": 100,
            "standard": 800,
            "final": 1800,
        }
        
        training_steps = tier_steps[compute_tier]
        
        return cls(
            exp_name=exp_name,
            model_size=model_size,
            gate_type=gate_type,
            compute_tier=compute_tier,
            training_steps=training_steps,
            **kwargs
        )


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    
    # Training progress
    step: int
    epoch: float
    total_steps: int
    
    # Model and optimizer
    model_state_dict: dict
    optimizer_state_dict: dict
    lr_scheduler_state_dict: dict
    
    # Gate state
    gate_type: str
    gate_state_dict: dict
    
    # Best metrics
    best_pass_at_1: float
    best_checkpoint_step: int
    
    # Random state for reproducibility
    torch_rng_state: Any
    numpy_rng_state: dict
    python_rng_state: tuple
    
    # Configuration
    config: ExperimentConfig
    
    # Metrics history
    metrics_history: dict
    
    # Metadata
    timestamp: str
    hostname: str
    gpu_name: str
    
    def __post_init__(self):
        """Adds runtime metadata if not provided."""
        if not hasattr(self, "timestamp") or self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        if not hasattr(self, "hostname") or self.hostname is None:
            import socket
            self.hostname = socket.gethostname()
        
        if not hasattr(self, "gpu_name") or self.gpu_name is None:
            import torch
            if torch.cuda.is_available():
                self.gpu_name = torch.cuda.get_device_name(0)
            else:
                self.gpu_name = "CPU"


@dataclass
class PassAtKResult:
    """Results from Pass@k evaluation."""
    
    dataset_name: str
    model_name: str
    checkpoint_step: int
    k_values: list[int]
    pass_at_k_values: list[float]
    num_samples: int
    timestamp: str
    
    def to_dict(self) -> dict:
        """Converts to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "checkpoint_step": self.checkpoint_step,
            "pass_at_k": {
                f"k={k}": v for k, v in zip(self.k_values, self.pass_at_k_values)
            },
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
        }


@dataclass
class ClusterAssignment:
    """Cluster assignment for a prompt."""
    
    prompt_id: str
    prompt_text: str
    cluster_id: int
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Converts to dictionary (excluding embedding for compactness)."""
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "cluster_id": self.cluster_id,
        }


@dataclass
class RolloutBatch:
    """Batch of rollouts for gate processing."""
    
    prompt_ids: list[str]
    prompts: list[str]
    cluster_ids: list[int]
    rollouts: list[list[str]]  # outer: prompts, inner: G rollouts
    rewards: torch.Tensor  # shape: (batch_size, G)
    advantages: torch.Tensor  # shape: (batch_size, G)
    
    def __len__(self) -> int:
        return len(self.prompt_ids)


@dataclass
class TrainerState:
    """Current trainer state passed to gates."""
    
    step: int
    epoch: float
    total_steps: int
    
    @property
    def progress(self) -> float:
        """Training progress as fraction in [0, 1]."""
        return self.step / self.total_steps if self.total_steps > 0 else 0.0
```

## Error Handling

### Error Handling Strategy

```python
class RobustTrainingLoop:
    """
    Training loop with comprehensive error handling.
    
    Handles:
    1. VRAM allocation failures
    2. Checkpoint loading failures
    3. Dataset loading failures
    4. Evaluation failures
    5. Training divergence
    6. Generation timeouts
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.error_log = []
    
    def train_with_recovery(self):
        """Main training with error recovery."""
        try:
            # 1. Environment setup
            self._setup_environment()
            
            # 2. Model loading with VRAM check
            model = self._load_model_safe()
            
            # 3. Dataset loading with retry
            train_dataset = self._load_dataset_safe()
            
            # 4. Resume or start fresh
            resume_checkpoint = self._handle_resume()
            
            # 5. Training loop with checkpoint safety
            self._train_loop_safe(model, train_dataset, resume_checkpoint)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self._save_emergency_checkpoint()
            raise
        
        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
            self._log_error(e)
            self._save_emergency_checkpoint()
            raise
    
    def _load_model_safe(self) -> PreTrainedModel:
        """Loads model with VRAM safety checks."""
        try:
            # Check available VRAM
            if torch.cuda.is_available():
                free_vram_gb = torch.cuda.mem_get_info()[0] / 1e9
                
                if free_vram_gb < 2.0:
                    raise RuntimeError(
                        f"Insufficient VRAM: {free_vram_gb:.2f} GB free. "
                        f"Minimum 2GB required for model loading. "
                        f"Try restarting runtime."
                    )
                
                print(f"✓ VRAM check passed: {free_vram_gb:.2f} GB available")
            
            # Load model with quantization
            model = load_quantized_model(
                model_name=f"Qwen2.5-{self.config.model_size}-Instruct",
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Apply QLoRA
            model = apply_lora(
                model,
                r=self.config.lora_r,
                alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
            )
            
            # Verify VRAM usage
            if torch.cuda.is_available():
                allocated_vram_gb = torch.cuda.memory_allocated() / 1e9
                print(f"✓ Model loaded: {allocated_vram_gb:.2f} GB VRAM allocated")
                
                if allocated_vram_gb > 14.0:
                    raise RuntimeError(
                        f"VRAM usage too high: {allocated_vram_gb:.2f} GB. "
                        f"Maximum 14GB recommended for training. "
                        f"Try reducing model size or batch size."
                    )
            
            return model
        
        except Exception as e:
            print(f"ERROR loading model: {e}")
            self._log_error(e)
            raise
    
    def _load_dataset_safe(self) -> Dataset:
        """Loads dataset with retry logic."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                dataset = load_dataset("openai/gsm8k", "main", split="train")
                print(f"✓ Dataset loaded: {len(dataset)} examples")
                return dataset
            
            except Exception as e:
                print(f"ERROR loading dataset (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(
                        "Failed to load dataset after multiple attempts. "
                        "Check internet connection and HuggingFace access."
                    )
                    self._log_error(e)
                    raise
    
    def _train_loop_safe(
        self,
        model: PreTrainedModel,
        train_dataset: Dataset,
        resume_checkpoint: Optional[str],
    ):
        """Training loop with divergence detection and partial result saving."""
        try:
            # Training setup
            training_loop = GRPOTrainingLoop(...)
            
            # Resume if needed
            if resume_checkpoint:
                training_loop.load_checkpoint(resume_checkpoint)
            
            # Training with monitoring
            for step in range(training_loop.current_step, self.config.training_steps):
                try:
                    # Single training step
                    loss = training_loop.step()
                    
                    # Divergence detection
                    if loss > 100.0:
                        print(f"\n⚠️ WARNING: Loss diverged (loss={loss:.2f})")
                        print("Saving emergency checkpoint and pausing...")
                        training_loop.save_checkpoint("diverged")
                        
                        response = input("Continue training? [y/n]: ")
                        if response.lower() != "y":
                            break
                    
                    # Checkpoint at intervals
                    if step % self.config.checkpoint_interval == 0:
                        training_loop.save_checkpoint("last")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️ WARNING: OOM at step {step}")
                        print("Clearing cache and reducing batch size...")
                        
                        torch.cuda.empty_cache()
                        self.config.batch_size = max(1, self.config.batch_size // 2)
                        print(f"New batch size: {self.config.batch_size}")
                        
                        # Save state before continuing
                        training_loop.save_checkpoint("oom_recovery")
                        continue
                    else:
                        raise
        
        except Exception as e:
            print(f"\nERROR in training loop: {e}")
            self._log_error(e)
            self._save_emergency_checkpoint()
            raise
    
    def _save_emergency_checkpoint(self):
        """Saves emergency checkpoint on unexpected failure."""
        try:
            print("Attempting to save emergency checkpoint...")
            # Emergency save logic
            print("✓ Emergency checkpoint saved")
        except:
            print("✗ Failed to save emergency checkpoint")
    
    def _log_error(self, error: Exception):
        """Logs error with context."""
        self.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "step": getattr(self, "current_step", "unknown"),
        })
```

This comprehensive design document provides the complete technical specification for implementing the RLVR Training Pipeline research notebook. All components, algorithms, and data models are specified in detail for implementation.



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

The RLVR Training Pipeline includes a parser and pretty-printer for model-generated outputs. These components must satisfy correctness properties to ensure reliable reward computation. The key properties focus on **round-trip equivalence** between parsing and printing operations.

### Property 1: Parser Extracts Reasoning Content

*For any* model output containing valid `<reasoning>...</reasoning>` tags, the parser SHALL extract the text content between the tags, stripped of leading/trailing whitespace.

**Validates: Requirements 36.1, 36.7**

### Property 2: Parser Extracts Answer Content

*For any* model output containing valid `<answer>...</answer>` tags, the parser SHALL extract the text content between the tags, stripped of leading/trailing whitespace.

**Validates: Requirements 36.2, 36.7**

### Property 3: Parser Extracts Boxed Content

*For any* model output containing `\boxed{...}` within the answer section, the parser SHALL extract the content within the braces.

**Validates: Requirements 36.3**

### Property 4: Pretty-Printer Formats Valid Structure

*For any* valid reasoning text and answer text, the pretty-printer SHALL produce output containing properly formatted `<reasoning>` tags, `<answer>` tags, and `\boxed{...}` content with escaped special characters.

**Validates: Requirements 37.1, 37.2, 37.3, 37.5**

### Property 5: Round-Trip Preservation (Primary Correctness Property)

*For any* valid output structure (reasoning, answer, boxed value), parsing then printing then parsing SHALL produce equivalent extracted components. That is:

```
parse(print(parse(text))) ≡ parse(text)
```

Where equivalence means:
- Same reasoning content (after whitespace normalization)
- Same answer content (after whitespace normalization)  
- Same boxed value

**Validates: Requirements 37.6, 38.1**

**This is the fundamental correctness property for the parser/printer system.** If this property holds for all valid inputs, we can trust that reward computation operates on correctly parsed data.

## Testing Strategy

### Dual Testing Approach

The RLVR Training Pipeline requires both **unit tests** and **property-based tests** for comprehensive correctness verification.

#### Unit Tests

Unit tests verify specific examples, edge cases, and integration points:

1. **Parser unit tests:**
   - Valid XML tags with simple content
   - Malformed tags return None (Req 36.4)
   - Nested tags extract outermost (Req 36.5)
   - Multi-line content handling (Req 36.6)

2. **Pretty-printer unit tests:**
   - Newline insertion between tags (Req 37.4)
   - Simple round-trip examples

3. **Reward computation unit tests:**
   - Known correct answers produce correctness reward = 1.0
   - Known incorrect answers produce correctness reward = 0.0 (positive mode)
   - Valid XML format produces format reward = 1.0
   - Missing tags produce format reward = 0.0

4. **Integration tests:**
   - End-to-end reward computation for sample problems
   - Checkpoint save/load for different gate types
   - Resume after simulated disconnect
   - Pass@k computation on small sample dataset

#### Property-Based Tests

Property-based tests verify universal properties across randomized inputs using **Hypothesis** (Python PBT library):

1. **Property 1-4**: Parser and pretty-printer behavior
   - Generator: Random strings with varying tag structures, special characters, whitespace patterns
   - Minimum 100 iterations per property
   - Tag format: `# Feature: rlvr-training-pipeline, Property 1: Parser extracts reasoning content`

2. **Property 5: Round-trip preservation** (Most critical)
   - Generator: Random valid output structures (reasoning, answer, boxed value)
   - Test: `parse(print(parse(text))) == parse(text)` for all generated inputs
   - Minimum 100 iterations
   - Tag format: `# Feature: rlvr-training-pipeline, Property 5: Round-trip preservation`

3. **Smoke tier validation:**
   - Run all property tests during smoke tier (100 steps) to verify pipeline correctness
   - Log any property failures with the counterexample that failed (Req 38.3)

#### Property Test Configuration

- **Library:** Hypothesis 6.x
- **Iterations:** Minimum 100 per property (due to randomization)
- **Shrinking:** Enabled (Hypothesis automatically minimizes failing examples)
- **Reproducibility:** Fixed random seed for deterministic test runs
- **Test tagging:** Each property test references design document property number

Example property test structure:

```python
from hypothesis import given, strategies as st

@given(
    reasoning=st.text(min_size=1),
    answer=st.text(min_size=1),
    boxed_value=st.text(min_size=1)
)
def test_round_trip_property(reasoning: str, answer: str, boxed_value: str):
    """
    Feature: rlvr-training-pipeline, Property 5: Round-trip preservation
    
    For any valid output structure, parse(print(parse(text))) == parse(text)
    """
    # Generate output using pretty-printer
    output = pretty_print(reasoning, answer, boxed_value)
    
    # Parse once
    parsed1 = parse(output)
    
    # Print and parse again
    output2 = pretty_print(parsed1.reasoning, parsed1.answer, parsed1.boxed)
    parsed2 = parse(output2)
    
    # Verify equivalence
    assert parsed1.reasoning.strip() == parsed2.reasoning.strip()
    assert parsed1.answer.strip() == parsed2.answer.strip()
    assert parsed1.boxed == parsed2.boxed
```

### Testing Approach Rationale

**Why property-based testing is appropriate:**
- Parser/pretty-printer are **pure functions** with clear input/output behavior
- **Round-trip property** is universal and should hold for all valid inputs
- Input space is large (arbitrary strings, tag structures, special characters)
- This is exactly the type of code PBT excels at testing

**Why integration tests are also necessary:**
- Training loop, evaluation, and checkpoint system involve external state and I/O
- Pass@k evaluation requires actual model generation (expensive, non-deterministic)
- Gate state persistence requires filesystem operations
- These are tested with representative examples rather than exhaustive randomization

**Balanced approach:**
- Property tests verify core parser/printer correctness (100+ iterations each)
- Unit tests verify specific edge cases and known examples
- Integration tests verify end-to-end workflow with real models and data
- Together, they provide confidence in system correctness

## Implementation Notes

### Memory Management for Colab T4 (16GB VRAM)

The system must operate within strict VRAM constraints:

1. **Model Loading:**
   - 4-bit NF4 quantization reduces model size by ~4x
   - QLoRA (r=16) adds minimal parameters (~2-8M depending on model size)
   - Gradient checkpointing trades computation for memory
   - Target: <6GB for 0.5B, <10GB for 1.5B, <14GB for 3B

2. **Training:**
   - Sequential rollout generation (generate one at a time)
   - Batch size tuning: start at 4, reduce to 2 or 1 if OOM
   - Gradient accumulation compensates for small batch size
   - Reward computation on CPU (regex operations)
   - Clear GPU cache between major operations

3. **Evaluation:**
   - Generate solutions sequentially, not in batches
   - Offload model to CPU between training and evaluation if needed
   - For Pass@k with n=64, generate in chunks of 8-10

4. **Monitoring:**
   - Log VRAM usage after model loading
   - Track peak VRAM during training
   - Automatic batch size reduction if VRAM exceeds 14GB
   - Warning if free VRAM drops below 2GB

### Colab Session Management

Handling 12-hour session limit and 90-minute idle disconnect:

1. **Checkpoint Frequency:**
   - Save every 150 steps for runs >4 hours
   - Save every 200 steps for runs <4 hours
   - Always save last.pt and best.pt
   - Maintain up to 3 incremental checkpoints

2. **Resume Protocol:**
   - Auto-detect latest checkpoint in Google Drive on restart
   - Display checkpoint info and prompt user to confirm resume
   - Verify integrity with SHA256 checksum
   - Restore model, optimizer, LR scheduler, gate, and RNG states

3. **Long Runs (Final Tier):**
   - Final tier (1800 steps) requires ~10-12 hours total
   - Expect 1-2 disconnects during run
   - Resume cycles extend total time but preserve progress
   - Save intermediate Pass@k results to avoid re-evaluation

4. **Google Drive Integration:**
   - Mount Drive at `/content/drive`
   - Store all checkpoints and results in Drive
   - Persist cluster assignments (compute once, reuse across sessions)

This design provides a complete specification for implementing a Colab-optimized RLVR training pipeline for studying the reasoning boundary paradox in small language models.

