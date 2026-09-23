"""
CBGRPOConfig - Configuration for Capacity-Balanced GRPO Training

Extends GRPOConfig with parameters for capacity-aware cluster reweighting.

Novel Contribution:
- Tracks cumulative gradient-mass spend per cluster via EMA
- Applies soft decay when clusters exceed population-relative threshold
- Prevents capacity monopolization by dominant clusters

Parameters:
    n_clusters: Number of prompt clusters (default: 16)
    ema_alpha: EMA decay rate for spend tracking (default: 0.01)
                Lower = longer memory (recommended: 0.01-0.05)
    decay_factor: Decay applied to over-spending clusters (default: 0.9)
                  Must be < 1.0 for soft penalty (recommended: 0.85-0.95)
    theta_threshold: Multiplier for population mean threshold (default: 1.2)
                     Clusters with spend > theta * mean_spend are penalized
    balance_window: Steps between balance logging (default: 50)
    track_perplexity: Whether to track perplexity per cluster (default: True)

Author: Research Team
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

# Import base GRPOConfig from TRL
try:
    from trl import GRPOConfig as BaseGRPOConfig
except ImportError:
    raise ImportError(
        "TRL library required. Install with: pip install trl>=0.12.0"
    )


@dataclass
class CBGRPOConfig:
    """
    Configuration for Capacity-Balanced GRPO (CB-GRPO) training.
    
    This configuration extends GRPOConfig with capacity-aware parameters
    that enable trajectory-aware reweighting of prompt clusters.
    
    The CB-GRPO algorithm:
    1. Tracks cumulative gradient-mass spend per cluster via EMA
    2. Computes population-relative spend ratios
    3. Applies soft decay to over-spending clusters
    4. Logs balance metrics for analysis
    
    Example:
        >>> config = CBGRPOConfig(
        ...     n_clusters=16,
        ...     ema_alpha=0.01,
        ...     decay_factor=0.9,
        ...     theta_threshold=1.2
        ... )
        >>> config.to_dict()  # Export for logging
    """
    
    # === Capacity Balancing Parameters ===
    
    n_clusters: int = field(
        default=16,
        metadata={
            "help": "Number of prompt clusters for capacity tracking. "
                    "Should match the KMeans clustering parameter."
        }
    )
    
    ema_alpha: float = field(
        default=0.01,
        metadata={
            "help": "EMA decay rate for spend tracking. "
                    "Lower values = longer memory of training history. "
                    "Recommended range: [0.01, 0.05]"
        }
    )
    
    decay_factor: float = field(
        default=0.9,
        metadata={
            "help": "Decay multiplier for over-spending clusters. "
                    "Must be < 1.0 for soft penalty. "
                    "Recommended range: [0.85, 0.95]"
        }
    )
    
    theta_threshold: float = field(
        default=1.2,
        metadata={
            "help": "Threshold multiplier for population mean. "
                    "Clusters with spend > theta * mean_spend are penalized. "
                    "Recommended range: [1.1, 1.5]"
        }
    )
    
    balance_window: int = field(
        default=50,
        metadata={
            "help": "Number of steps between balance metric logging. "
                    "Set higher to reduce logging overhead."
        }
    )
    
    track_perplexity: bool = field(
        default=True,
        metadata={
            "help": "Whether to compute and track perplexity per cluster. "
                    "Adds computational overhead but provides richer metrics."
        }
    )
    
    # === Inherited GRPO Parameters ===
    # (These will be passed to GRPOConfig internally)
    
    output_dir: str = field(default="./outputs")
    num_train_epochs: int = field(default=1)
    max_steps: int = field(default=-1)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-6)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.1)
    
    # GRPO-specific
    beta: float = field(default=0.1)  # KL penalty
    num_generation_per_prompt: int = field(default=4)  # G rollouts
    
    # Generation
    max_new_tokens: int = field(default=512)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    
    # Checkpointing
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=3)
    
    # Memory optimization
    gradient_checkpointing: bool = field(default=True)
    fp16: bool = field(default=True)
    
    # Misc
    logging_steps: int = field(default=10)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="none")
    seed: int = field(default=42)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        
        # Validate capacity balancing parameters
        if self.ema_alpha <= 0 or self.ema_alpha > 1:
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {self.ema_alpha}"
            )
        
        if self.decay_factor <= 0 or self.decay_factor >= 1:
            raise ValueError(
                f"decay_factor must be in (0, 1), got {self.decay_factor}"
            )
        
        if self.theta_threshold <= 1.0:
            raise ValueError(
                f"theta_threshold must be > 1.0, got {self.theta_threshold}"
            )
        
        if self.n_clusters <= 0:
            raise ValueError(
                f"n_clusters must be positive, got {self.n_clusters}"
            )
    
    def to_grpo_config(self) -> "BaseGRPOConfig":
        """
        Convert CBGRPOConfig to TRL's GRPOConfig for trainer initialization.
        
        Returns:
            GRPOConfig instance with compatible parameters
        """
        return BaseGRPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            beta=self.beta,
            num_generation_per_prompt=self.num_generation_per_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            gradient_checkpointing=self.gradient_checkpointing,
            fp16=self.fp16,
            logging_steps=self.logging_steps,
            remove_unused_columns=self.remove_unused_columns,
            report_to=self.report_to,
            seed=self.seed,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for logging/saving."""
        return {
            # Capacity balancing
            "n_clusters": self.n_clusters,
            "ema_alpha": self.ema_alpha,
            "decay_factor": self.decay_factor,
            "theta_threshold": self.theta_threshold,
            "balance_window": self.balance_window,
            "track_perplexity": self.track_perplexity,
            
            # Training
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            
            # GRPO
            "beta": self.beta,
            "num_generation_per_prompt": self.num_generation_per_prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            
            # Checkpointing
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            
            # Memory
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": self.fp16,
            
            # Misc
            "logging_steps": self.logging_steps,
            "remove_unused_columns": self.remove_unused_columns,
            "report_to": self.report_to,
            "seed": self.seed,
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CBGRPOConfig":
        """Create CBGRPOConfig from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str) -> "CBGRPOConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# === Convenience Factory Functions ===

def create_cbgrpo_config_from_experiment(
    experiment_name: str,
    base_config: Any,  # ExperimentConfig from notebook
    **capacity_kwargs
) -> CBGRPOConfig:
    """
    Create CBGRPOConfig from experiment configuration.
    
    This factory function bridges the notebook's ExperimentConfig
    with CBGRPOConfig for seamless integration.
    
    Args:
        experiment_name: Name of the experiment (e.g., "exp2_n1_cbgrpo_0.5B")
        base_config: ExperimentConfig instance with training parameters
        **capacity_kwargs: Override capacity balancing parameters
    
    Returns:
        CBGRPOConfig instance ready for training
    
    Example:
        >>> config = create_cbgrpo_config_from_experiment(
        ...     "exp2_n1_cbgrpo_0.5B",
        ...     smoke_config,
        ...     ema_alpha=0.02,  # Custom capacity params
        ...     theta_threshold=1.3
        ... )
    """
    return CBGRPOConfig(
        # Capacity balancing (with overrides)
        n_clusters=capacity_kwargs.get('n_clusters', 16),
        ema_alpha=capacity_kwargs.get('ema_alpha', 0.01),
        decay_factor=capacity_kwargs.get('decay_factor', 0.9),
        theta_threshold=capacity_kwargs.get('theta_threshold', 1.2),
        balance_window=capacity_kwargs.get('balance_window', 50),
        track_perplexity=capacity_kwargs.get('track_perplexity', True),
        
        # Training parameters from base config
        output_dir=f"/content/drive/MyDrive/RLVR_Research/checkpoints/{experiment_name}",
        num_train_epochs=1,
        max_steps=base_config.training_steps,
        per_device_train_batch_size=base_config.batch_size,
        gradient_accumulation_steps=base_config.gradient_accumulation_steps,
        learning_rate=base_config.learning_rate,
        lr_scheduler_type=base_config.lr_scheduler_type,
        warmup_ratio=base_config.warmup_ratio,
        
        # GRPO parameters
        beta=0.1,
        num_generation_per_prompt=base_config.rollouts_per_prompt,
        max_new_tokens=base_config.max_new_tokens,
        temperature=base_config.temperature,
        top_p=base_config.top_p,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=base_config.checkpoint_interval,
        save_total_limit=3,
        
        # Memory
        gradient_checkpointing=True,
        fp16=True,
        
        # Logging
        logging_steps=base_config.log_interval,
        remove_unused_columns=False,
        report_to="none",
        seed=base_config.random_seed,
    )


# === CLI / Testing ===

if __name__ == "__main__":
    # Test configuration creation
    print("=" * 60)
    print("CBGRPOConfig Test")
    print("=" * 60)
    
    config = CBGRPOConfig(
        n_clusters=16,
        ema_alpha=0.01,
        decay_factor=0.9,
        theta_threshold=1.2,
        max_steps=800,
        output_dir="./test_output"
    )
    
    print(f"\n✓ Created CBGRPOConfig")
    print(f"  n_clusters: {config.n_clusters}")
    print(f"  ema_alpha: {config.ema_alpha}")
    print(f"  decay_factor: {config.decay_factor}")
    print(f"  theta_threshold: {config.theta_threshold}")
    
    # Test conversion to GRPOConfig
    grpo_config = config.to_grpo_config()
    print(f"\n✓ Converted to GRPOConfig")
    print(f"  Type: {type(grpo_config)}")
    print(f"  max_steps: {grpo_config.max_steps}")
    
    # Test serialization
    config_dict = config.to_dict()
    print(f"\n✓ Serialized to dict with {len(config_dict)} fields")
    
    # Test deserialization
    config2 = CBGRPOConfig.from_dict(config_dict)
    print(f"✓ Deserialized from dict")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
