"""
CBGRPOTrainer - Capacity-Balanced GRPO Trainer

This module implements the novel CB-GRPO algorithm for balanced capacity allocation
during RLVR training of small language models.

Key Innovation:
- Trajectory-aware reweighting via EMA-tracked cumulative gradient-mass spend
- Soft decay for over-spending clusters (not hard clipping)
- Population-relative thresholding (adapts to dataset distribution)

Author: Research Team
Date: 2025
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings

from trl import GRPOTrainer
from transformers import PreTrainedModel


class CBGRPOTrainer(GRPOTrainer):
    """
    Capacity-Balanced GRPO Trainer (CB-GRPO)
    
    Extends TRL's GRPOTrainer with trajectory-aware capacity allocation.
    
    The CB-GRPO algorithm addresses the capacity allocation problem in RLVR:
    - Problem: Some prompt clusters dominate gradient updates, leaving others under-trained
    - Solution: Track cumulative spend per cluster via EMA, apply soft decay to over-spenders
    
    Mathematical Formulation:
    -------------------------
    Let S_c(t) = EMA of gradient-mass spend for cluster c at step t
    
    Update rule:
        S_c(t) = α * |∇L_c(t)| + (1-α) * S_c(t-1)
    
    where α is the EMA decay rate (default: 0.01)
    
    Capacity-aware weight:
        w_c(t) = decay^(S_c(t) / S_mean(t) - θ)   if S_c(t) / S_mean(t) > θ
               = 1.0                              otherwise
    
    where:
        - decay ∈ (0, 1) is the decay factor (default: 0.9)
        - θ > 1.0 is the threshold multiplier (default: 1.2)
    
    Final loss:
        L_cbgrpo = L_grpo * mean(w_c for all clusters in batch)
    
    Example:
        >>> from cbgrpo_config import CBGRPOConfig
        >>> from cbgrpo_trainer import CBGRPOTrainer
        >>> 
        >>> config = CBGRPOConfig(n_clusters=16, ema_alpha=0.01)
        >>> trainer = CBGRPOTrainer(
        ...     model=model,
        ...     args=config.to_grpo_config(),
        ...     train_dataset=train_dataset,
        ...     cbgrpo_config=config,  # Pass CB-GRPO config
        ...     tokenizer=tokenizer,
        ... )
        >>> trainer.train()
    
    Attributes:
        spend_ema: Tensor of shape (n_clusters,) tracking cumulative spend per cluster
        step_count: Current training step
        balance_history: List of balance metrics logged at each balance_window
    """
    
    def __init__(
        self,
        model: PreTrainedModel = None,
        args: Any = None,  # GRPOConfig
        train_dataset: Any = None,
        cbgrpo_config: Any = None,  # CBGRPOConfig
        tokenizer: Any = None,
        **kwargs
    ):
        """
        Initialize CBGRPOTrainer.
        
        Args:
            model: Pre-trained language model
            args: GRPOConfig instance (from cbgrpo_config.to_grpo_config())
            train_dataset: Training dataset with 'cluster_id' field
            cbgrpo_config: CBGRPOConfig instance with capacity balancing parameters
            tokenizer: Tokenizer for the model
            **kwargs: Additional arguments passed to GRPOTrainer
        """
        # Initialize parent GRPOTrainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # Extract CB-GRPO specific parameters
        if cbgrpo_config is not None:
            self.n_clusters = cbgrpo_config.n_clusters
            self.ema_alpha = cbgrpo_config.ema_alpha
            self.decay_factor = cbgrpo_config.decay_factor
            self.theta_threshold = cbgrpo_config.theta_threshold
            self.balance_window = cbgrpo_config.balance_window
            self.track_perplexity = cbgrpo_config.track_perplexity
        else:
            # Default parameters
            self.n_clusters = 16
            self.ema_alpha = 0.01
            self.decay_factor = 0.9
            self.theta_threshold = 1.2
            self.balance_window = 50
            self.track_perplexity = True
        
        # Initialize spend tracking
        # Use 1.0 as initial value to avoid division by zero
        # This represents "unit spend" before any training
        self.register_buffer(
            'spend_ema',
            torch.ones(self.n_clusters, dtype=torch.float32)
        )
        
        # Tracking variables
        self.step_count = 0
        self.balance_history: List[Dict[str, Any]] = []
        self.cluster_update_counts = torch.zeros(self.n_clusters, dtype=torch.long)
        
        # Log initialization
        print("\n" + "=" * 60)
        print("CB-GRPO TRAINER INITIALIZED")
        print("=" * 60)
        print(f"  n_clusters: {self.n_clusters}")
        print(f"  ema_alpha: {self.ema_alpha}")
        print(f"  decay_factor: {self.decay_factor}")
        print(f"  theta_threshold: {self.theta_threshold}")
        print(f"  balance_window: {self.balance_window}")
        print(f"  track_perplexity: {self.track_perplexity}")
        print("=" * 60 + "\n")
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a tensor as a buffer (moved to correct device automatically)."""
        if hasattr(self, 'model') and hasattr(self.model, 'register_buffer'):
            self.model.register_buffer(name, tensor)
        else:
            # Fallback: store as attribute
            setattr(self, name, tensor.to(self.args.device if hasattr(self, 'args') else 'cuda'))
    
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> torch.Tensor:
        """
        Compute capacity-balanced GRPO loss.
        
        This method:
        1. Computes standard GRPO loss via parent class
        2. Extracts cluster IDs from inputs
        3. Updates EMA spend for clusters in current batch
        4. Computes capacity-aware weights
        5. Reweights loss to balance capacity allocation
        
        Args:
            model: The model being trained
            inputs: Batch dictionary containing:
                - 'input_ids': Input token IDs
                - 'attention_mask': Attention mask
                - 'cluster_id': Cluster ID for each sample (required for CB-GRPO)
            return_outputs: Whether to return model outputs
        
        Returns:
            Reweighted loss (and optionally outputs)
        """
        # Step 1: Compute standard GRPO loss
        outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Step 2: Extract cluster IDs
        cluster_ids = inputs.get('cluster_id', None)
        
        # If no cluster IDs, return standard loss (fallback to vanilla GRPO)
        if cluster_ids is None:
            warnings.warn(
                "No 'cluster_id' field in inputs. CB-GRPO requires cluster assignments. "
                "Falling back to vanilla GRPO (uniform weighting).",
                UserWarning
            )
            return outputs if return_outputs else loss
        
        # Step 3: Update EMA spend for clusters in this batch
        # Use gradient magnitude as proxy for "capacity consumption"
        with torch.no_grad():
            # Get batch-level loss magnitude
            batch_loss_val = loss.detach().abs().item()
            
            # Update spend for each unique cluster in the batch
            unique_clusters = torch.unique(cluster_ids)
            
            for cid in unique_clusters:
                cid_int = int(cid.item())
                
                # EMA update: S_c = α * |loss| + (1-α) * S_c
                self.spend_ema[cid_int] = (
                    self.ema_alpha * batch_loss_val +
                    (1 - self.ema_alpha) * self.spend_ema[cid_int]
                )
                
                # Track update counts
                self.cluster_update_counts[cid_int] += 1
        
        # Step 4: Compute capacity-aware weights
        mean_spend = self.spend_ema.mean()
        
        # Compute spend ratios for clusters in this batch
        # ratio = S_c / mean(S)
        cluster_spend = self.spend_ema[cluster_ids.long()]
        ratios = cluster_spend / (mean_spend + 1e-8)
        
        # Compute weights: w = decay^(ratio - θ) if ratio > θ, else 1.0
        weights = torch.where(
            ratios > self.theta_threshold,
            torch.pow(self.decay_factor, ratios - self.theta_threshold),
            torch.ones_like(ratios)
        )
        
        # Step 5: Reweight loss
        # Use mean weight across batch
        loss = loss * weights.mean()
        
        # Step 6: Log balance metrics periodically
        self.step_count += 1
        if self.step_count % self.balance_window == 0:
            self._log_balance_metrics()
        
        return (loss, outputs[1]) if return_outputs else loss
    
    def _log_balance_metrics(self):
        """Log capacity balance metrics for analysis."""
        # Compute balance metrics
        mean_spend = self.spend_ema.mean().item()
        std_spend = self.spend_ema.std().item()
        max_spend = self.spend_ema.max().item()
        min_spend = self.spend_ema.min().item()
        
        # Compute over-spending clusters
        overspending = (self.spend_ema > self.theta_threshold * mean_spend).sum().item()
        
        # Compute Gini coefficient (measure of inequality)
        sorted_spend = torch.sort(self.spend_ema)[0]
        n = len(sorted_spend)
        cumsum = torch.cumsum(sorted_spend, dim=0)
        gini = (2 * torch.sum(torch.arange(1, n+1).float() * sorted_spend) - (n + 1) * sorted_spend.sum()) / (n * sorted_spend.sum())
        gini_coef = gini.item()
        
        # Store in history
        metrics = {
            "step": self.step_count,
            "mean_spend": mean_spend,
            "std_spend": std_spend,
            "max_spend": max_spend,
            "min_spend": min_spend,
            "spend_ratio": max_spend / (min_spend + 1e-8),
            "overspending_clusters": overspending,
            "gini_coefficient": gini_coef,
            "spend_distribution": self.spend_ema.cpu().tolist()
        }
        self.balance_history.append(metrics)
        
        # Print summary
        print(f"\n[Step {self.step_count}] Capacity Balance Metrics:")
        print(f"  Mean spend: {mean_spend:.4f}")
        print(f"  Std spend: {std_spend:.4f}")
        print(f"  Max/Min ratio: {max_spend / (min_spend + 1e-8):.2f}")
        print(f"  Over-spending clusters: {overspending}/{self.n_clusters}")
        print(f"  Gini coefficient: {gini_coef:.4f} (0=perfect balance, 1=max inequality)")
    
    def save_balance_history(self, output_dir: str):
        """Save balance history to JSON file for analysis."""
        output_path = Path(output_dir) / "cbgrpo_balance_history.json"
        
        with open(output_path, 'w') as f:
            json.dump({
                "config": {
                    "n_clusters": self.n_clusters,
                    "ema_alpha": self.ema_alpha,
                    "decay_factor": self.decay_factor,
                    "theta_threshold": self.theta_threshold,
                    "balance_window": self.balance_window,
                },
                "history": self.balance_history
            }, f, indent=2)
        
        print(f"✓ Saved balance history to: {output_path}")
    
    def get_balance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive balance report.
        
        Returns:
            Dictionary with balance statistics and recommendations
        """
        if len(self.balance_history) == 0:
            return {"error": "No balance history recorded yet"}
        
        # Get final metrics
        final = self.balance_history[-1]
        
        # Compute trends
        initial_gini = self.balance_history[0]["gini_coefficient"]
        final_gini = final["gini_coefficient"]
        gini_change = final_gini - initial_gini
        
        # Identify most/least spending clusters
        spend_dist = np.array(final["spend_distribution"])
        most_spending = np.argsort(spend_dist)[-3:][::-1].tolist()
        least_spending = np.argsort(spend_dist)[:3].tolist()
        
        return {
            "summary": {
                "total_steps": self.step_count,
                "final_gini": final_gini,
                "gini_change": gini_change,
                "balance_improved": gini_change < 0,
            },
            "spend_distribution": final["spend_distribution"],
            "most_spending_clusters": most_spending,
            "least_spending_clusters": least_spending,
            "recommendations": self._generate_recommendations(final)
        }
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate tuning recommendations based on balance metrics."""
        recommendations = []
        
        gini = metrics["gini_coefficient"]
        overspending = metrics["overspending_clusters"]
        
        if gini > 0.3:
            recommendations.append(
                f"High inequality (Gini={gini:.2f}). Consider increasing ema_alpha "
                f"or decreasing theta_threshold."
            )
        
        if overspending > self.n_clusters // 2:
            recommendations.append(
                f"Many clusters over-spending ({overspending}/{self.n_clusters}). "
                f"Consider increasing decay_factor (less aggressive decay)."
            )
        
        if overspending == 0:
            recommendations.append(
                "No clusters over-spending. Consider decreasing theta_threshold "
                "for more active balancing."
            )
        
        return recommendations if recommendations else ["Balance looks good!"]


# === Testing ===

if __name__ == "__main__":
    print("=" * 60)
    print("CBGRPOTrainer Test")
    print("=" * 60)
    
    # Test initialization
    trainer = CBGRPOTrainer(
        model=None,
        args=None,
        train_dataset=None,
        cbgrpo_config=None
    )
    
    print(f"\n✓ Trainer initialized")
    print(f"  spend_ema shape: {trainer.spend_ema.shape}")
    print(f"  spend_ema device: {trainer.spend_ema.device}")
    
    print("\n" + "=" * 60)
    print("✅ CBGRPOTrainer ready for training!")
    print("=" * 60)
