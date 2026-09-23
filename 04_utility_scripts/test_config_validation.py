"""
Test script for configuration validation (Task 1.6)

This script demonstrates the configuration validation function with various
valid and invalid configurations to show error messages and correction guidance.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from config_validation import validate_config, display_validation_results


@dataclass
class ExperimentConfig:
    """
    Simplified ExperimentConfig for testing validation.
    """
    exp_name: str
    model_size: str
    compute_tier: str
    gate_type: str
    training_steps: int = 1000
    batch_size: int = 4
    reward_mode: str = "positive-only"
    gate_params: Dict = field(default_factory=dict)
    gradient_accumulation_steps: int = 1
    
    @classmethod
    def from_compute_tier(cls, exp_name, model_size, gate_type, compute_tier, **kwargs):
        """Factory method to create config from compute tier"""
        tier_steps = {
            "smoke": 100,
            "standard": 800,
            "final": 1800,
        }
        training_steps = tier_steps.get(compute_tier, 1000)
        return cls(
            exp_name=exp_name,
            model_size=model_size,
            gate_type=gate_type,
            compute_tier=compute_tier,
            training_steps=training_steps,
            **kwargs
        )


def test_examples():
    """Run test examples to demonstrate validation"""
    
    # Example 1: Valid configuration - 0.5B with VanillaGate
    print("\n" + "#" * 70)
    print("# Example 1: Valid Configuration - 0.5B with VanillaGate (smoke tier)")
    print("#" * 70)
    
    config_valid = ExperimentConfig.from_compute_tier(
        exp_name="test_valid_0.5B_vanilla",
        model_size="0.5B",
        gate_type="VanillaGate",
        compute_tier="smoke"
    )
    
    is_valid, messages = validate_config(config_valid)
    display_validation_results(is_valid, messages, config_valid)
    
    
    # Example 2: Invalid configuration - 0.5B with OSELFGate (incompatible)
    print("\n\n" + "#" * 70)
    print("# Example 2: Invalid Configuration - 0.5B with OSELFGate (incompatible)")
    print("#" * 70)
    
    config_invalid = ExperimentConfig.from_compute_tier(
        exp_name="test_invalid_0.5B_oself",
        model_size="0.5B",
        gate_type="OSELFGate",  # This should fail
        compute_tier="standard"
    )
    
    is_valid, messages = validate_config(config_invalid)
    display_validation_results(is_valid, messages, config_invalid)
    
    
    # Example 3: Valid configuration - 1.5B with CBGRPOGate
    print("\n\n" + "#" * 70)
    print("# Example 3: Valid Configuration - 1.5B with CBGRPOGate")
    print("#" * 70)
    
    config_cb = ExperimentConfig.from_compute_tier(
        exp_name="exp2_n1_cbgrpo_1.5B",
        model_size="1.5B",
        gate_type="CBGRPOGate",
        compute_tier="standard",
        gate_params={"theta": 1.5, "decay": 0.98, "ema_alpha": 0.05}
    )
    
    is_valid, messages = validate_config(config_cb)
    display_validation_results(is_valid, messages, config_cb)
    
    
    # Example 4: Invalid model size
    print("\n\n" + "#" * 70)
    print("# Example 4: Invalid Configuration - Wrong model size")
    print("#" * 70)
    
    config_bad_size = ExperimentConfig(
        exp_name="test_bad_size",
        model_size="2B",  # Invalid size
        gate_type="VanillaGate",
        compute_tier="smoke"
    )
    
    is_valid, messages = validate_config(config_bad_size)
    display_validation_results(is_valid, messages, config_bad_size)
    
    
    # Example 5: Warning - Smoke tier with "final" in name
    print("\n\n" + "#" * 70)
    print("# Example 5: Warning - Smoke tier with 'final' in experiment name")
    print("#" * 70)
    
    config_warning = ExperimentConfig.from_compute_tier(
        exp_name="final_results_smoke",
        model_size="1.5B",
        gate_type="VanillaGate",
        compute_tier="smoke"  # This should produce a warning
    )
    
    is_valid, messages = validate_config(config_warning)
    display_validation_results(is_valid, messages, config_warning)
    
    
    # Example 6: Invalid gate type
    print("\n\n" + "#" * 70)
    print("# Example 6: Invalid Configuration - Wrong gate type")
    print("#" * 70)
    
    config_bad_gate = ExperimentConfig(
        exp_name="test_bad_gate",
        model_size="1.5B",
        gate_type="SuperGate",  # Invalid gate
        compute_tier="standard"
    )
    
    is_valid, messages = validate_config(config_bad_gate)
    display_validation_results(is_valid, messages, config_bad_gate)
    
    
    # Example 7: Batch size too large for 3B model
    print("\n\n" + "#" * 70)
    print("# Example 7: Invalid Configuration - Batch size too large for 3B")
    print("#" * 70)
    
    config_large_batch = ExperimentConfig(
        exp_name="test_large_batch",
        model_size="3B",
        gate_type="VanillaGate",
        compute_tier="standard",
        batch_size=8  # Too large for 3B
    )
    
    is_valid, messages = validate_config(config_large_batch)
    display_validation_results(is_valid, messages, config_large_batch)
    
    
    print("\n\n" + "=" * 70)
    print("✅ CONFIGURATION VALIDATION TESTING COMPLETE")
    print("=" * 70)
    print("\nThe validation function is ready for use in experiments.")
    print("Always validate your configuration before starting training!")


if __name__ == "__main__":
    test_examples()
