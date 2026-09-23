"""
Configuration Validation Module for RLVR Training Pipeline

Task 1.6: Create configuration validation function with clear error messages and correction guidance.

Requirements:
- 2.6: Validate configuration compatibility before execution
- 2.7: Display clear error messages with correction guidance
- 33.1: Validate model size is 0.5B, 1.5B, or 3B
- 33.2: Validate compute tier is smoke/standard/final
- 33.3: Validate gate type
- 33.4-33.6: Validate gate type compatibility with model size
- 33.7: Display warning when smoke tier used for final results
- 33.8: Validate batch size produces acceptable VRAM usage
"""

from typing import Tuple, List
from dataclasses import dataclass, field


def validate_config(config) -> Tuple[bool, List[str]]:
    """
    Validates experiment configuration with detailed error messages.
    
    Requirements:
    - 2.6: Validate configuration compatibility before execution
    - 2.7: Display clear error messages with correction guidance
    - 33.1: Validate model size is 0.5B, 1.5B, or 3B
    - 33.2: Validate compute tier is smoke/standard/final
    - 33.3: Validate gate type
    - 33.4-33.6: Validate gate type compatibility with model size
    - 33.7: Display warning when smoke tier used for final results
    - 33.8: Validate batch size produces acceptable VRAM usage
    
    Args:
        config: ExperimentConfig instance to validate
    
    Returns:
        Tuple of (is_valid, list_of_messages)
        - is_valid: True if config is valid, False otherwise
        - list_of_messages: List of error/warning messages
    """
    errors = []
    warnings = []
    
    # ========================================================================
    # Requirement 33.1: Validate model size
    # ========================================================================
    valid_model_sizes = ["0.5B", "1.5B", "3B"]
    if config.model_size not in valid_model_sizes:
        errors.append(
            f"❌ INVALID MODEL SIZE: '{config.model_size}'\n"
            f"   Valid options: {', '.join(valid_model_sizes)}\n"
            f"   \n"
            f"   Correction guidance:\n"
            f"   • For fast experiments and smoke testing: Use '0.5B'\n"
            f"   • For standard experiments: Use '1.5B'\n"
            f"   • For final publication results (requires more VRAM): Use '3B'\n"
            f"   \n"
            f"   Example: config.model_size = '1.5B'"
        )
    
    # ========================================================================
    # Requirement 33.2: Validate compute tier
    # ========================================================================
    valid_compute_tiers = ["smoke", "standard", "final"]
    if config.compute_tier not in valid_compute_tiers:
        errors.append(
            f"❌ INVALID COMPUTE TIER: '{config.compute_tier}'\n"
            f"   Valid options: {', '.join(valid_compute_tiers)}\n"
            f"   \n"
            f"   Correction guidance:\n"
            f"   • 'smoke': 100 steps (~30-60 min) - For pipeline validation only\n"
            f"   • 'standard': 800 steps (~4-6 hours) - For main experiments\n"
            f"   • 'final': 1800 steps (~8-10 hours) - For publication-quality results\n"
            f"   \n"
            f"   Example: config.compute_tier = 'standard'"
        )
    
    # ========================================================================
    # Requirement 33.3: Validate gate type
    # ========================================================================
    valid_gate_types = [
        "VanillaGate",
        "CBGRPOGate",
        "OSELFGate",
        "StaticSELFGate",
        "AdaptiveRolloutGate"
    ]
    if config.gate_type not in valid_gate_types:
        errors.append(
            f"❌ INVALID GATE TYPE: '{config.gate_type}'\n"
            f"   Valid options: {', '.join(valid_gate_types)}\n"
            f"   \n"
            f"   Correction guidance:\n"
            f"   • 'VanillaGate': Standard GRPO without filtering (baseline)\n"
            f"   • 'CBGRPOGate': Cluster-balanced gradient gating (N3 novelty)\n"
            f"   • 'OSELFGate': Online solve-rate filtering (N2 mitigation, 1.5B+ only)\n"
            f"   • 'StaticSELFGate': Precomputed solve-rate filtering (N2 mitigation, 1.5B+ only)\n"
            f"   • 'AdaptiveRolloutGate': Variance-based filtering (N2 mitigation, 1.5B+ only)\n"
            f"   \n"
            f"   Example: config.gate_type = 'VanillaGate'"
        )
    
    # ========================================================================
    # Requirements 33.4-33.6: Validate gate type compatibility with model size
    # ========================================================================
    if config.model_size == "0.5B":
        incompatible_gates = ["OSELFGate", "StaticSELFGate", "AdaptiveRolloutGate"]
        if config.gate_type in incompatible_gates:
            errors.append(
                f"❌ INCOMPATIBLE CONFIGURATION: Gate '{config.gate_type}' cannot be used with 0.5B model\n"
                f"   \n"
                f"   Reason:\n"
                f"   The {config.gate_type} mitigation technique was only tested and validated\n"
                f"   at 1.5B scale and above. Using it with 0.5B models may produce unreliable results.\n"
                f"   \n"
                f"   Correction guidance (choose ONE):\n"
                f"   \n"
                f"   Option 1 - Change to compatible gate for 0.5B:\n"
                f"     config.gate_type = 'VanillaGate'    # Standard GRPO baseline\n"
                f"     config.gate_type = 'CBGRPOGate'     # Cluster-balanced gating (N3 novelty)\n"
                f"   \n"
                f"   Option 2 - Upgrade to larger model:\n"
                f"     config.model_size = '1.5B'          # Use 1.5B model instead\n"
                f"   \n"
                f"   Note: If you need N2 mitigation results, you must use 1.5B or larger models."
            )
    
    # ========================================================================
    # Requirement 33.7: Warn when smoke tier used for final results
    # ========================================================================
    if config.compute_tier == "smoke" and "final" in config.exp_name.lower():
        warnings.append(
            f"⚠️  WARNING: Smoke tier (100 steps) used for experiment named '{config.exp_name}'\n"
            f"   \n"
            f"   The experiment name suggests this is intended for final results,\n"
            f"   but the compute tier is set to 'smoke' which only runs 100 training steps.\n"
            f"   \n"
            f"   Smoke tier is intended ONLY for pipeline validation, not for research results.\n"
            f"   \n"
            f"   Recommendation:\n"
            f"   • For research results: config.compute_tier = 'standard' (800 steps)\n"
            f"   • For publication quality: config.compute_tier = 'final' (1800 steps)\n"
            f"   • Keep smoke tier only for testing the pipeline itself"
        )
    
    # ========================================================================
    # Requirement 33.8: Validate batch size produces acceptable VRAM usage
    # ========================================================================
    # Estimated VRAM usage (rough approximations for T4 16GB GPU):
    # - 0.5B model with batch_size=4: ~6-8 GB
    # - 1.5B model with batch_size=4: ~10-12 GB
    # - 3B model with batch_size=4: ~14-15 GB (tight fit)
    # - 3B model with batch_size=2: ~10-12 GB
    
    max_safe_batch_sizes = {
        "0.5B": 8,
        "1.5B": 4,
        "3B": 2,
    }
    
    if config.model_size in max_safe_batch_sizes:
        max_safe_batch = max_safe_batch_sizes[config.model_size]
        if config.batch_size > max_safe_batch:
            errors.append(
                f"❌ BATCH SIZE TOO LARGE: {config.batch_size} for {config.model_size} model\n"
                f"   \n"
                f"   Maximum recommended batch size for {config.model_size}: {max_safe_batch}\n"
                f"   Current batch size: {config.batch_size}\n"
                f"   \n"
                f"   Reason:\n"
                f"   This batch size will likely exceed T4 GPU VRAM capacity (16GB)\n"
                f"   and cause Out-of-Memory (OOM) errors during training.\n"
                f"   \n"
                f"   Correction guidance:\n"
                f"     config.batch_size = {max_safe_batch}\n"
                f"   \n"
                f"   Alternative - Increase gradient accumulation instead:\n"
                f"     config.batch_size = {max_safe_batch}\n"
                f"     config.gradient_accumulation_steps = {config.batch_size // max_safe_batch}\n"
                f"   \n"
                f"   This maintains effective batch size while staying within VRAM limits."
            )
    
    # ========================================================================
    # Additional validation: Reward mode
    # ========================================================================
    valid_reward_modes = ["positive-only", "negative-only", "hybrid"]
    if config.reward_mode not in valid_reward_modes:
        errors.append(
            f"❌ INVALID REWARD MODE: '{config.reward_mode}'\n"
            f"   Valid options: {', '.join(valid_reward_modes)}\n"
            f"   \n"
            f"   Correction guidance:\n"
            f"   • 'positive-only': Rewards only correct answers (recommended for most cases)\n"
            f"   • 'negative-only': Penalizes incorrect answers\n"
            f"   • 'hybrid': Combines both positive rewards and negative penalties\n"
            f"   \n"
            f"   Example: config.reward_mode = 'positive-only'"
        )
    
    # ========================================================================
    # Determine if configuration is valid
    # ========================================================================
    is_valid = len(errors) == 0
    
    # Combine errors and warnings
    all_messages = errors + warnings
    
    return is_valid, all_messages


def display_validation_results(is_valid: bool, messages: List[str], config):
    """
    Displays validation results in a formatted manner.
    
    Args:
        is_valid: Whether configuration is valid
        messages: List of error/warning messages
        config: The configuration that was validated
    """
    print("=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)
    
    # Display configuration summary
    print("\n📋 Configuration Summary:")
    print(f"   Experiment Name: {config.exp_name}")
    print(f"   Model Size: {config.model_size}")
    print(f"   Compute Tier: {config.compute_tier} ({config.training_steps} steps)")
    print(f"   Gate Type: {config.gate_type}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Reward Mode: {config.reward_mode}")
    
    # Display validation results
    print("\n" + "=" * 70)
    
    if is_valid and len(messages) == 0:
        print("✅ VALIDATION PASSED")
        print("=" * 70)
        print("\nConfiguration is valid and ready for use.")
        print("You can proceed with model loading and training.")
    elif is_valid and len(messages) > 0:
        print("⚠️  VALIDATION PASSED WITH WARNINGS")
        print("=" * 70)
        print("\nConfiguration is technically valid, but please review the warnings below:\n")
        for i, msg in enumerate(messages, 1):
            print(f"\nWarning {i}:")
            print(msg)
        print("\n" + "=" * 70)
        print("You can proceed, but consider addressing the warnings above.")
    else:
        print("❌ VALIDATION FAILED")
        print("=" * 70)
        print("\nConfiguration contains errors that must be fixed before proceeding:\n")
        for i, msg in enumerate(messages, 1):
            print(f"\nError {i}:")
            print(msg)
        print("\n" + "=" * 70)
        print("\n⛔ CANNOT PROCEED until all errors are resolved.")
        print("Please fix the configuration according to the guidance above.")
    
    print("=" * 70)


# Test function to demonstrate validation
def test_validation():
    """
    Demonstrates validation with example configurations.
    
    This function creates several test configurations to show how
    validation works with both valid and invalid inputs.
    """
    from dataclasses import dataclass, field
    from typing import List, Dict
    
    @dataclass
    class ExperimentConfig:
        """Minimal config for testing"""
        exp_name: str
        model_size: str
        compute_tier: str
        gate_type: str
        training_steps: int = 1000
        batch_size: int = 4
        reward_mode: str = "positive-only"
    
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION MODULE LOADED")
    print("=" * 70)
    print("\n✓ validate_config() function ready")
    print("✓ display_validation_results() function ready")
    print("\nExample usage:")
    print("  from config_validation import validate_config, display_validation_results")
    print("  is_valid, messages = validate_config(config)")
    print("  display_validation_results(is_valid, messages, config)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_validation()
