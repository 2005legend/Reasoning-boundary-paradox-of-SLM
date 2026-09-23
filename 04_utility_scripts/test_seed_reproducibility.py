"""
Test script to verify random seed reproducibility for Task 1.4

This script validates that the seed-setting functionality works correctly
by testing determinism across multiple runs.
"""

import torch
import numpy as np
import random
import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking and reproducibility."""
    seed: int
    timestamp: str
    pytorch_version: str
    numpy_version: str
    python_version: str
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    
def set_random_seeds(seed: int = 42) -> ExperimentMetadata:
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python random.
    
    Args:
        seed: Random seed value (default: 42)
    
    Returns:
        ExperimentMetadata: Metadata object containing seed value and environment info
    
    Requirements:
        - 1.6: Set random seeds for reproducibility
        - 29.1: Set random seed for PyTorch, NumPy, and Python random module
        - 29.2: Save random seed value to experiment configuration
    """
    # Set PyTorch seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set Python random seed
    random.seed(seed)
    
    # Set environment variable for CUBLAS (additional determinism for CUDA operations)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms when available (Requirement 29.3)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")
    
    # Collect environment information for reproducibility (Requirement 29.4)
    import sys
    
    cuda_version = None
    gpu_name = None
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
    
    metadata = ExperimentMetadata(
        seed=seed,
        timestamp=datetime.now().isoformat(),
        pytorch_version=torch.__version__,
        numpy_version=np.__version__,
        python_version=sys.version,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
    )
    
    return metadata

def save_experiment_metadata(metadata: ExperimentMetadata, output_dir: str = "./results"):
    """
    Save experiment metadata to disk for reproducibility tracking.
    
    Args:
        metadata: ExperimentMetadata object to save
        output_dir: Directory to save metadata file (default: ./results)
    
    Requirements:
        - 29.2: Save random seed value to experiment configuration
        - 29.4: Save complete environment information
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata_file = output_path / "experiment_metadata.json"
    
    with open(metadata_file, 'w') as f:
        json.dump(asdict(metadata), f, indent=2)
    
    return metadata_file

def test_determinism(seed: int = 42):
    """
    Test that random seed setting produces deterministic results.
    
    Args:
        seed: Seed value to test
    
    Returns:
        bool: True if determinism test passes, False otherwise
    """
    print("="*60)
    print("Testing Random Seed Determinism (Task 1.4)")
    print("="*60)
    
    # First run
    metadata1 = set_random_seeds(seed=seed)
    torch_vals_1 = torch.rand(5).tolist()
    numpy_vals_1 = np.random.rand(5).tolist()
    python_vals_1 = [random.random() for _ in range(5)]
    
    print(f"\nFirst run with seed={seed}:")
    print(f"  PyTorch: {torch_vals_1}")
    print(f"  NumPy:   {numpy_vals_1}")
    print(f"  Python:  {python_vals_1}")
    
    # Second run with same seed
    metadata2 = set_random_seeds(seed=seed)
    torch_vals_2 = torch.rand(5).tolist()
    numpy_vals_2 = np.random.rand(5).tolist()
    python_vals_2 = [random.random() for _ in range(5)]
    
    print(f"\nSecond run with seed={seed}:")
    print(f"  PyTorch: {torch_vals_2}")
    print(f"  NumPy:   {numpy_vals_2}")
    print(f"  Python:  {python_vals_2}")
    
    # Check if values match
    torch_match = torch_vals_1 == torch_vals_2
    numpy_match = numpy_vals_1 == numpy_vals_2
    python_match = python_vals_1 == python_vals_2
    
    print("\n" + "="*60)
    print("Determinism Test Results:")
    print("="*60)
    print(f"  PyTorch deterministic: {'✓ PASS' if torch_match else '✗ FAIL'}")
    print(f"  NumPy deterministic:   {'✓ PASS' if numpy_match else '✗ FAIL'}")
    print(f"  Python deterministic:  {'✓ PASS' if python_match else '✗ FAIL'}")
    
    all_pass = torch_match and numpy_match and python_match
    
    if all_pass:
        print("\n✓ All determinism tests PASSED!")
    else:
        print("\n✗ Some determinism tests FAILED!")
    
    # Save metadata
    metadata_file = save_experiment_metadata(metadata1)
    print(f"\nMetadata saved to: {metadata_file}")
    
    # Display metadata content
    print("\nExperiment Metadata:")
    print("="*60)
    for key, value in asdict(metadata1).items():
        if key != 'python_version':  # Python version is too long
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value.split()[0]}")
    print("="*60)
    
    return all_pass

if __name__ == "__main__":
    # Run determinism test
    success = test_determinism(seed=42)
    
    # Exit with appropriate code
    exit(0 if success else 1)
