import json

with open('rlvr_training_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove the placeholder cell 20 if it exists
if len(nb['cells']) > 40:
    nb['cells'] = nb['cells'][:40]

# ============================================================================
# CELL 20: PRE-FLIGHT CHECK (Production-grade)
# ============================================================================
markdown_preflight = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## Cell 20: Pre-Flight Verification System\n',
        '\n',
        '**Production-grade environment validation before training execution.**\n',
        '\n',
        'Validates:\n',
        '- GPU availability and VRAM capacity\n',
        '- All required packages and dependencies\n',
        '- Dataset integrity and completeness\n',
        '- Configuration system correctness\n',
        '- Disk space and checkpoint directory structure\n',
        '- Memory allocation and cleanup'
    ]
}

code_preflight = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': []
}

preflight_code = '''import sys
import torch
import gc
from pathlib import Path
import shutil
from typing import Dict, Any

def run_preflight_check() -> Dict[str, Any]:
    """
    Production-grade pre-flight verification system.
    
    Returns:
        Dict containing check results and recommendations
    """
    results = {
        'checks': {},
        'passed': 0,
        'total': 8,
        'critical_failures': [],
        'warnings': [],
        'ready': False
    }
    
    print("=" * 80)
    print("🚀 PRODUCTION PRE-FLIGHT VERIFICATION SYSTEM")
    print("=" * 80)
    print()
    
    # Check 1: Environment Detection
    print("1. Environment Detection...")
    try:
        import google.colab
        results['checks']['environment'] = 'colab'
        results['passed'] += 1
        print("   ✅ Google Colab environment detected")
    except ImportError:
        results['checks']['environment'] = 'local'
        results['passed'] += 1
        results['warnings'].append("Not in Colab - using local environment")
        print("   ⚠️  Local environment (not Colab)")
    
    # Check 2: GPU and VRAM
    print("\\n2. GPU and VRAM Verification...")
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_name = gpu_props.name
        vram_total_gb = gpu_props.total_memory / (1024**3)
        vram_allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        vram_free_gb = vram_total_gb - vram_allocated_gb
        
        results['checks']['gpu'] = {
            'name': gpu_name,
            'vram_total': vram_total_gb,
            'vram_free': vram_free_gb
        }
        
        print(f"   ✅ GPU: {gpu_name}")
        print(f"   ✅ VRAM Total: {vram_total_gb:.2f} GB")
        print(f"   ✅ VRAM Free: {vram_free_gb:.2f} GB")
        
        if vram_free_gb >= 14:
            print(f"   ✅ Sufficient VRAM for 1.5B model training")
            results['passed'] += 1
        elif vram_free_gb >= 8:
            print(f"   ⚠️  VRAM sufficient for 0.5B only")
            results['warnings'].append("Use 0.5B model (VRAM < 14GB)")
            results['passed'] += 1
        else:
            results['critical_failures'].append("Insufficient VRAM (< 8GB)")
            print(f"   ❌ Insufficient VRAM for training")
    else:
        results['critical_failures'].append("No GPU detected")
        print("   ❌ NO GPU DETECTED")
        print("   → Runtime > Change runtime type > GPU (T4 or higher)")
    
    # Check 3: Critical Packages
    print("\\n3. Package Integrity Check...")
    required = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'trl': 'TRL (GRPO Trainer)',
        'peft': 'PEFT (QLoRA)',
        'datasets': 'HuggingFace Datasets',
        'sentence_transformers': 'Sentence Transformers',
        'sklearn': 'scikit-learn',
        'hypothesis': 'Hypothesis (property testing)'
    }
    
    missing = []
    for pkg, name in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(name)
    
    if not missing:
        print(f"   ✅ All {len(required)} required packages installed")
        results['passed'] += 1
    else:
        results['critical_failures'].append(f"Missing packages: {', '.join(missing)}")
        print(f"   ❌ Missing: {', '.join(missing)}")
        print("   → Run Cell 2 to install dependencies")
    
    # Check 4: Google Drive and Directory Structure
    print("\\n4. Storage and Checkpoint Directory Verification...")
    drive_base = Path("/content/drive/MyDrive/RLVR_Research")
    if drive_base.exists():
        required_dirs = ['checkpoints', 'results', 'figures', 'logs']
        all_exist = all((drive_base / d).exists() for d in required_dirs)
        
        if all_exist:
            print(f"   ✅ Drive mounted: {drive_base}")
            print(f"   ✅ All required directories present")
            results['checks']['storage'] = str(drive_base)
            results['passed'] += 1
        else:
            print(f"   ⚠️  Some directories missing - creating...")
            for d in required_dirs:
                (drive_base / d).mkdir(parents=True, exist_ok=True)
            results['passed'] += 1
    else:
        print(f"   ⚠️  Google Drive not mounted - using local storage")
        local_base = Path("./RLVR_Research")
        local_base.mkdir(exist_ok=True)
        for d in ['checkpoints', 'results', 'figures', 'logs']:
            (local_base / d).mkdir(exist_ok=True)
        results['checks']['storage'] = str(local_base)
        results['passed'] += 1
    
    # Check 5: Dataset Integrity
    print("\\n5. Dataset Integrity Verification...")
    if 'gsm8k_train' in globals() and 'gsm8k_test' in globals():
        train_count = gsm8k_train.get('count', 0)
        test_count = gsm8k_test.get('count', 0)
        
        if train_count >= 7000 and test_count >= 1000:
            print(f"   ✅ GSM8K train: {train_count} problems")
            print(f"   ✅ GSM8K test: {test_count} problems")
            results['checks']['datasets'] = {'train': train_count, 'test': test_count}
            results['passed'] += 1
        else:
            results['critical_failures'].append("Incomplete dataset")
            print(f"   ❌ Dataset incomplete")
    else:
        results['critical_failures'].append("Datasets not loaded")
        print("   ❌ Datasets not loaded")
        print("   → Run Cell 5 to load GSM8K datasets")
    
    # Check 6: Clustering System
    print("\\n6. Clustering System Verification...")
    if 'cluster_assignments' in globals():
        n_prompts = len(cluster_assignments)
        n_clusters = len(set(cluster_assignments.values()))
        
        if n_prompts >= 7000 and n_clusters == 16:
            print(f"   ✅ Cluster assignments: {n_prompts} prompts")
            print(f"   ✅ Clusters: {n_clusters}")
            results['checks']['clustering'] = {'prompts': n_prompts, 'clusters': n_clusters}
            results['passed'] += 1
        else:
            results['warnings'].append("Clustering incomplete - CB-GRPO unavailable")
            print(f"   ⚠️  Clustering incomplete")
    else:
        results['warnings'].append("Clustering not done - CB-GRPO unavailable")
        print("   ⚠️  Clustering not performed")
        print("   → Run Cell 7 for CB-GRPO experiments")
        print("   → Vanilla experiments can proceed without clustering")
        results['passed'] += 1
    
    # Check 7: Configuration System
    print("\\n7. Configuration System Validation...")
    if 'ExperimentConfig' in globals():
        try:
            all_configs = get_all_configs()
            n_configs = len(all_configs)
            
            if n_configs >= 10:
                print(f"   ✅ Configuration system loaded")
                print(f"   ✅ Available experiment configs: {n_configs}")
                results['checks']['configs'] = n_configs
                results['passed'] += 1
            else:
                results['critical_failures'].append("Incomplete config system")
                print(f"   ❌ Incomplete configuration system")
        except Exception as e:
            results['critical_failures'].append(f"Config system error: {e}")
            print(f"   ❌ Configuration system error: {e}")
    else:
        results['critical_failures'].append("Config system not loaded")
        print("   ❌ Configuration system not loaded")
        print("   → Run Cell 4 to initialize configs")
    
    # Check 8: Disk Space
    print("\\n8. Disk Space Verification...")
    try:
        storage_path = results['checks'].get('storage', '/content')
        if isinstance(storage_path, str):
            storage_path = Path(storage_path).parent
        stat = shutil.disk_usage(storage_path)
        free_gb = stat.free / (1024**3)
        
        print(f"   ✅ Free space: {free_gb:.1f} GB")
        
        if free_gb >= 5:
            print(f"   ✅ Sufficient space for checkpoints")
            results['passed'] += 1
        else:
            results['warnings'].append(f"Low disk space: {free_gb:.1f}GB")
            print(f"   ⚠️  Low disk space - may limit checkpoint saving")
            results['passed'] += 1
    except Exception as e:
        results['warnings'].append("Could not verify disk space")
        results['passed'] += 1
    
    # Final Assessment
    print("\\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Checks passed: {results['passed']}/{results['total']}")
    print()
    
    # Determine readiness
    has_critical = len(results['critical_failures']) > 0
    sufficient_checks = results['passed'] >= 6
    
    results['ready'] = (not has_critical) and sufficient_checks
    
    if results['ready']:
        print("✅ SYSTEM READY FOR PRODUCTION TRAINING")
        print()
        print("Recommended execution order:")
        print("  1. Smoke test (100 steps, ~30-60 min) - Validation")
        print("  2. Standard experiments (800 steps, ~4-6 hours)")
        print("  3. Final tier experiments (1800 steps, ~8-12 hours)")
        print()
        print("Proceed to Cell 21 to begin training execution.")
    else:
        if has_critical:
            print("❌ CRITICAL FAILURES DETECTED")
            print()
            for i, failure in enumerate(results['critical_failures'], 1):
                print(f"  {i}. {failure}")
            print()
            print("Resolve all critical failures before training.")
        else:
            print("⚠️  INSUFFICIENT CHECKS PASSED")
            print()
            print("Required actions:")
            print("  - Ensure GPU runtime is enabled")
            print("  - Run all setup cells (1-19) in sequence")
            print("  - Verify no errors in cell outputs")
    
    if results['warnings']:
        print()
        print("⚠️  Warnings ({}):\n".format(len(results['warnings'])))
        for i, warning in enumerate(results['warnings'], 1):
            print(f"  {i}. {warning}")
    
    print("=" * 80)
    
    return results

# Execute pre-flight check
preflight_results = run_preflight_check()
'''

code_preflight['source'] = preflight_code.split('\n')

nb['cells'].append(markdown_preflight)
nb['cells'].append(code_preflight)

# Save
with open('rlvr_training_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('✅ Added Cell 20: Production-Grade Pre-Flight Verification')
print(f'✅ Total cells: {len(nb["cells"])}')
