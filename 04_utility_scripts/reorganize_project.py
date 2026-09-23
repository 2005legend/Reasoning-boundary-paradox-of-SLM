"""
Project Reorganization Script
Cleans up and organizes the RLVR research project directory
"""

import os
import shutil
from pathlib import Path

# Base directory
base = Path(".")

# Create new structure
folders = {
    "docs/papers": "Research papers (PDFs)",
    "docs/reports": "Status reports and summaries",
    "scripts/notebook_fixes": "Scripts that fix/modify notebook",
    "scripts/validation": "Validation and verification scripts",
    "scripts/testing": "Test scripts",
    "notebook_cells": "Cell code snippets for manual pasting",
    "archive": "Old/unused files"
}

print("=" * 80)
print("🗂️  PROJECT REORGANIZATION")
print("=" * 80)
print()

# Create directories
print("Creating directory structure...")
for folder, desc in folders.items():
    (base / folder).mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {folder:30s} - {desc}")

print()
print("=" * 80)
print("MOVING FILES")
print("=" * 80)
print()

# File movements
moves = [
    # Papers
    ("2410.05695v2 ( base foundation).pdf", "docs/papers/"),
    ("2510.04028v1(debate on RLVR).pdf", "docs/papers/"),
    ("2604.18381v1(base paper close).pdf", "docs/papers/"),
    ("base close and base foundation merged.pdf", "docs/papers/"),
    ("PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_Diffusion_Models.pdf", "docs/papers/"),
    ("Using_Explainable_Techniques_to_Enhance_Chain_of_Thoughts_in_LLMs.pdf", "docs/papers/"),
    
    # Reports & Documentation
    ("BUG_FIX_REPORT.md", "docs/reports/"),
    ("COMPARISON_vs_SELF.md", "docs/reports/"),
    ("EXECUTION_ANALYSIS.md", "docs/reports/"),
    ("FINAL_FIX_INSTRUCTIONS.md", "docs/reports/"),
    ("FINAL_VALIDATION_REPORT.md", "docs/reports/"),
    ("IMPLEMENTATION_STATUS.md", "docs/reports/"),
    ("novelty_and_implementation_plan.md", "docs/reports/"),
    ("PRODUCTION_READY_SUMMARY.md", "docs/reports/"),
    ("READY_TO_TRAIN.md", "docs/reports/"),
    ("TASK_1.3_IMPLEMENTATION_SUMMARY.md", "docs/reports/"),
    ("task_1.6_completion_summary.md", "docs/reports/"),
    ("TASK_1_4_IMPLEMENTATION.md", "docs/reports/"),
    ("cell_validation_report.txt", "docs/reports/"),
    ("validation_report.txt", "docs/reports/"),
    
    # Notebook fix scripts
    ("add_cell5_simple.py", "scripts/notebook_fixes/"),
    ("add_model_loader_cell.py", "scripts/notebook_fixes/"),
    ("add_production_cells.py", "scripts/notebook_fixes/"),
    ("add_training_cell.py", "scripts/notebook_fixes/"),
    ("cell6_clustering.py", "scripts/notebook_fixes/"),
    ("diagnose_cell.py", "scripts/notebook_fixes/"),
    ("embed_preflight.py", "scripts/notebook_fixes/"),
    ("final_embed.py", "scripts/notebook_fixes/"),
    ("fix.py", "scripts/notebook_fixes/"),
    ("fix_cell20_properly.py", "scripts/notebook_fixes/"),
    ("fix_cell20_syntax.py", "scripts/notebook_fixes/"),
    ("fix_newlines.py", "scripts/notebook_fixes/"),
    ("fix_notebook.py", "scripts/notebook_fixes/"),
    ("fix_notebook_proper.py", "scripts/notebook_fixes/"),
    ("insert_cell5.py", "scripts/notebook_fixes/"),
    
    # Validation scripts
    ("check.py", "scripts/validation/"),
    ("comprehensive_validation.py", "scripts/validation/"),
    ("config_validation.py", "scripts/validation/"),
    ("final_cell_validation.py", "scripts/validation/"),
    ("final_verify.py", "scripts/validation/"),
    ("full_syntax_check.py", "scripts/validation/"),
    ("validate_all_cells.py", "scripts/validation/"),
    ("verify_execution.py", "scripts/validation/"),
    
    # Test scripts
    ("run_all_tests.py", "scripts/testing/"),
    ("test_config_validation.py", "scripts/testing/"),
    ("test_gpu_verification.py", "scripts/testing/"),
    ("test_seed_reproducibility.py", "scripts/testing/"),
    
    # Cell code snippets
    ("CELL_20_FIXED.txt", "notebook_cells/"),
    ("CELL_21_FINAL.txt", "notebook_cells/"),
    ("CELL_21_FIXED.txt", "notebook_cells/"),
    ("CELL_21_PRODUCTION_TRAINING.py", "notebook_cells/"),
    ("CELL_21_WORKING.txt", "notebook_cells/"),
    ("PREFLIGHT_CELL.txt", "notebook_cells/"),
    ("pre_flight_check.py", "notebook_cells/"),
    ("SMOKE_TEST_CELL.txt", "notebook_cells/"),
    ("TRAINING_CELL.py", "notebook_cells/"),
]

# Execute moves
moved = 0
skipped = 0
for src, dst in moves:
    src_path = base / src
    dst_path = base / dst / src
    
    if src_path.exists():
        try:
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ {src:50s} → {dst}")
            moved += 1
        except Exception as e:
            print(f"  ⚠️  {src:50s} - Error: {e}")
            skipped += 1
    else:
        print(f"  ⏭️  {src:50s} - Not found")
        skipped += 1

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ Moved: {moved} files")
print(f"⏭️  Skipped: {skipped} files")
print()
print("📁 New structure:")
print()
for folder in folders.keys():
    files = list((base / folder).glob("*"))
    print(f"  {folder}/ ({len(files)} files)")

print()
print("=" * 80)
print("✅ REORGANIZATION COMPLETE!")
print("=" * 80)
print()
print("Keep in root:")
print("  • rlvr_training_pipeline.ipynb (main notebook)")
print("  • results/ (training outputs)")
print("  • SELF_reference/ (reference implementation)")
print("  • .kiro/ (IDE configuration)")
print("  • reorganize_project.py (this script)")
