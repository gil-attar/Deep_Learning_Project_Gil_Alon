"""
Verification script to check Step 3 setup is ready.

Run this before training to ensure everything is configured correctly.
"""

import sys
from pathlib import Path
import json


def check_file_exists(path, required=True):
    """Check if a file exists and return status."""
    exists = Path(path).exists()
    status = "✓" if exists else ("✗" if required else "⚠")
    req_str = "(required)" if required else "(optional)"
    print(f"  {status} {path} {req_str}")
    return exists


def check_directory_exists(path, required=True):
    """Check if a directory exists and return status."""
    exists = Path(path).is_dir()
    status = "✓" if exists else ("✗" if required else "⚠")
    req_str = "(required)" if required else "(will be created)"
    print(f"  {status} {path}/ {req_str}")
    return exists


def verify_test_index(path):
    """Verify test_index.json structure."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)

        num_images = len(data.get('images', []))
        has_difficulty = all('difficulty' in img for img in data.get('images', []))

        print(f"    - {num_images} test images")
        print(f"    - Difficulty labels: {'✓' if has_difficulty else '✗'}")

        if 'difficulty_distribution' in data.get('metadata', {}):
            dist = data['metadata']['difficulty_distribution']
            print(f"    - Easy: {dist.get('easy', 0)}, Medium: {dist.get('medium', 0)}, Hard: {dist.get('hard', 0)}")

        return num_images > 0 and has_difficulty
    except Exception as e:
        print(f"    ✗ Error reading file: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("STEP 3 SETUP VERIFICATION")
    print("="*60)

    all_checks_passed = True

    # 1. Check repository structure
    print("\n[1] Repository Structure:")
    required_dirs = [
        "data/processed/splits",
        "data/processed/evaluation",
        "scripts",
        "notebooks",
        "evaluation/metrics",
    ]

    for dir_path in required_dirs:
        if not check_directory_exists(dir_path, required=True):
            all_checks_passed = False

    # Optional directories (will be created)
    check_directory_exists("models", required=False)
    check_directory_exists("evaluation/plots", required=False)

    # 2. Check required data files (Step 2 artifacts)
    print("\n[2] Step 2 Data Files (Must exist from previous step):")
    required_files = [
        "data/processed/splits/split_manifest.json",
        "data/processed/evaluation/test_index.json",
    ]

    for file_path in required_files:
        if not check_file_exists(file_path, required=True):
            all_checks_passed = False

    # Verify test_index.json structure
    print("\n  Validating test_index.json:")
    if not verify_test_index("data/processed/evaluation/test_index.json"):
        all_checks_passed = False

    # 3. Check scripts exist
    print("\n[3] Evaluation Scripts:")
    script_files = [
        "scripts/evaluate_baseline.py",
        "scripts/create_data_yaml.py",
    ]

    for file_path in script_files:
        if not check_file_exists(file_path, required=True):
            all_checks_passed = False

    # 4. Check notebook exists
    print("\n[4] Training Notebook:")
    if not check_file_exists("notebooks/02_train_models.ipynb", required=True):
        all_checks_passed = False

    # 5. Check Python dependencies
    print("\n[5] Python Dependencies:")
    try:
        import ultralytics
        print(f"  ✓ ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  ✗ ultralytics (pip install ultralytics)")
        all_checks_passed = False

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_str = f"CUDA {torch.version.cuda}" if cuda_available else "CPU only"
        print(f"  ✓ torch {torch.__version__} ({cuda_str})")
        if not cuda_available:
            print("    ⚠ No GPU detected - training will be slow!")
    except ImportError:
        print("  ✗ torch (pip install torch)")
        all_checks_passed = False

    try:
        import yaml
        print("  ✓ pyyaml")
    except ImportError:
        print("  ✗ pyyaml (pip install pyyaml)")
        all_checks_passed = False

    # 6. Check dataset (optional - downloaded in Colab)
    print("\n[6] Dataset (Optional for local, downloaded in Colab):")
    dataset_exists = check_directory_exists("data/raw/test/images", required=False)
    if dataset_exists:
        import os
        num_images = len(list(Path("data/raw/test/images").glob("*.jpg")))
        print(f"    - {num_images} test images found")
    else:
        print("    ⚠ Dataset not downloaded (will download in Colab)")

    # 7. Summary
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ ALL REQUIRED CHECKS PASSED")
        print("="*60)
        print("\nYou're ready to run Step 3!")
        print("\nNext steps:")
        print("  1. Open notebooks/02_train_models.ipynb in Google Colab")
        print("  2. Set your ROBOFLOW_API_KEY in cell 4")
        print("  3. Run all cells sequentially")
        print("  4. Download generated JSON files and weights")
        print()
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  - Missing Step 2 files: Run build_evaluation_index.py first")
        print("  - Missing scripts: Ensure you're in the project root directory")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
