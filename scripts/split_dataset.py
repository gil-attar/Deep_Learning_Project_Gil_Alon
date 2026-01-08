"""
Dataset Split Script
====================
Combines all images from Roboflow download and re-splits with custom ratios.

Phase 1 (hyperparameter tuning):
    python scripts/split_dataset.py
    → 70% train / 10% valid / 20% test

Phase 2 (final training):
    python scripts/split_dataset.py --final
    → 80% train / 20% test (valid merged into train)

Output:
    data/processed/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Fixed seed for reproducibility
RANDOM_SEED = 42

# Split ratios
PHASE1_RATIOS = (0.70, 0.10, 0.20)  # train, valid, test
PHASE2_RATIOS = (0.80, 0.00, 0.20)  # train, (no valid), test


def find_dataset_dir() -> Path:
    """Find the downloaded dataset directory."""
    for subdir in DATA_RAW_DIR.iterdir():
        if subdir.is_dir() and (subdir / "data.yaml").exists():
            return subdir
    
    if (DATA_RAW_DIR / "data.yaml").exists():
        return DATA_RAW_DIR
    
    raise FileNotFoundError(
        f"Dataset not found in {DATA_RAW_DIR}. "
        "Run download_dataset.py first."
    )


def collect_all_images(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Collect all image-label pairs from all splits.
    
    Returns:
        List of (image_path, label_path) tuples
    """
    all_pairs = []
    
    for split in ["train", "valid", "test"]:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        
        if not images_dir.exists():
            continue
        
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                label_path = labels_dir / (img_path.stem + ".txt")
                all_pairs.append((img_path, label_path))
    
    return all_pairs


def split_data(pairs: List[Tuple[Path, Path]], 
               ratios: Tuple[float, float, float],
               seed: int = 42) -> Tuple[List, List, List]:
    """
    Split data into train/valid/test based on ratios.
    
    Args:
        pairs: List of (image_path, label_path)
        ratios: (train_ratio, valid_ratio, test_ratio)
        seed: Random seed for reproducibility
    
    Returns:
        (train_pairs, valid_pairs, test_pairs)
    """
    # Shuffle with fixed seed
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * ratios[0])
    valid_end = train_end + int(n * ratios[1])
    
    train_pairs = shuffled[:train_end]
    valid_pairs = shuffled[train_end:valid_end]
    test_pairs = shuffled[valid_end:]
    
    return train_pairs, valid_pairs, test_pairs


def copy_split(pairs: List[Tuple[Path, Path]], 
               output_dir: Path, 
               split_name: str):
    """Copy image-label pairs to output directory."""
    if not pairs:
        return
    
    images_out = output_dir / split_name / "images"
    labels_out = output_dir / split_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    for img_path, label_path in pairs:
        # Copy image
        shutil.copy2(img_path, images_out / img_path.name)
        
        # Copy label if exists
        if label_path.exists():
            shutil.copy2(label_path, labels_out / label_path.name)


def create_data_yaml(output_dir: Path, class_names: dict, nc: int):
    """Create data.yaml for the new split."""
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': nc,
        'names': class_names
    }
    
    with open(output_dir / "data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Split dataset with custom ratios")
    parser.add_argument('--final', action='store_true',
                        help='Use final split (80/0/20) instead of tuning split (70/10/20)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed for shuffling (default: {RANDOM_SEED})')
    parser.add_argument('--clean', action='store_true',
                        help='Remove existing processed data before splitting')
    
    args = parser.parse_args()
    seed = args.seed
    
    # Determine ratios
    if args.final:
        ratios = PHASE2_RATIOS
        print("=" * 60)
        print("FINAL SPLIT MODE")
        print("=" * 60)
        print("Ratios: 80% train / 0% valid / 20% test")
    else:
        ratios = PHASE1_RATIOS
        print("=" * 60)
        print("HYPERPARAMETER TUNING SPLIT")
        print("=" * 60)
        print("Ratios: 70% train / 10% valid / 20% test")
    
    print(f"Random seed: {seed}")
    
    # Find dataset
    print("\n1. Finding dataset...")
    dataset_dir = find_dataset_dir()
    print(f"   Found: {dataset_dir}")
    
    # Load class names
    with open(dataset_dir / "data.yaml", 'r') as f:
        config = yaml.safe_load(f)
    class_names = config.get('names', {})
    nc = config.get('nc', len(class_names))
    
    # Collect all images
    print("\n2. Collecting all images...")
    all_pairs = collect_all_images(dataset_dir)
    print(f"   Total: {len(all_pairs)} image-label pairs")
    
    # Clean output directory if requested
    if args.clean and DATA_PROCESSED_DIR.exists():
        print("\n3. Cleaning existing processed data...")
        for split in ["train", "valid", "test"]:
            split_dir = DATA_PROCESSED_DIR / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        print("   Done.")
    
    # Split data
    print("\n3. Splitting data...")
    train_pairs, valid_pairs, test_pairs = split_data(all_pairs, ratios, seed=seed)
    
    print(f"   Train: {len(train_pairs)} ({100*len(train_pairs)/len(all_pairs):.1f}%)")
    print(f"   Valid: {len(valid_pairs)} ({100*len(valid_pairs)/len(all_pairs):.1f}%)")
    print(f"   Test:  {len(test_pairs)} ({100*len(test_pairs)/len(all_pairs):.1f}%)")
    
    # Copy files
    print("\n4. Copying files to data/processed/...")
    copy_split(train_pairs, DATA_PROCESSED_DIR, "train")
    copy_split(valid_pairs, DATA_PROCESSED_DIR, "valid")
    copy_split(test_pairs, DATA_PROCESSED_DIR, "test")
    print("   Done.")
    
    # Create data.yaml
    print("\n5. Creating data.yaml...")
    create_data_yaml(DATA_PROCESSED_DIR, class_names, nc)
    print(f"   Saved: {DATA_PROCESSED_DIR / 'data.yaml'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {DATA_PROCESSED_DIR}")
    print(f"  ├── train/  ({len(train_pairs)} images)")
    print(f"  ├── valid/  ({len(valid_pairs)} images)")
    print(f"  ├── test/   ({len(test_pairs)} images)")
    print(f"  └── data.yaml")
    
    print("\nNext steps:")
    if args.final:
        print("  1. Train final models with: python scripts/train_models.py")
        print("  2. Evaluate on test set: python scripts/evaluate_models.py")
    else:
        print("  1. Build evaluation index: python scripts/build_evaluation_index.py")
        print("  2. Train models: python scripts/train_models.py")
        print("  3. Tune hyperparameters, then run: python scripts/split_dataset.py --final")


if __name__ == "__main__":
    main()
