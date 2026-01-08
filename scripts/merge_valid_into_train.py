"""
Merge Valid into Train (Phase 2) - OPTIONAL LATER EXPERIMENT
=============================================================
** NOT PART OF STEP 2 - Do not run until Step 4+ **

For final model training, merge the validation set into training.
This gives us 80% train / 20% test while keeping the SAME test set.

The test set NEVER changes - this is critical for fair evaluation.

When to use:
    - After baseline training is complete (Step 3)
    - After hyperparameter tuning on 70/10/20 split (Step 6)
    - For final model training before evaluation (Step 7+)

Usage:
    python scripts/merge_valid_into_train.py --dataset_root data/raw

Arguments:
    --dataset_root : Path to dataset (default: data/raw)

Before:
    train: 70% (1386 images)
    valid: 10% (198 images)  
    test:  20% (396 images)

After:
    train: 80% (1584 images) = original train + valid
    valid: 0% (empty)
    test:  20% (396 images) - UNCHANGED!
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge validation set into training set (Phase 2)"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/raw",
        help="Path to dataset directory (default: data/raw)"
    )
    return parser.parse_args()


def merge_valid_into_train(dataset_root: str):
    """Merge validation set into training set."""
    
    dataset_dir = Path(dataset_root)
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"
    
    if not valid_dir.exists():
        print("Valid directory not found. Already merged?")
        return
    
    print("=" * 60)
    print("MERGING VALID INTO TRAIN (Phase 2)")
    print("=" * 60)
    print("\n⚠️  WARNING: This modifies data/raw/ which should be immutable!")
    print("    Only run this after completing Step 3 (baseline training).")
    
    # Count before
    train_images = list((train_dir / "images").glob("*"))
    valid_images = list((valid_dir / "images").glob("*"))
    test_images = list((test_dir / "images").glob("*"))
    
    print(f"\nBefore merge:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Valid: {len(valid_images)} images")
    print(f"  Test:  {len(test_images)} images (unchanged)")
    
    # Move valid images to train
    valid_images_dir = valid_dir / "images"
    valid_labels_dir = valid_dir / "labels"
    train_images_dir = train_dir / "images"
    train_labels_dir = train_dir / "labels"
    
    moved_count = 0
    for img_path in valid_images_dir.glob("*"):
        # Move image
        shutil.move(str(img_path), str(train_images_dir / img_path.name))
        
        # Move corresponding label
        label_path = valid_labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(label_path), str(train_labels_dir / label_path.name))
        
        moved_count += 1
    
    # Remove empty valid directory
    shutil.rmtree(valid_dir)
    
    # Create empty valid directory (for compatibility)
    (valid_dir / "images").mkdir(parents=True, exist_ok=True)
    (valid_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Count after
    train_images_after = list((train_dir / "images").glob("*"))
    
    print(f"\nAfter merge:")
    print(f"  Train: {len(train_images_after)} images")
    print(f"  Valid: 0 images (empty)")
    print(f"  Test:  {len(test_images)} images (unchanged)")
    
    print(f"\n✓ Moved {moved_count} images from valid to train")
    print("\nPhase 2 merge complete. Ready for final model training.")


if __name__ == "__main__":
    args = parse_args()
    merge_valid_into_train(args.dataset_root)
