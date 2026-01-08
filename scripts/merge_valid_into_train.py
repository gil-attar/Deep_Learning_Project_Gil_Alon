"""
Merge Valid into Train (Phase 2)
================================
For final model training, merge the validation set into training.
This gives us 80% train / 20% test while keeping the SAME test set.

The test set NEVER changes - this is critical for fair evaluation.

Usage:
    python scripts/merge_valid_into_train.py

Before:
    train: 70% (1386 images)
    valid: 10% (198 images)  
    test:  20% (396 images)

After:
    train: 80% (1584 images) = original train + valid
    valid: 0% (empty)
    test:  20% (396 images) - UNCHANGED!
"""

import shutil
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def merge_valid_into_train():
    """Merge validation set into training set."""
    
    train_dir = DATA_PROCESSED_DIR / "train"
    valid_dir = DATA_PROCESSED_DIR / "valid"
    
    if not valid_dir.exists():
        print("Valid directory not found. Already merged?")
        return
    
    print("=" * 60)
    print("MERGING VALID INTO TRAIN (Phase 2)")
    print("=" * 60)
    
    # Count before
    train_images = list((train_dir / "images").glob("*"))
    valid_images = list((valid_dir / "images").glob("*"))
    test_images = list((DATA_PROCESSED_DIR / "test" / "images").glob("*"))
    
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
    print(f"  Train: {len(train_images_after)} images (+{moved_count})")
    print(f"  Valid: 0 images (merged into train)")
    print(f"  Test:  {len(test_images)} images (unchanged)")
    
    print("\n" + "=" * 60)
    print("✓ Phase 2 split ready! (80% train / 20% test)")
    print("=" * 60)


if __name__ == "__main__":
    merge_valid_into_train()
