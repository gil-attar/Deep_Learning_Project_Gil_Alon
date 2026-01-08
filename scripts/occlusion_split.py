"""
Occlusion Scoring & Dataset Splitting
======================================
This module implements our first novelty:
Splitting the test set into Easy/Medium/Hard based on bounding box overlap (IoU).

Hypothesis: Transformer models (RT-DETR) should perform better on "Hard" images
because they use global attention, while CNNs (YOLO) may struggle with occlusion.

Usage:
    python scripts/occlusion_split.py

Output:
    Creates data/processed/ with:
    - test_easy/    (images with low overlap)
    - test_medium/  (images with moderate overlap)  
    - test_hard/    (images with high overlap)
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Occlusion thresholds (tunable hyperparameters)
EASY_THRESHOLD = 0.05      # Max avg IoU for "easy"
MEDIUM_THRESHOLD = 0.15    # Max avg IoU for "medium"
# Above MEDIUM_THRESHOLD = "hard"


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a YOLO format label file.
    
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    
    Returns:
        List of (class_id, x_center, y_center, width, height)
    """
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append((class_id, x_center, y_center, width, height))
    
    return boxes


def yolo_to_corners(box: Tuple[int, float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert YOLO format (center, width, height) to corner format (x1, y1, x2, y2).
    
    Args:
        box: (class_id, x_center, y_center, width, height)
    
    Returns:
        (x1, y1, x2, y2) - top-left and bottom-right corners
    """
    _, x_center, y_center, width, height = box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return (x1, y1, x2, y2)


def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1, box2: (x1, y1, x2, y2) format
    
    Returns:
        IoU value between 0 and 1
    """
    # Get intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_occlusion_score(boxes: List[Tuple[int, float, float, float, float]]) -> float:
    """
    Calculate the occlusion score for an image based on pairwise IoU.
    
    Higher score = more overlapping objects = harder image.
    
    Args:
        boxes: List of YOLO format boxes
    
    Returns:
        Average pairwise IoU (0 if single or no objects)
    """
    if len(boxes) <= 1:
        return 0.0
    
    # Convert all boxes to corner format
    corner_boxes = [yolo_to_corners(box) for box in boxes]
    
    # Calculate pairwise IoU
    total_iou = 0.0
    num_pairs = 0
    
    for i in range(len(corner_boxes)):
        for j in range(i + 1, len(corner_boxes)):
            iou = compute_iou(corner_boxes[i], corner_boxes[j])
            total_iou += iou
            num_pairs += 1
    
    # Return average IoU
    return total_iou / num_pairs if num_pairs > 0 else 0.0


def classify_difficulty(occlusion_score: float) -> str:
    """
    Classify an image into difficulty category based on occlusion score.
    
    Args:
        occlusion_score: Average pairwise IoU
    
    Returns:
        'easy', 'medium', or 'hard'
    """
    if occlusion_score <= EASY_THRESHOLD:
        return 'easy'
    elif occlusion_score <= MEDIUM_THRESHOLD:
        return 'medium'
    else:
        return 'hard'


def process_split(split_name: str = "test") -> Dict[str, List[Path]]:
    """
    Process a dataset split and classify images by occlusion difficulty.
    
    Args:
        split_name: 'train', 'valid', or 'test'
    
    Returns:
        Dictionary mapping difficulty -> list of image paths
    """
    # Find the dataset directory (Roboflow creates a subdirectory)
    raw_dirs = list(DATA_RAW_DIR.glob("*"))
    dataset_dir = None
    for d in raw_dirs:
        if d.is_dir() and (d / split_name / "images").exists():
            dataset_dir = d
            break
    
    if dataset_dir is None:
        # Try direct structure
        if (DATA_RAW_DIR / split_name / "images").exists():
            dataset_dir = DATA_RAW_DIR
        else:
            raise FileNotFoundError(f"Cannot find {split_name} split in {DATA_RAW_DIR}")
    
    images_dir = dataset_dir / split_name / "images"
    labels_dir = dataset_dir / split_name / "labels"
    
    print(f"\nProcessing {split_name} split from: {dataset_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"  Found {len(image_files)} images")
    
    # Classify each image
    results = {'easy': [], 'medium': [], 'hard': []}
    scores = []
    
    for img_path in tqdm(image_files, desc="Calculating occlusion scores"):
        # Find corresponding label file
        label_path = labels_dir / (img_path.stem + ".txt")
        
        # Parse boxes and calculate score
        boxes = parse_yolo_label(label_path)
        score = calculate_occlusion_score(boxes)
        scores.append(score)
        
        # Classify
        difficulty = classify_difficulty(score)
        results[difficulty].append(img_path)
    
    # Print statistics
    print(f"\n  Occlusion Score Statistics:")
    print(f"    Min: {min(scores):.4f}")
    print(f"    Max: {max(scores):.4f}")
    print(f"    Mean: {np.mean(scores):.4f}")
    print(f"    Median: {np.median(scores):.4f}")
    
    print(f"\n  Difficulty Distribution:")
    for diff, imgs in results.items():
        print(f"    {diff.upper()}: {len(imgs)} images ({100*len(imgs)/len(image_files):.1f}%)")
    
    return results, dataset_dir


def create_difficulty_splits(results: Dict[str, List[Path]], 
                             source_dataset_dir: Path,
                             split_name: str = "test"):
    """
    Create separate directories for each difficulty level.
    
    Args:
        results: Dictionary from process_split()
        source_dataset_dir: Path to the original dataset
        split_name: 'train', 'valid', or 'test'
    """
    labels_dir = source_dataset_dir / split_name / "labels"
    
    for difficulty, image_paths in results.items():
        # Create output directories
        out_images_dir = DATA_PROCESSED_DIR / f"{split_name}_{difficulty}" / "images"
        out_labels_dir = DATA_PROCESSED_DIR / f"{split_name}_{difficulty}" / "labels"
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {difficulty} images to {out_images_dir.parent}")
        
        for img_path in tqdm(image_paths, desc=f"  {difficulty}"):
            # Copy image
            shutil.copy2(img_path, out_images_dir / img_path.name)
            
            # Copy label
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, out_labels_dir / (img_path.stem + ".txt"))
    
    print(f"\n✅ Created difficulty splits in {DATA_PROCESSED_DIR}")


def create_data_yaml():
    """
    Create the data.yaml file for training.
    Uses the original train/valid splits, but we have difficulty-split test sets.
    """
    # Find the dataset directory
    raw_dirs = list(DATA_RAW_DIR.glob("*"))
    dataset_dir = None
    for d in raw_dirs:
        if d.is_dir() and (d / "data.yaml").exists():
            dataset_dir = d
            break
    
    if dataset_dir is None:
        if (DATA_RAW_DIR / "data.yaml").exists():
            dataset_dir = DATA_RAW_DIR
        else:
            print("Warning: Could not find original data.yaml")
            return
    
    # Read original data.yaml to get class names
    import yaml
    with open(dataset_dir / "data.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    # Create new data.yaml pointing to our structure
    new_config = {
        'path': str(DATA_RAW_DIR.absolute()),
        'train': str((dataset_dir / "train" / "images").absolute()),
        'val': str((dataset_dir / "valid" / "images").absolute()),
        'test': str((dataset_dir / "test" / "images").absolute()),
        'names': original_config.get('names', {}),
        'nc': original_config.get('nc', len(original_config.get('names', {})))
    }
    
    # Save to project root
    output_path = PROJECT_ROOT / "data.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    print(f"\n✅ Created {output_path}")
    print(f"   Classes: {new_config['nc']}")


def main():
    """Main function to run occlusion-based dataset splitting."""
    print("=" * 60)
    print("OCCLUSION-BASED DATASET SPLITTING")
    print("=" * 60)
    print(f"\nThresholds:")
    print(f"  Easy:   IoU <= {EASY_THRESHOLD}")
    print(f"  Medium: {EASY_THRESHOLD} < IoU <= {MEDIUM_THRESHOLD}")
    print(f"  Hard:   IoU > {MEDIUM_THRESHOLD}")
    
    # Process test split
    results, dataset_dir = process_split("test")
    
    # Create difficulty-based directories
    create_difficulty_splits(results, dataset_dir, "test")
    
    # Create data.yaml
    create_data_yaml()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the splits in data/processed/")
    print("  2. Train models using: python scripts/train_models.py")


if __name__ == "__main__":
    main()
