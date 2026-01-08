"""
Build Evaluation Index
======================
Creates all Step-2 evaluation artifacts:
  1. split_manifest.json   - Lists all filenames per split (reproducibility)
  2. test_index.json       - Ground truth + occlusion difficulty for each test image
  3. difficulty_summary.csv - Statistics per difficulty level

Occlusion Difficulty Definition (Core Novelty):
-----------------------------------------------
Difficulty is based on MAXIMUM pairwise IoU between ground-truth boxes:
  - Easy:   max_iou < 0.05  (no/minimal overlap)
  - Medium: 0.05 <= max_iou < 0.15 (partial overlap)
  - Hard:   max_iou >= 0.15 (significant overlap/occlusion)

Usage:
    python scripts/build_evaluation_index.py --dataset_root data/raw --output_dir data/processed

Arguments:
    --dataset_root : Path to raw dataset (default: data/raw)
    --output_dir   : Path to output directory (default: data/processed)
    --seed         : Random seed for reproducibility (default: 42)

This script does NOT:
- Move or copy image files
- Run any model predictions
- Require GPU

Note:
    This script is part of Step 2 (Data Pipeline & Evaluation Foundations).
    After running, Step 2 is FROZEN. Do not regenerate these files.
"""

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import yaml

# ==============================================================================
# OCCLUSION DIFFICULTY THRESHOLDS (CORE NOVELTY - DO NOT CHANGE AFTER STEP 2)
# ==============================================================================
# Difficulty is assigned based on MAXIMUM pairwise IoU between GT boxes:
#   - Easy:   max_iou < EASY_THRESHOLD     (no significant overlap)
#   - Medium: EASY_THRESHOLD <= max_iou < HARD_THRESHOLD (partial overlap)  
#   - Hard:   max_iou >= HARD_THRESHOLD    (significant occlusion)
# ==============================================================================
EASY_THRESHOLD = 0.05
HARD_THRESHOLD = 0.15
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build evaluation index with occlusion difficulty labels"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory (default: data/raw)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Path to output directory (default: data/processed)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def get_image_files(directory: Path) -> List[str]:
    """Get sorted list of image filenames in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if not directory.exists():
        return []
    images = sorted([
        f.name for f in directory.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    return images


def load_class_names(dataset_dir: Path) -> Dict[int, str]:
    """Load class names from data.yaml."""
    yaml_path = dataset_dir / "data.yaml"
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    names = config.get('names', {})
    
    # Handle both list and dict formats
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    else:
        return {}


def parse_yolo_label(label_path: Path) -> List[Dict]:
    """
    Parse a YOLO format label file.
    
    YOLO format: class_id x_center y_center width height (normalized 0-1)
    
    Returns:
        List of dicts with class_id and bbox_yolo
    """
    boxes = []
    
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append({
                    'class_id': int(parts[0]),
                    'bbox_yolo': [float(parts[1]), float(parts[2]), 
                                  float(parts[3]), float(parts[4])]
                })
    
    return boxes


def yolo_to_xyxy(bbox_yolo: List[float], img_size: int = 640) -> List[int]:
    """
    Convert YOLO format to xyxy pixel coordinates.
    
    Args:
        bbox_yolo: [x_center, y_center, width, height] normalized
        img_size: Image size (assumes square)
    
    Returns:
        [x1, y1, x2, y2] in pixels
    """
    x_center, y_center, width, height = bbox_yolo
    
    x1 = int((x_center - width / 2) * img_size)
    y1 = int((y_center - height / 2) * img_size)
    x2 = int((x_center + width / 2) * img_size)
    y2 = int((y_center + height / 2) * img_size)
    
    return [x1, y1, x2, y2]


def compute_iou(box1_yolo: List[float], box2_yolo: List[float]) -> float:
    """
    Compute IoU between two YOLO format boxes.
    
    Args:
        box1_yolo, box2_yolo: [x_center, y_center, width, height] normalized
    
    Returns:
        IoU value between 0 and 1
    """
    # Convert to corners
    x1_1 = box1_yolo[0] - box1_yolo[2] / 2
    y1_1 = box1_yolo[1] - box1_yolo[3] / 2
    x2_1 = box1_yolo[0] + box1_yolo[2] / 2
    y2_1 = box1_yolo[1] + box1_yolo[3] / 2
    
    x1_2 = box2_yolo[0] - box2_yolo[2] / 2
    y1_2 = box2_yolo[1] - box2_yolo[3] / 2
    x2_2 = box2_yolo[0] + box2_yolo[2] / 2
    y2_2 = box2_yolo[1] + box2_yolo[3] / 2
    
    # Intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    
    # Union
    area1 = box1_yolo[2] * box1_yolo[3]
    area2 = box2_yolo[2] * box2_yolo[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_occlusion_stats(boxes: List[Dict]) -> Dict:
    """
    Calculate occlusion statistics for an image.
    
    Args:
        boxes: List of box dicts with 'bbox_yolo' key
    
    Returns:
        Dict with pairwise_ious, avg_iou, max_iou, num_overlapping_pairs
    """
    if len(boxes) <= 1:
        return {
            'pairwise_ious': [],
            'avg_iou': 0.0,
            'max_iou': 0.0,
            'num_overlapping_pairs': 0
        }
    
    pairwise_ious = []
    
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = compute_iou(boxes[i]['bbox_yolo'], boxes[j]['bbox_yolo'])
            pairwise_ious.append(round(iou, 4))
    
    num_overlapping = sum(1 for iou in pairwise_ious if iou > 0)
    
    return {
        'pairwise_ious': pairwise_ious,
        'avg_iou': round(sum(pairwise_ious) / len(pairwise_ious), 4) if pairwise_ious else 0.0,
        'max_iou': round(max(pairwise_ious), 4) if pairwise_ious else 0.0,
        'num_overlapping_pairs': num_overlapping
    }


def assign_difficulty(max_iou: float) -> str:
    """
    Assign difficulty label based on MAXIMUM pairwise IoU.
    
    Thresholds (defined at top of file):
        - Easy:   max_iou < EASY_THRESHOLD (0.05)
        - Medium: EASY_THRESHOLD <= max_iou < HARD_THRESHOLD (0.05 - 0.15)
        - Hard:   max_iou >= HARD_THRESHOLD (0.15+)
    
    Args:
        max_iou: Maximum pairwise IoU between any two GT boxes
    
    Returns:
        'easy', 'medium', or 'hard'
    """
    if max_iou < EASY_THRESHOLD:
        return 'easy'
    elif max_iou < HARD_THRESHOLD:
        return 'medium'
    else:
        return 'hard'


def build_test_index(dataset_dir: Path, class_names: Dict[int, str]) -> Dict:
    """
    Build the complete evaluation index.
    
    Args:
        dataset_dir: Path to dataset root
        class_names: Mapping of class_id to class_name
    
    Returns:
        Complete index dict ready to save as JSON
    """
    test_images_dir = dataset_dir / "test" / "images"
    test_labels_dir = dataset_dir / "test" / "labels"
    
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images not found at {test_images_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in test_images_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    print(f"Found {len(image_files)} test images")
    
    # Process each image
    images_data = []
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    total_objects = 0
    class_counts = {}
    
    for img_path in image_files:
        # Find corresponding label
        label_path = test_labels_dir / (img_path.stem + ".txt")
        
        # Parse ground truth boxes
        boxes = parse_yolo_label(label_path)
        
        # Add class names and xyxy coordinates
        for box in boxes:
            box['class_name'] = class_names.get(box['class_id'], f"class_{box['class_id']}")
            box['bbox_xyxy'] = yolo_to_xyxy(box['bbox_yolo'])
            
            # Count classes
            class_id = box['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Calculate occlusion stats
        occlusion_stats = calculate_occlusion_stats(boxes)
        
        # Assign difficulty based on MAX IoU (not average!)
        difficulty = assign_difficulty(occlusion_stats['max_iou'])
        difficulty_counts[difficulty] += 1
        
        total_objects += len(boxes)
        
        # Build image entry
        image_entry = {
            'image_id': img_path.stem,
            'image_filename': img_path.name,
            'ground_truth': boxes,
            'num_objects': len(boxes),
            'occlusion_stats': occlusion_stats,
            'difficulty': difficulty
        }
        
        images_data.append(image_entry)
    
    # Compute difficulty statistics for summary
    difficulty_stats = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_images = [img for img in images_data if img['difficulty'] == diff]
        if diff_images:
            avg_objects = sum(img['num_objects'] for img in diff_images) / len(diff_images)
            max_ious = [img['occlusion_stats']['max_iou'] for img in diff_images]
            difficulty_stats[diff] = {
                'count': len(diff_images),
                'avg_num_objects': round(avg_objects, 2),
                'avg_max_iou': round(sum(max_ious) / len(max_ious), 4),
                'max_max_iou': round(max(max_ious), 4)
            }
        else:
            difficulty_stats[diff] = {
                'count': 0,
                'avg_num_objects': 0,
                'avg_max_iou': 0,
                'max_max_iou': 0
            }
    
    # Build metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'dataset_version': 'food-ingredients-dataset-2-rewtd-v1',
        'num_images': len(images_data),
        'num_classes': len(class_names),
        'total_objects': total_objects,
        'class_names': {str(k): v for k, v in class_names.items()},
        'class_distribution': {
            class_names.get(k, f"class_{k}"): v 
            for k, v in sorted(class_counts.items())
        },
        'difficulty_distribution': difficulty_counts,
        'difficulty_thresholds': {
            'easy': f'max_iou < {EASY_THRESHOLD}',
            'medium': f'{EASY_THRESHOLD} <= max_iou < {HARD_THRESHOLD}',
            'hard': f'max_iou >= {HARD_THRESHOLD}'
        },
        'difficulty_stats': difficulty_stats
    }
    
    return {
        'metadata': metadata,
        'images': images_data
    }


def build_split_manifest(dataset_dir: Path, seed: int) -> Dict:
    """
    Build the split manifest listing all filenames per split.
    
    Args:
        dataset_dir: Path to dataset root
        seed: Random seed used (for documentation)
    
    Returns:
        Split manifest dict
    """
    manifest = {
        'created_at': datetime.now().isoformat(),
        'dataset_version': 'food-ingredients-dataset-2-rewtd-v1',
        'random_seed': seed,
        'splits': {}
    }
    
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_dir / split / 'images'
        filenames = get_image_files(images_dir)
        manifest['splits'][split] = {
            'count': len(filenames),
            'filenames': filenames
        }
    
    return manifest


def save_difficulty_summary(test_index: Dict, output_path: Path):
    """
    Save difficulty statistics as CSV.
    
    Args:
        test_index: The complete test index
        output_path: Path to save CSV
    """
    stats = test_index['metadata']['difficulty_stats']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['difficulty', 'count', 'avg_num_objects', 'avg_max_iou', 'max_max_iou'])
        for diff in ['easy', 'medium', 'hard']:
            s = stats[diff]
            writer.writerow([diff, s['count'], s['avg_num_objects'], s['avg_max_iou'], s['max_max_iou']])


def main():
    args = parse_args()
    
    dataset_dir = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    seed = args.seed
    
    print("=" * 60)
    print("BUILD EVALUATION INDEX (Step 2)")
    print("=" * 60)
    print(f"  Dataset root: {dataset_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Seed:         {seed}")
    print("=" * 60)
    
    # Validate dataset exists
    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. Run download_dataset.py first."
        )
    
    # Load class names
    print("\n1. Loading class names...")
    class_names = load_class_names(dataset_dir)
    print(f"   Found {len(class_names)} classes")
    
    # Create output directories
    splits_dir = output_dir / "splits"
    evaluation_dir = output_dir / "evaluation"
    splits_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    # Build split manifest
    print("\n2. Building split manifest...")
    manifest = build_split_manifest(dataset_dir, seed)
    manifest_path = splits_dir / "split_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"   Train: {manifest['splits']['train']['count']} images")
    print(f"   Valid: {manifest['splits']['valid']['count']} images")
    print(f"   Test:  {manifest['splits']['test']['count']} images")
    print(f"   Saved: {manifest_path}")
    
    # Build test index
    print("\n3. Building test index with occlusion difficulty...")
    test_index = build_test_index(dataset_dir, class_names)
    index_path = evaluation_dir / "test_index.json"
    with open(index_path, 'w') as f:
        json.dump(test_index, f, indent=2)
    print(f"   Saved: {index_path}")
    
    # Save difficulty summary CSV
    print("\n4. Saving difficulty summary...")
    summary_path = evaluation_dir / "difficulty_summary.csv"
    save_difficulty_summary(test_index, summary_path)
    print(f"   Saved: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("STEP 2 ARTIFACTS COMPLETE")
    print("=" * 60)
    print(f"\nDifficulty Distribution (based on MAX pairwise IoU):")
    print(f"  Thresholds: Easy < {EASY_THRESHOLD}, Medium < {HARD_THRESHOLD}, Hard >= {HARD_THRESHOLD}")
    for diff, count in test_index['metadata']['difficulty_distribution'].items():
        pct = 100 * count / test_index['metadata']['num_images']
        print(f"    {diff.upper():8s}: {count:4d} ({pct:.1f}%)")
    
    print(f"\nFiles created:")
    print(f"  ✓ {manifest_path}")
    print(f"  ✓ {index_path}")
    print(f"  ✓ {summary_path}")
    
    print("\n" + "=" * 60)
    print("STEP 2 FROZEN - Do not regenerate these files")
    print("=" * 60)


if __name__ == "__main__":
    main()
