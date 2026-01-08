"""
Build Evaluation Index
======================
Creates a single JSON file containing all test image metadata, ground truth boxes,
and occlusion difficulty labels. This becomes the "source of truth" for all 
downstream analyses (calibration, visualization, hallucination study).

Usage:
    python scripts/build_evaluation_index.py

Output:
    data/processed/test_index.json

This script does NOT:
- Move or copy image files
- Run any model predictions
- Require GPU
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Occlusion thresholds (tunable)
EASY_MAX_IOU = 0.05
MEDIUM_MAX_IOU = 0.15


def find_dataset_dir() -> Path:
    """Find the dataset directory (prefers processed, falls back to raw)."""
    # First check processed directory (after split_dataset.py)
    if (DATA_PROCESSED_DIR / "data.yaml").exists():
        return DATA_PROCESSED_DIR
    
    # Fall back to raw directory (Roboflow creates a subdirectory)
    for subdir in DATA_RAW_DIR.iterdir():
        if subdir.is_dir() and (subdir / "data.yaml").exists():
            return subdir
    
    # Check if data.yaml is directly in raw
    if (DATA_RAW_DIR / "data.yaml").exists():
        return DATA_RAW_DIR
    
    raise FileNotFoundError(
        f"Dataset not found. Run download_dataset.py and split_dataset.py first."
    )


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


def assign_difficulty(avg_iou: float) -> str:
    """Assign difficulty label based on average IoU."""
    if avg_iou <= EASY_MAX_IOU:
        return 'easy'
    elif avg_iou <= MEDIUM_MAX_IOU:
        return 'medium'
    else:
        return 'hard'


def build_index(dataset_dir: Path, class_names: Dict[int, str]) -> Dict:
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
        
        # Assign difficulty
        difficulty = assign_difficulty(occlusion_stats['avg_iou'])
        difficulty_counts[difficulty] += 1
        
        total_objects += len(boxes)
        
        # Build image entry
        image_entry = {
            'image_id': img_path.stem,
            'image_path': str(img_path.relative_to(PROJECT_ROOT)),
            'label_path': str(label_path.relative_to(PROJECT_ROOT)) if label_path.exists() else None,
            'ground_truth': boxes,
            'num_objects': len(boxes),
            'occlusion_stats': occlusion_stats,
            'difficulty': difficulty
        }
        
        images_data.append(image_entry)
    
    # Build metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'dataset_path': str(dataset_dir.relative_to(PROJECT_ROOT)),
        'num_images': len(images_data),
        'num_classes': len(class_names),
        'total_objects': total_objects,
        'class_names': {str(k): v for k, v in class_names.items()},
        'class_distribution': {
            class_names.get(k, f"class_{k}"): v 
            for k, v in sorted(class_counts.items())
        },
        'difficulty_distribution': difficulty_counts,
        'thresholds': {
            'easy_max_iou': EASY_MAX_IOU,
            'medium_max_iou': MEDIUM_MAX_IOU
        }
    }
    
    return {
        'metadata': metadata,
        'images': images_data
    }


def main():
    print("=" * 60)
    print("BUILD EVALUATION INDEX")
    print("=" * 60)
    
    # Find dataset
    print("\n1. Finding dataset...")
    dataset_dir = find_dataset_dir()
    print(f"   Found: {dataset_dir}")
    
    # Load class names
    print("\n2. Loading class names...")
    class_names = load_class_names(dataset_dir)
    print(f"   Found {len(class_names)} classes")
    
    # Build index
    print("\n3. Processing test images...")
    index = build_index(dataset_dir, class_names)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total images: {index['metadata']['num_images']}")
    print(f"  Total objects: {index['metadata']['total_objects']}")
    print(f"  Classes: {index['metadata']['num_classes']}")
    print(f"\n  Difficulty distribution:")
    for diff, count in index['metadata']['difficulty_distribution'].items():
        pct = 100 * count / index['metadata']['num_images']
        print(f"    {diff.upper():8s}: {count:4d} ({pct:.1f}%)")
    
    # Save index
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED_DIR / "test_index.json"
    
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\n✅ Saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Train models (scripts/train_models.py)")
    print("  2. Run evaluation (scripts/evaluate_models.py)")


if __name__ == "__main__":
    main()
