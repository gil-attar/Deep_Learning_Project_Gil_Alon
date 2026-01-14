"""
Build Evaluation Indices for Train/Val/Test Splits
===================================================
Creates ground truth index files for all three splits:
  - train_index.json (1384 images)
  - val_index.json (200 images)
  - test_index.json (400 images)

Each index contains:
  - Image IDs and filenames
  - Ground truth bounding boxes
  - Class labels
  - (Optional) Difficulty labels based on occlusion IoU

Usage:
    python scripts/build_evaluation_indices.py \\
        --dataset_root data/raw \\
        --output_dir data/processed/evaluation

This script is run ONCE to freeze the evaluation protocol.
Do not regenerate these files after experiments begin.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build train/val/test evaluation indices"
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
        default="data/processed/evaluation",
        help="Path to output directory (default: data/processed/evaluation)"
    )
    return parser.parse_args()


def load_class_names(dataset_root: Path) -> Dict[int, str]:
    """Load class names from data.yaml."""
    # Try data/processed/data.yaml first (most common location)
    yaml_path = Path("data/processed/data.yaml")

    # Fallback to dataset_root/data.yaml
    if not yaml_path.exists():
        yaml_path = dataset_root / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found. Tried:\n"
            f"  - data/processed/data.yaml\n"
            f"  - {dataset_root}/data.yaml\n"
            f"Run: python scripts/create_data_yaml.py"
        )

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    names = config.get('names', {})

    # Convert to dict with integer keys
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    else:
        raise ValueError(f"Unexpected 'names' format in data.yaml: {type(names)}")


def parse_yolo_label(label_path: Path, image_width: int = 640, image_height: int = 640) -> List[Dict]:
    """
    Parse YOLO format label file.

    Args:
        label_path: Path to .txt label file
        image_width: Image width for bbox conversion (default: 640)
        image_height: Image height for bbox conversion (default: 640)

    Returns:
        List of bounding boxes with class_id and bbox_yolo/bbox_xyxy
    """
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            bbox_yolo = [float(x) for x in parts[1:5]]

            # Convert YOLO format to xyxy
            x_center, y_center, width, height = bbox_yolo
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)

            boxes.append({
                'class_id': class_id,
                'bbox_yolo': bbox_yolo,
                'bbox_xyxy': [x1, y1, x2, y2]
            })

    return boxes


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_max_pairwise_iou(boxes: List[Dict]) -> float:
    """
    Compute maximum pairwise IoU among all boxes.

    Used to determine occlusion difficulty:
    - Easy: max_iou < 0.10
    - Medium: 0.10 <= max_iou < 0.25
    - Hard: max_iou >= 0.25
    """
    if len(boxes) < 2:
        return 0.0

    max_iou = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = compute_iou(boxes[i]['bbox_xyxy'], boxes[j]['bbox_xyxy'])
            max_iou = max(max_iou, iou)

    return max_iou


def assign_difficulty(max_iou: float) -> str:
    """Assign difficulty label based on max pairwise IoU."""
    if max_iou < 0.10:
        return "easy"
    elif max_iou < 0.25:
        return "medium"
    else:
        return "hard"


def build_index_for_split(
    split_name: str,
    dataset_root: Path,
    class_names: Dict[int, str]
) -> Dict:
    """
    Build evaluation index for a single split.

    Args:
        split_name: "train", "val", or "test"
        dataset_root: Root directory of dataset
        class_names: Dict mapping class_id to class_name

    Returns:
        Index dictionary ready to save as JSON
    """
    split_dir = dataset_root / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Get all image files
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    print(f"\nProcessing {split_name} split: {len(image_files)} images...")

    images_data = []
    class_distribution = {name: 0 for name in class_names.values()}
    difficulty_distribution = {"easy": 0, "medium": 0, "hard": 0}
    total_objects = 0

    for img_file in image_files:
        # Image ID is filename without extension
        image_id = img_file.stem
        image_filename = img_file.name

        # Load ground truth from label file
        label_file = labels_dir / f"{image_id}.txt"
        ground_truth = parse_yolo_label(label_file)

        # Add class names to ground truth
        for obj in ground_truth:
            obj['class_name'] = class_names.get(obj['class_id'], f"class_{obj['class_id']}")
            class_distribution[obj['class_name']] += 1

        total_objects += len(ground_truth)

        # Compute occlusion difficulty
        max_iou = compute_max_pairwise_iou(ground_truth)
        difficulty = assign_difficulty(max_iou)
        difficulty_distribution[difficulty] += 1

        images_data.append({
            'image_id': image_id,
            'image_filename': image_filename,
            'ground_truth': ground_truth,
            'num_objects': len(ground_truth),
            'max_iou': round(max_iou, 4),
            'difficulty': difficulty
        })

    # Build index
    index = {
        'metadata': {
            'split': split_name,
            'created_at': datetime.now().isoformat(),
            'num_images': len(images_data),
            'num_classes': len(class_names),
            'total_objects': total_objects,
            'class_names': class_names,
            'class_distribution': class_distribution,
            'difficulty_distribution': difficulty_distribution
        },
        'images': images_data
    }

    return index


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILD EVALUATION INDICES (Train/Val/Test)")
    print("=" * 70)
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")

    # Load class names
    class_names = load_class_names(dataset_root)
    print(f"\nLoaded {len(class_names)} classes from data.yaml")

    # Build indices for all splits
    splits = ['train', 'valid', 'test']  # Note: 'valid' in data, but we'll name it 'val'

    for split in splits:
        # Use 'valid' for directory, but 'val' for output filename
        split_dir_name = split
        split_output_name = 'val' if split == 'valid' else split

        try:
            index = build_index_for_split(split_dir_name, dataset_root, class_names)

            # Save index
            output_path = output_dir / f"{split_output_name}_index.json"
            with open(output_path, 'w') as f:
                json.dump(index, f, indent=2)

            print(f"✓ Saved {split_output_name}_index.json")
            print(f"  - Images: {index['metadata']['num_images']}")
            print(f"  - Objects: {index['metadata']['total_objects']}")
            print(f"  - Difficulty: {index['metadata']['difficulty_distribution']}")

        except FileNotFoundError as e:
            print(f"⚠ Skipping {split}: {e}")

    print("\n" + "=" * 70)
    print("✓ All evaluation indices created!")
    print("=" * 70)
    print(f"\nGenerated files in {output_dir}/:")
    print("  - train_index.json")
    print("  - val_index.json")
    print("  - test_index.json")
    print("\nThese files are now FROZEN for reproducibility.")
    print("Do not regenerate unless absolutely necessary.")


if __name__ == "__main__":
    main()
