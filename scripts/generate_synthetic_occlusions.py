"""
Generate Synthetic Occlusions
=============================
Creates test sets with controlled synthetic occlusions for robustness evaluation.

For each test image, generates versions with bounding boxes partially covered
using a GRID-BASED approach at different occlusion levels (20%, 40%, 60%).

Grid-Based Occlusion Method:
---------------------------
Each bounding box is divided into a 10x10 grid (100 cells, each = 1% of bbox).
To achieve X% occlusion, X random cells are filled with black.

Benefits:
- Exact coverage (20% = exactly 20 cells filled)
- No overlap between occlusion rectangles
- Scattered pattern (more realistic than single rectangle)
- Deterministic with seed (reproducible)

This enables controlled comparison of model robustness:
- Same images, same objects
- Only variable: occlusion percentage
- Compare degradation curves: YOLOv8 vs RT-DETR

Usage:
    python scripts/generate_synthetic_occlusions.py --test_index data/processed/evaluation/test_index.json

Arguments:
    --test_index   : Path to test_index.json (default: data/processed/evaluation/test_index.json)
    --images_dir   : Path to test images (default: data/raw/test/images)
    --output_dir   : Output directory (default: data/synthetic_occlusion)
    --levels       : Occlusion levels as comma-separated values (default: 0.2,0.4,0.6)
    --seed         : Random seed for reproducibility (default: 42)

Output Structure:
    data/synthetic_occlusion/
    ├── level_020/          # 20% occlusion (20 cells per bbox)
    │   ├── images/
    │   └── labels/         # Same labels as original (object still there)
    ├── level_040/          # 40% occlusion (40 cells per bbox)
    │   ├── images/
    │   └── labels/
    └── level_060/          # 60% occlusion (60 cells per bbox)
        ├── images/
        └── labels/

Note:
    Labels are COPIED unchanged because:
    - The object is still there (just partially hidden)
    - We want to measure if the model can still detect it
    - Ground truth doesn't change, only image visibility
"""

import json
import argparse
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic occlusion test sets"
    )
    parser.add_argument(
        "--test_index",
        type=str,
        default="data/processed/evaluation/test_index.json",
        help="Path to test_index.json"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/raw/test/images",
        help="Path to test images directory"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="data/raw/test/labels",
        help="Path to test labels directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic_occlusion",
        help="Output directory for synthetic test sets"
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="0.2,0.4,0.6",
        help="Occlusion levels as comma-separated values (e.g., 0.2,0.4,0.6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_test_index(index_path: Path) -> Dict:
    """Load test index JSON."""
    with open(index_path, 'r') as f:
        return json.load(f)


def generate_grid_occlusion_cells(
    bbox_xyxy: List[int],
    occlusion_level: float,
    grid_size: int,
    seed: int
) -> List[Tuple[int, int, int, int]]:
    """
    Generate multiple small rectangles using a grid-based approach.
    
    Divides the bounding box into a grid (e.g., 10x10 = 100 cells).
    Randomly selects cells to fill black until target coverage is reached.
    
    Benefits:
    - Exact coverage (20% = exactly 20 cells out of 100)
    - No overlap between occlusion rectangles
    - More realistic scattered occlusion pattern
    
    Args:
        bbox_xyxy: [x1, y1, x2, y2] bounding box coordinates
        occlusion_level: Fraction of bbox to cover (0.0 to 1.0)
        grid_size: Number of cells per row/column (default 10 = 100 cells)
        seed: Random seed for reproducibility
    
    Returns:
        List of (x1, y1, x2, y2) rectangles to fill black
    """
    random.seed(seed)
    
    x1, y1, x2, y2 = bbox_xyxy
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width <= 0 or bbox_height <= 0:
        return []
    
    # Calculate cell dimensions
    cell_width = bbox_width / grid_size
    cell_height = bbox_height / grid_size
    
    # Total cells and how many to fill
    total_cells = grid_size * grid_size
    cells_to_fill = int(occlusion_level * total_cells)
    
    # Create list of all cell coordinates (row, col)
    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    
    # Randomly select cells to occlude
    random.shuffle(all_cells)
    selected_cells = all_cells[:cells_to_fill]
    
    # Convert cell coordinates to pixel rectangles
    rectangles = []
    for (row, col) in selected_cells:
        rect_x1 = int(x1 + col * cell_width)
        rect_y1 = int(y1 + row * cell_height)
        rect_x2 = int(x1 + (col + 1) * cell_width)
        rect_y2 = int(y1 + (row + 1) * cell_height)
        rectangles.append((rect_x1, rect_y1, rect_x2, rect_y2))
    
    return rectangles


def apply_occlusions(
    image: Image.Image,
    boxes: List[Dict],
    occlusion_level: float,
    base_seed: int,
    grid_size: int = 10
) -> Image.Image:
    """
    Apply synthetic grid-based occlusions to all bounding boxes in an image.
    
    Each bounding box is divided into a grid (default 10x10 = 100 cells).
    Random cells are filled with black until target coverage is reached.
    
    Example: 20% occlusion with 10x10 grid = 20 random cells per bbox filled black.
    
    Args:
        image: PIL Image
        boxes: List of ground truth boxes with 'bbox_xyxy'
        occlusion_level: Fraction of each bbox to cover (e.g., 0.2 = 20%)
        base_seed: Base seed (combined with box index for reproducibility)
        grid_size: Grid divisions per side (10 = 100 cells, each cell = 1%)
    
    Returns:
        Modified image with grid-based occlusions
    """
    # Make a copy to avoid modifying original
    occluded_img = image.copy()
    draw = ImageDraw.Draw(occluded_img)
    
    for i, box in enumerate(boxes):
        bbox_xyxy = box.get('bbox_xyxy')
        if not bbox_xyxy:
            continue
        
        # Generate grid-based occlusion rectangles for this box
        rectangles = generate_grid_occlusion_cells(
            bbox_xyxy,
            occlusion_level,
            grid_size=grid_size,
            seed=base_seed + i
        )
        
        # Draw all black rectangles for this bbox
        for rect in rectangles:
            draw.rectangle(rect, fill='black')
    
    return occluded_img


def process_test_set(
    test_index: Dict,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    occlusion_level: float,
    seed: int
) -> Dict:
    """
    Generate synthetic occlusion test set for a specific level.
    
    Args:
        test_index: Loaded test_index.json
        images_dir: Path to original test images
        labels_dir: Path to original test labels
        output_dir: Output directory for this level
        occlusion_level: Fraction to occlude (e.g., 0.4)
        seed: Random seed
    
    Returns:
        Statistics dict
    """
    # Create output directories
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'total_boxes_occluded': 0,
        'occlusion_level': occlusion_level
    }
    
    for img_data in test_index['images']:
        filename = img_data['image_filename']
        image_id = img_data['image_id']
        boxes = img_data['ground_truth']
        
        # Load original image
        img_path = images_dir / filename
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image = Image.open(img_path).convert('RGB')
        
        # Apply occlusions
        occluded_image = apply_occlusions(
            image,
            boxes,
            occlusion_level,
            base_seed=seed + hash(image_id) % 10000
        )
        
        # Save occluded image
        occluded_image.save(out_images / filename)
        
        # Copy label file unchanged (ground truth stays the same)
        label_filename = image_id + ".txt"
        src_label = labels_dir / label_filename
        dst_label = out_labels / label_filename
        
        if src_label.exists():
            shutil.copy(src_label, dst_label)
        
        stats['total_images'] += 1
        stats['total_boxes_occluded'] += len(boxes)
    
    return stats


def create_data_yaml(output_dir: Path, original_yaml_path: Path, level_name: str):
    """
    Create a data.yaml for this synthetic test set.
    
    Args:
        output_dir: Directory for this occlusion level
        original_yaml_path: Path to original data.yaml
        level_name: Name of this level (e.g., "level_040")
    """
    import yaml
    
    # Load original to get class names
    with open(original_yaml_path, 'r') as f:
        original = yaml.safe_load(f)
    
    # Create config for this synthetic set
    config = {
        'path': str(output_dir.absolute()),
        'test': 'images',
        'names': original['names'],
        'nc': len(original['names']) if isinstance(original['names'], list) else original.get('nc', 0),
        '_info': f'Synthetic occlusion test set: {level_name}'
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    args = parse_args()
    
    # Parse occlusion levels
    levels = [float(x.strip()) for x in args.levels.split(',')]
    
    print("=" * 60)
    print("GENERATE SYNTHETIC OCCLUSIONS")
    print("=" * 60)
    print(f"  Test index:  {args.test_index}")
    print(f"  Images dir:  {args.images_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Levels:      {levels}")
    print(f"  Seed:        {args.seed}")
    print("=" * 60)
    
    # Load test index
    test_index_path = Path(args.test_index)
    if not test_index_path.exists():
        raise FileNotFoundError(
            f"Test index not found: {test_index_path}\n"
            "Run build_evaluation_index.py first."
        )
    
    test_index = load_test_index(test_index_path)
    print(f"\nLoaded {len(test_index['images'])} test images from index")
    
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    
    # Check for original data.yaml
    original_yaml = images_dir.parent.parent / "data.yaml"
    if not original_yaml.exists():
        original_yaml = images_dir.parent / "data.yaml"
    
    # Process each occlusion level
    all_stats = {}
    
    for level in levels:
        level_name = f"level_{int(level * 100):03d}"
        level_output = output_dir / level_name
        
        print(f"\nProcessing {level_name} ({int(level * 100)}% occlusion)...")
        
        stats = process_test_set(
            test_index=test_index,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=level_output,
            occlusion_level=level,
            seed=args.seed
        )
        
        # Create data.yaml for this level
        if original_yaml.exists():
            create_data_yaml(level_output, original_yaml, level_name)
        
        all_stats[level_name] = stats
        print(f"  ✓ Created {stats['total_images']} images, {stats['total_boxes_occluded']} boxes occluded")
    
    # Save manifest
    manifest = {
        'seed': args.seed,
        'source_test_index': str(test_index_path),
        'occlusion_levels': {
            f"level_{int(lvl * 100):03d}": f"{int(lvl * 100)}% of each bbox covered"
            for lvl in levels
        },
        'statistics': all_stats
    }
    
    manifest_path = output_dir / "occlusion_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SYNTHETIC OCCLUSION TEST SETS COMPLETE")
    print("=" * 60)
    print(f"\nCreated test sets:")
    for level in levels:
        level_name = f"level_{int(level * 100):03d}"
        print(f"  ✓ {output_dir / level_name}")
    print(f"\nManifest: {manifest_path}")
    
    print("\n" + "=" * 60)
    print("NEXT: Run evaluation on each test set")
    print("=" * 60)
    print("""
Example usage in evaluation:

    # For each occlusion level:
    model = YOLO('models/yolov8n_baseline.pt')
    results = model.val(data='data/synthetic_occlusion/level_020/data.yaml')
    
    # Compare mAP across levels to see degradation curve
""")


if __name__ == "__main__":
    main()
