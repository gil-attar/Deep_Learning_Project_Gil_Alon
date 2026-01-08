"""
Dataset Download Script
=======================
Downloads the Food Ingredients dataset from Roboflow.
The dataset will be saved in YOLO format for compatibility with both YOLOv8 and RT-DETR.

Usage:
    python scripts/download_dataset.py

Requires:
    Environment variable ROBOFLOW_API_KEY must be set.
    Get your API key at: https://app.roboflow.com/settings/api
"""

import os
from pathlib import Path

# Get API key from environment variable
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    """Download the Food Ingredients dataset from Roboflow."""
    
    from roboflow import Roboflow
    
    print("=" * 50)
    print("Downloading Food Ingredients Dataset")
    print("=" * 50)
    
    # Initialize Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    # Access the dataset
    # Dataset: https://universe.roboflow.com/samuels/food-ingredients-dataset-2
    project = rf.workspace("samuels").project("food-ingredients-dataset-2")
    
    # Download version 4 (latest) in YOLO format
    dataset = project.version(4).download(
        model_format="yolov8",
        location=str(DATA_RAW_DIR)
    )
    
    print("\n" + "=" * 50)
    print(f"Dataset downloaded to: {DATA_RAW_DIR}")
    print("=" * 50)
    
    # Print dataset structure
    print("\nDataset structure:")
    for split in ["train", "valid", "test"]:
        split_dir = DATA_RAW_DIR / split
        if split_dir.exists():
            images = list((split_dir / "images").glob("*")) if (split_dir / "images").exists() else []
            labels = list((split_dir / "labels").glob("*")) if (split_dir / "labels").exists() else []
            print(f"  {split}: {len(images)} images, {len(labels)} labels")
    
    return dataset


def print_examples(num_examples: int = 3):
    """
    Print example images and their label contents.
    
    Args:
        num_examples: Number of examples to show
    """
    print("\n" + "=" * 50)
    print("EXAMPLE DATA")
    print("=" * 50)
    
    # Find the dataset directory
    dataset_dir = None
    for subdir in DATA_RAW_DIR.iterdir():
        if subdir.is_dir() and (subdir / "train" / "images").exists():
            dataset_dir = subdir
            break
    
    if dataset_dir is None:
        if (DATA_RAW_DIR / "train" / "images").exists():
            dataset_dir = DATA_RAW_DIR
        else:
            print("Dataset not found!")
            return
    
    # Load class names from data.yaml
    import yaml
    yaml_path = dataset_dir / "data.yaml"
    class_names = {}
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            names = config.get('names', {})
            if isinstance(names, list):
                class_names = {i: name for i, name in enumerate(names)}
            else:
                class_names = {int(k): v for k, v in names.items()}
        print(f"\nClasses ({len(class_names)}): {list(class_names.values())[:10]}...")
    
    # Show examples from train split
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    print(f"\n--- Showing {min(num_examples, len(image_files))} examples from train split ---\n")
    
    for i, img_path in enumerate(image_files[:num_examples]):
        print(f"Example {i+1}:")
        print(f"  Image: {img_path.name}")
        
        # Read corresponding label
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            print(f"  Labels ({len(lines)} objects):")
            for line in lines[:5]:  # Show max 5 objects per image
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    x, y, w, h = [float(p) for p in parts[1:5]]
                    print(f"    - {class_name}: center=({x:.2f}, {y:.2f}), size=({w:.2f}, {h:.2f})")
            
            if len(lines) > 5:
                print(f"    ... and {len(lines) - 5} more objects")
        else:
            print(f"  Labels: (no label file)")
        
        print()


if __name__ == "__main__":
    # Ensure data directory exists
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ROBOFLOW_API_KEY:
        print("ERROR: ROBOFLOW_API_KEY environment variable not set!")
        print("\nTo fix this, run:")
        print("  export ROBOFLOW_API_KEY='your_private_api_key'")
        print("\nGet your API key at: https://app.roboflow.com/settings/api")
    else:
        download_dataset()
        print_examples(num_examples=3)
