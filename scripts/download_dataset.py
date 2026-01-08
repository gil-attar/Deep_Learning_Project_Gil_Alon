"""
Dataset Download Script
=======================
Downloads the Food Ingredients dataset from Roboflow directly to data/processed/.
Uses Roboflow's pre-configured train/valid/test splits.

Usage:
    python scripts/download_dataset.py

Requires:
    Environment variable ROBOFLOW_API_KEY must be set.
"""

import os
import shutil
from pathlib import Path

# Get API key from environment variable
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_dataset():
    """Download the Food Ingredients dataset from Roboflow."""
    
    from roboflow import Roboflow
    
    print("=" * 50)
    print("Downloading Food Ingredients Dataset")
    print("=" * 50)
    
    # Initialize Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    # Access the dataset
    project = rf.workspace("gaworkspace-utcbg").project("food-ingredients-dataset-2-rewtd")
    
    # Try versions 1, 2, 3
    dataset = None
    version_num = None
    for v in [1, 2, 3]:
        try:
            print(f"Trying version {v}...")
            version = project.version(v)
            
            # Debug: print the dataset location before download
            print(f"[DEBUG] Download location set to: {DATA_PROCESSED_DIR}")
            print(f"[DEBUG] Current working directory: {os.getcwd()}")
            
            dataset = version.download(
                model_format="yolov8",
                location=str(DATA_PROCESSED_DIR)
            )
            
            # Check where files actually went
            print(f"[DEBUG] Dataset location after download: {dataset.location}")
            
            version_num = v
            print(f"✓ Downloaded version {v}")
            break
        except RuntimeError:
            print(f"  Version {v} not found")
            continue
    
    if dataset is None:
        raise RuntimeError("Could not find any valid version")
    
    # Get the actual location where Roboflow downloaded
    actual_location = Path(dataset.location)
    print(f"\n[DEBUG] Actual dataset location: {actual_location}")
    
    # If files are not in DATA_PROCESSED_DIR, move them there
    if actual_location != DATA_PROCESSED_DIR and actual_location.exists():
        print(f"[DEBUG] Moving from {actual_location} to {DATA_PROCESSED_DIR}")
        for item in actual_location.iterdir():
            dest = DATA_PROCESSED_DIR / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        # Try to remove the empty source directory
        try:
            actual_location.rmdir()
        except:
            pass
    
    # Debug: Show what's in the processed directory
    print(f"\n[DEBUG] Contents of {DATA_PROCESSED_DIR}:")
    for item in DATA_PROCESSED_DIR.iterdir():
        print(f"  {item.name}/ " if item.is_dir() else f"  {item.name}")
        if item.is_dir():
            for subitem in list(item.iterdir())[:5]:
                print(f"    {subitem.name}")
    
    # Roboflow creates a subfolder - move contents up
    # e.g., data/processed/food-ingredients-dataset-2-rewtd-1/ → data/processed/
    moved = False
    for subdir in DATA_PROCESSED_DIR.iterdir():
        if subdir.is_dir() and "food-ingredients" in subdir.name.lower():
            print(f"\nMoving files from {subdir.name}/ to processed/...")
            for item in subdir.iterdir():
                dest = DATA_PROCESSED_DIR / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            subdir.rmdir()
            moved = True
            break
    
    if not moved:
        print("\n[DEBUG] No subfolder found to move. Checking direct structure...")
    
    print("\n" + "=" * 50)
    print("Download Complete!")
    print("=" * 50)
    
    # Print structure
    print("\nDataset structure:")
    for split in ["train", "valid", "test"]:
        split_dir = DATA_PROCESSED_DIR / split
        if split_dir.exists():
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            n_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            n_labels = len(list(labels_dir.glob("*"))) if labels_dir.exists() else 0
            print(f"  {split}: {n_images} images, {n_labels} labels")
        else:
            print(f"  {split}: (not found)")
    
    return dataset


if __name__ == "__main__":
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ROBOFLOW_API_KEY:
        print("ERROR: ROBOFLOW_API_KEY not set!")
        print("Run: export ROBOFLOW_API_KEY='your_key'")
    else:
        download_dataset()
