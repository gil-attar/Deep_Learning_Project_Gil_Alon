"""
Dataset Download Script
=======================
Downloads the Food Ingredients dataset from Roboflow.

Usage:
    python scripts/download_dataset.py

Requires:
    Environment variable ROBOFLOW_API_KEY must be set.
"""

import os
import shutil
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Roboflow settings
WORKSPACE = "gaworkspace-utcbg"
PROJECT = "food-ingredients-dataset-2-rewtd"
VERSION = 1


def download_dataset():
    """Download the Food Ingredients dataset from Roboflow."""
    from roboflow import Roboflow
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY environment variable not set!")
    
    print("=" * 50)
    print("Downloading Food Ingredients Dataset")
    print("=" * 50)
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    
    print(f"Downloading version {VERSION}...")
    dataset = project.version(VERSION).download("yolov8")
    
    print(f"Downloaded to: {dataset.location}")
    
    # Move to data/processed/
    src = Path(dataset.location)
    dest = DATA_PROCESSED_DIR
    dest.mkdir(parents=True, exist_ok=True)
    
    # Clean destination
    for split in ["train", "valid", "test"]:
        split_dest = dest / split
        if split_dest.exists():
            shutil.rmtree(split_dest)
    
    # Move train, valid, test folders
    for split in ["train", "valid", "test"]:
        if (src / split).exists():
            shutil.move(str(src / split), str(dest / split))
            print(f"Moved {split}/")
    
    # Copy data.yaml
    if (src / "data.yaml").exists():
        shutil.copy(str(src / "data.yaml"), str(dest / "data.yaml"))
        print("Copied data.yaml")
    
    # Clean up source folder
    shutil.rmtree(src, ignore_errors=True)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Download Complete!")
    print("=" * 50)
    
    for split in ["train", "valid", "test"]:
        split_dir = dest / split / "images"
        if split_dir.exists():
            count = len(list(split_dir.glob("*")))
            print(f"  {split}: {count} images")
    
    print(f"\n✓ Dataset ready in {dest}")


if __name__ == "__main__":
    download_dataset()
