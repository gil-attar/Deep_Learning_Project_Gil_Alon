"""
Dataset Download Script
=======================
Downloads the Food Ingredients dataset from Roboflow to data/raw/.
The raw data is treated as IMMUTABLE - never modify files in data/raw/.

Usage:
    python scripts/download_dataset.py --output_dir data/raw

Arguments:
    --output_dir : Directory to save dataset (default: data/raw)
    --version    : Roboflow dataset version (default: 1)

Requires:
    Environment variable ROBOFLOW_API_KEY must be set.

Note:
    This script is part of Step 2 (Data Pipeline & Evaluation Foundations).
    The downloaded data should never be modified after initial download.
"""

import os
import argparse
import shutil
from pathlib import Path

# Roboflow settings
WORKSPACE = "gaworkspace-utcbg"
PROJECT = "food-ingredients-dataset-2-rewtd"
DEFAULT_VERSION = 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Food Ingredients dataset from Roboflow"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save dataset (default: data/raw)"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        help=f"Roboflow dataset version (default: {DEFAULT_VERSION})"
    )
    return parser.parse_args()


def download_dataset(output_dir: str, version: int):
    """
    Download the Food Ingredients dataset from Roboflow.
    
    Args:
        output_dir: Directory to save the dataset
        version: Roboflow dataset version to download
    """
    from roboflow import Roboflow
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY environment variable not set!")
    
    print("=" * 60)
    print("DOWNLOADING FOOD INGREDIENTS DATASET")
    print("=" * 60)
    print(f"  Workspace: {WORKSPACE}")
    print(f"  Project:   {PROJECT}")
    print(f"  Version:   {version}")
    print(f"  Output:    {output_dir}")
    print("=" * 60)
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    
    print(f"\nDownloading version {version}...")
    dataset = project.version(version).download("yolov8")
    
    print(f"Downloaded to: {dataset.location}")
    
    # Move to output directory
    src = Path(dataset.location)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    # Clean destination if exists
    for split in ["train", "valid", "test"]:
        split_dest = dest / split
        if split_dest.exists():
            shutil.rmtree(split_dest)
    
    # Move train, valid, test folders
    for split in ["train", "valid", "test"]:
        if (src / split).exists():
            shutil.move(str(src / split), str(dest / split))
            print(f"  Moved {split}/")
    
    # Copy data.yaml
    if (src / "data.yaml").exists():
        shutil.copy(str(src / "data.yaml"), str(dest / "data.yaml"))
        print("  Copied data.yaml")
    
    # Clean up source folder
    shutil.rmtree(src, ignore_errors=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    
    for split in ["train", "valid", "test"]:
        split_dir = dest / split / "images"
        if split_dir.exists():
            count = len(list(split_dir.glob("*")))
            print(f"  {split}: {count} images")
    
    print(f"\n✓ Dataset saved to {dest}")
    print("\nIMPORTANT: data/raw/ is READ-ONLY. Never modify these files.")
    print("\nNext step: Run build_evaluation_index.py to create evaluation artifacts.")


if __name__ == "__main__":
    args = parse_args()
    download_dataset(args.output_dir, args.version)
