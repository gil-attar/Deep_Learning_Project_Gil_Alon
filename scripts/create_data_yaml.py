"""
Create data.yaml for Ultralytics training/validation.

This script creates a canonical data.yaml in data/processed/ that points
to the dataset with absolute or relative paths.
"""

import yaml
from pathlib import Path
import argparse


def create_data_yaml(dataset_root, output_path, use_absolute_paths=False):
    """
    Create data.yaml for Ultralytics.

    Args:
        dataset_root: Root directory containing train/valid/test folders
        output_path: Where to save data.yaml
        use_absolute_paths: Whether to use absolute paths (for Colab)
    """
    dataset_root = Path(dataset_root).resolve()

    # Read original data.yaml to get class names
    original_yaml = dataset_root / "data.yaml"
    if not original_yaml.exists():
        raise FileNotFoundError(f"Original data.yaml not found: {original_yaml}")

    with open(original_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # Create new config
    if use_absolute_paths:
        # For Colab - use absolute paths
        data_config = {
            'path': str(dataset_root),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': config['names'],
            'nc': len(config['names'])
        }
    else:
        # For local - use relative paths from project root
        # Assumes data.yaml is in data/processed/ and dataset is in data/raw/
        data_config = {
            'path': '../raw',  # Relative to data/processed/
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': config['names'],
            'nc': len(config['names'])
        }

    # Save new data.yaml
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created data.yaml: {output_path}")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Path: {data_config['path']}")
    print(f"  Train: {data_config['train']}")
    print(f"  Val: {data_config['val']}")
    print(f"  Test: {data_config['test']}")

    return data_config


def main():
    parser = argparse.ArgumentParser(description="Create data.yaml for training/evaluation")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/raw",
        help="Root directory of dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/data.yaml",
        help="Output path for data.yaml"
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute paths (for Colab)"
    )

    args = parser.parse_args()

    create_data_yaml(
        dataset_root=args.dataset_root,
        output_path=args.output,
        use_absolute_paths=args.absolute
    )


if __name__ == "__main__":
    main()
