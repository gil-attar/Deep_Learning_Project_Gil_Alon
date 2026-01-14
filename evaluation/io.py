"""
I/O utilities for loading predictions and ground truth.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_predictions(predictions_path: str, split: Optional[str] = None) -> List[Dict]:
    """
    Load model predictions from JSON file.

    Args:
        predictions_path: Path to predictions JSON file
        split: Expected split name for validation (optional, e.g., "train", "val", "test")

    Returns:
        List of per-image predictions in format:
        [
            {
                "image_id": str,
                "detections": [
                    {
                        "class_id": int,
                        "class_name": str,
                        "confidence": float,
                        "bbox": [x1, y1, x2, y2],  # xyxy format
                    },
                    ...
                ]
            },
            ...
        ]

    Raises:
        FileNotFoundError: If predictions file doesn't exist
        ValueError: If split doesn't match (when split is specified)
    """
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(pred_path, 'r') as f:
        data = json.load(f)

    # Validate split if specified
    if split and 'split' in data:
        if data['split'] != split:
            raise ValueError(
                f"Expected split '{split}', but predictions file has split '{data['split']}'"
            )

    # Extract predictions array
    predictions = data.get('predictions', [])

    if not predictions:
        raise ValueError(f"No predictions found in {predictions_path}")

    return predictions


def load_ground_truth(
    index_path: str,
    split: Optional[str] = None,
    include_metadata: bool = False
) -> List[Dict]:
    """
    Load ground truth annotations from index JSON file.

    Args:
        index_path: Path to *_index.json (e.g., test_index.json, val_index.json)
        split: Expected split name for validation (optional)
        include_metadata: If True, return (images, metadata) tuple

    Returns:
        List of per-image ground truth in format:
        [
            {
                "image_id": str,
                "image_filename": str,
                "ground_truth": [
                    {
                        "class_id": int,
                        "class_name": str,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_yolo": [x_center, y_center, w, h]  # optional
                    },
                    ...
                ]
            },
            ...
        ]

    Raises:
        FileNotFoundError: If index file doesn't exist
        ValueError: If split doesn't match (when split is specified)
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Ground truth index not found: {index_path}")

    with open(index_path, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})

    # Validate split if specified
    if split and 'split' in metadata:
        if metadata['split'] != split:
            raise ValueError(
                f"Expected split '{split}', but index file has split '{metadata['split']}'"
            )

    images = data.get('images', [])

    if not images:
        raise ValueError(f"No images found in {index_path}")

    if include_metadata:
        return images, metadata
    return images


def load_class_names(index_path: str) -> Dict[int, str]:
    """
    Load class names mapping from index file.

    Args:
        index_path: Path to *_index.json

    Returns:
        Dict mapping class_id (int) to class_name (str)
    """
    index_path = Path(index_path)
    with open(index_path, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    class_names = metadata.get('class_names', {})

    # Convert string keys to int
    return {int(k): v for k, v in class_names.items()}


def save_metrics(results: Dict, output_path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: Dictionary containing evaluation metrics
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved metrics to: {output_path}")


def save_summary_csv(results: Dict, output_path: str):
    """
    Save a summary CSV for easy copy-paste into reports.

    Args:
        results: Dictionary containing threshold_sweep results
        output_path: Path to save CSV file
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract threshold sweep data
    threshold_sweep = results.get('threshold_sweep', {})

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['conf_threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'])

        for threshold in sorted(threshold_sweep.keys(), key=float):
            metrics = threshold_sweep[threshold]
            writer.writerow([
                threshold,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                metrics['tp'],
                metrics['fp'],
                metrics['fn']
            ])

    print(f"✓ Saved summary CSV to: {output_path}")
