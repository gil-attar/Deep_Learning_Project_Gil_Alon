"""
Evaluation Module for Detection Metrics

This module provides comprehensive evaluation metrics for object detection models.
Supports YOLOv8 and RT-DETR predictions with standardized JSON formats.

Main Components:
- io: Load predictions and ground truth from JSON files
- matching: IoU computation and greedy box matching
- metrics: Core evaluation functions (P/R/F1, per-class, counting)
- plots: Visualization functions

Usage:
    from evaluation.metrics import (
        eval_detection_prf_at_iou,
        eval_per_class_metrics_and_confusions,
        eval_counting_quality
    )
    from evaluation.io import load_predictions, load_ground_truth

    preds = load_predictions("path/to/predictions.json")
    gts = load_ground_truth("path/to/test_index.json")

    results = eval_detection_prf_at_iou(preds, gts)
"""

__version__ = "1.0.0"
