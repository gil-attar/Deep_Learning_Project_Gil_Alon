"""
Quick Test Script for Evaluation System
========================================
Tests the evaluation module without training a model.
Uses existing test_index.json to create dummy predictions.

This is MUCH faster than training even 5 epochs.
Just tests that all functions work without errors.

Usage:
    python scripts/test_evaluation_quick.py
"""

import sys
from pathlib import Path
import json
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.io import load_ground_truth, load_class_names, save_metrics
from evaluation.metrics import (
    eval_detection_prf_at_iou,
    eval_per_class_metrics_and_confusions,
    eval_counting_quality
)
from evaluation.plots import plot_all_metrics


def create_dummy_predictions(ground_truths, noise_level=0.3):
    """
    Create dummy predictions from ground truth (with noise).

    Args:
        ground_truths: List of GT images
        noise_level: Fraction of boxes to randomly modify (0-1)

    Returns:
        Predictions in the correct format
    """
    predictions = []

    for gt_img in ground_truths[:50]:  # Use only first 50 for quick test
        image_id = gt_img['image_id']
        gt_boxes = gt_img['ground_truth']

        detections = []

        # Copy GT boxes with some noise
        for gt_box in gt_boxes:
            # Randomly decide: keep, drop, or add noise
            rand = random.random()

            if rand < noise_level:
                # Drop this detection (creates FN)
                continue
            elif rand < noise_level * 2:
                # Add noise to bbox (might create FP or lower IoU)
                bbox = gt_box['bbox_xyxy']
                noisy_bbox = [
                    bbox[0] + random.randint(-20, 20),
                    bbox[1] + random.randint(-20, 20),
                    bbox[2] + random.randint(-20, 20),
                    bbox[3] + random.randint(-20, 20)
                ]

                detections.append({
                    'class_id': gt_box['class_id'],
                    'class_name': gt_box['class_name'],
                    'confidence': random.uniform(0.5, 0.99),
                    'bbox': noisy_bbox,
                    'bbox_format': 'xyxy'
                })
            else:
                # Perfect detection
                detections.append({
                    'class_id': gt_box['class_id'],
                    'class_name': gt_box['class_name'],
                    'confidence': random.uniform(0.7, 0.99),
                    'bbox': gt_box['bbox_xyxy'],
                    'bbox_format': 'xyxy'
                })

        # Add some random false positives
        if random.random() < 0.2:
            detections.append({
                'class_id': random.randint(0, 25),
                'class_name': f"class_{random.randint(0, 25)}",
                'confidence': random.uniform(0.3, 0.7),
                'bbox': [100, 100, 200, 200],
                'bbox_format': 'xyxy'
            })

        predictions.append({
            'image_id': image_id,
            'detections': detections
        })

    return predictions


def main():
    print("=" * 70)
    print("QUICK TEST - EVALUATION SYSTEM")
    print("=" * 70)
    print("\nThis test creates dummy predictions from ground truth.")
    print("No model training required - just tests that functions work!\n")

    # Check if test_index exists
    test_index_path = "data/processed/evaluation/test_index.json"
    if not Path(test_index_path).exists():
        print(f"❌ Test index not found: {test_index_path}")
        print("\nRun this first:")
        print("  python scripts/build_evaluation_indices.py")
        return 1

    # Load ground truth
    print("Loading ground truth...")
    ground_truths = load_ground_truth(test_index_path)
    class_names = load_class_names(test_index_path)
    print(f"✓ Loaded {len(ground_truths)} GT images")
    print(f"✓ Loaded {len(class_names)} classes")

    # Create dummy predictions
    print("\nCreating dummy predictions (with noise)...")
    random.seed(42)
    predictions = create_dummy_predictions(ground_truths, noise_level=0.2)

    # Filter GTs to match predictions
    pred_image_ids = {p['image_id'] for p in predictions}
    ground_truths = [g for g in ground_truths if g['image_id'] in pred_image_ids]

    print(f"✓ Created {len(predictions)} dummy predictions")

    # Test metrics
    print("\n" + "=" * 70)
    print("TESTING METRICS")
    print("=" * 70)

    print("\n1. Testing P/R/F1 threshold sweep...")
    try:
        threshold_sweep = eval_detection_prf_at_iou(
            predictions, ground_truths,
            iou_threshold=0.5,
            conf_thresholds=[0.2, 0.4, 0.6, 0.8]
        )
        print("✓ P/R/F1 evaluation works!")
        print(f"   Results at conf=0.5: F1={threshold_sweep.get('0.5', {}).get('f1', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    print("\n2. Testing per-class metrics...")
    try:
        per_class_results = eval_per_class_metrics_and_confusions(
            predictions, ground_truths,
            iou_threshold=0.5,
            conf_threshold=0.5,
            class_names=class_names
        )
        print("✓ Per-class evaluation works!")
        print(f"   Evaluated {len(per_class_results['per_class'])} classes")
        print(f"   Found {len(per_class_results['top_confusions'])} confusions")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    print("\n3. Testing counting quality...")
    try:
        counting_results = eval_counting_quality(
            predictions, ground_truths,
            iou_threshold=0.5,
            conf_threshold=0.5,
            class_names=class_names
        )
        print("✓ Counting evaluation works!")
        print(f"   Matched-only MAE: {counting_results['matched_only']['global_mae']:.4f}")
        print(f"   All-predictions MAE: {counting_results['all_predictions']['global_mae']:.4f}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    # Test plots
    print("\n" + "=" * 70)
    print("TESTING PLOTS")
    print("=" * 70)

    output_dir = "evaluation/results/quick_test/"

    try:
        plot_all_metrics(
            threshold_sweep=threshold_sweep,
            per_class_results=per_class_results['per_class'],
            confusion_data=per_class_results,
            counting_results=counting_results,
            output_dir=output_dir,
            run_name="Quick Test (Dummy Predictions)"
        )
        print(f"\n✓ All plots generated successfully!")
        print(f"   Saved to: {output_dir}")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        return 1

    # Save results
    print("\n" + "=" * 70)
    print("TESTING FILE I/O")
    print("=" * 70)

    try:
        results = {
            'threshold_sweep': threshold_sweep,
            'per_class': per_class_results['per_class'],
            'counting': counting_results
        }

        save_metrics(results, f"{output_dir}/test_metrics.json")
        print("✓ Metrics saved successfully!")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
        return 1

    # Final summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe evaluation system is working correctly!")
    print("\nNext steps:")
    print("  1. Train a real model (or use existing weights)")
    print("  2. Generate predictions on train/val/test")
    print("  3. Run evaluation: python scripts/evaluate_run.py")
    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
