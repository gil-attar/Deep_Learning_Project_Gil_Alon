"""
Evaluate Run - Standalone Evaluation from Prediction JSONs
===========================================================
Run comprehensive evaluation on saved predictions without re-running inference.

This script:
1. Loads predictions and ground truth from JSON files
2. Runs all 3 core metrics (P/R/F1, per-class, counting)
3. Generates plots and summary files

Usage:
    # Evaluate a single run
    python scripts/evaluate_run.py \\
        --predictions evaluation/metrics/baseline_yolo_test_predictions.json \\
        --ground_truth data/processed/evaluation/test_index.json \\
        --output_dir evaluation/results/baseline_yolo/test/ \\
        --run_name "Baseline YOLO"

    # With custom thresholds
    python scripts/evaluate_run.py \\
        --predictions path/to/predictions.json \\
        --ground_truth path/to/index.json \\
        --output_dir results/my_run/ \\
        --conf_thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \\
        --iou_threshold 0.5
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path to import evaluation module
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.io import load_predictions, load_ground_truth, load_class_names, save_metrics, save_summary_csv
from evaluation.metrics import (
    eval_detection_prf_at_iou,
    eval_per_class_metrics_and_confusions,
    eval_counting_quality
)
from evaluation.plots import plot_all_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate detection model from saved predictions"
    )

    # Required arguments
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth index JSON file (e.g., test_index.json)"
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/results/run/",
        help="Output directory for results (default: evaluation/results/run/)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this evaluation run (for plot titles)"
    )
    parser.add_argument(
        "--conf_thresholds",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated confidence thresholds (default: 0.0 to 0.9)"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Expected split name (optional validation)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("EVALUATE RUN")
    print("=" * 70)
    print(f"Predictions:   {args.predictions}")
    print(f"Ground Truth:  {args.ground_truth}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print("=" * 70)

    # Parse confidence thresholds
    conf_thresholds = [float(x.strip()) for x in args.conf_thresholds.split(',')]
    print(f"\nEvaluating at {len(conf_thresholds)} confidence thresholds: {conf_thresholds}")

    # Load predictions
    print(f"\nLoading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions, split=args.split)
    print(f"✓ Loaded {len(predictions)} images")

    # Load ground truth
    print(f"\nLoading ground truth from {args.ground_truth}...")
    ground_truths = load_ground_truth(args.ground_truth, split=args.split)
    print(f"✓ Loaded {len(ground_truths)} images")

    # Load class names
    class_names = load_class_names(args.ground_truth)
    print(f"✓ Loaded {len(class_names)} classes")

    # Validate matching image counts
    if len(predictions) != len(ground_truths):
        print(f"\n⚠ WARNING: Prediction count ({len(predictions)}) != GT count ({len(ground_truths)})")

    # Run evaluations
    print("\n" + "=" * 70)
    print("RUNNING EVALUATIONS")
    print("=" * 70)

    # 1. Detection P/R/F1 at multiple thresholds
    print("\n1. Detection P/R/F1 at IoU threshold...")
    threshold_sweep = eval_detection_prf_at_iou(
        predictions, ground_truths,
        iou_threshold=args.iou_threshold,
        conf_thresholds=conf_thresholds
    )

    # Find best threshold by F1
    best_conf_thr = max(threshold_sweep.keys(), key=lambda k: threshold_sweep[k]['f1'])
    best_f1 = threshold_sweep[best_conf_thr]['f1']
    print(f"✓ Best conf_threshold: {best_conf_thr} (F1={best_f1:.4f})")

    # 2. Per-class metrics at best threshold
    print(f"\n2. Per-class metrics at conf={best_conf_thr}...")
    per_class_results = eval_per_class_metrics_and_confusions(
        predictions, ground_truths,
        iou_threshold=args.iou_threshold,
        conf_threshold=float(best_conf_thr),
        class_names=class_names
    )
    num_confusions = len([c for c in per_class_results['top_confusions'] if c['count'] > 0])
    print(f"✓ Computed metrics for {len(per_class_results['per_class'])} classes")
    print(f"✓ Found {num_confusions} class confusions")

    # 3. Counting quality at best threshold
    print(f"\n3. Counting quality at conf={best_conf_thr}...")
    counting_results = eval_counting_quality(
        predictions, ground_truths,
        iou_threshold=args.iou_threshold,
        conf_threshold=float(best_conf_thr),
        class_names=class_names
    )
    print(f"✓ Matched-only MAE: {counting_results['matched_only']['global_mae']:.4f}")
    print(f"✓ All-predictions MAE: {counting_results['all_predictions']['global_mae']:.4f}")

    # Compile results
    results = {
        'run_name': args.run_name or Path(args.predictions).stem,
        'predictions_path': str(args.predictions),
        'ground_truth_path': str(args.ground_truth),
        'evaluation_settings': {
            'iou_threshold': args.iou_threshold,
            'conf_thresholds_evaluated': conf_thresholds
        },
        'threshold_sweep': threshold_sweep,
        'best_threshold': {
            'value': float(best_conf_thr),
            'selected_by': 'max_f1',
            'f1': best_f1
        },
        'per_class_metrics': {
            'conf_threshold': float(best_conf_thr),
            'classes': per_class_results['per_class']
        },
        'confusion_matrix': {
            'conf_threshold': float(best_conf_thr),
            'matrix': per_class_results['confusion_matrix'],
            'class_order': per_class_results['class_order'],
            'top_confusions': per_class_results['top_confusions']
        },
        'counting_quality': counting_results
    }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    save_metrics(results, metrics_path)

    # Save summary CSV
    csv_path = output_dir / "summary.csv"
    save_summary_csv(results, csv_path)

    # Generate plots
    print("\nGenerating plots...")
    run_name = args.run_name or Path(args.predictions).stem

    plot_all_metrics(
        threshold_sweep=threshold_sweep,
        per_class_results=per_class_results['per_class'],
        confusion_data=per_class_results,
        counting_results=counting_results,
        output_dir=str(output_dir),
        run_name=run_name
    )

    # Final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nQuick Summary (at conf={best_conf_thr}):")
    print(f"  Precision: {threshold_sweep[best_conf_thr]['precision']:.4f}")
    print(f"  Recall:    {threshold_sweep[best_conf_thr]['recall']:.4f}")
    print(f"  F1 Score:  {best_f1:.4f}")
    print(f"  TP: {threshold_sweep[best_conf_thr]['tp']}, "
          f"FP: {threshold_sweep[best_conf_thr]['fp']}, "
          f"FN: {threshold_sweep[best_conf_thr]['fn']}")
    print(f"\n  Count MAE (matched): {counting_results['matched_only']['global_mae']:.4f}")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
