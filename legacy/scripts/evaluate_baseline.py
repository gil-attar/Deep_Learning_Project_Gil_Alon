"""
Baseline Model Evaluation Script - Step 3.2

Generates all 6 required JSON files:
1. baseline_yolo_run.json (run metadata)
2. baseline_rtdetr_run.json (run metadata)
3. baseline_yolo_metrics.json (aggregate metrics)
4. baseline_rtdetr_metrics.json (aggregate metrics)
5. baseline_yolo_predictions.json (per-image predictions)
6. baseline_rtdetr_predictions.json (per-image predictions)
"""

import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import platform

from ultralytics import YOLO, RTDETR
import torch
from tqdm import tqdm


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def get_environment_info():
    """Get hardware and software environment information."""
    import ultralytics
    import sys

    env_info = {
        "platform": "google_colab" if 'google.colab' in sys.modules else "local_wsl",
        "gpu": "None",
        "python_version": sys.version.split()[0],
        "ultralytics_version": ultralytics.__version__,
        "pytorch_version": torch.__version__
    }

    # Get GPU info
    if torch.cuda.is_available():
        env_info["gpu"] = torch.cuda.get_device_name(0)

    return env_info


def create_run_metadata(model_family, model_name, weights_path, dataset_root, output_dir):
    """
    Create run metadata JSON (Step 3.1 - Protocol Freezing).

    Args:
        model_family: "yolo" or "rtdetr"
        model_name: "yolov8n" or "rtdetr-l"
        weights_path: Path to trained weights
        dataset_root: Root directory of dataset
        output_dir: Output directory for evaluation artifacts
    """
    env_info = get_environment_info()
    git_commit = get_git_commit()

    run_id = f"baseline_{model_family}_001"

    run_metadata = {
        "run_id": run_id,
        "stage": "3.2_baseline",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": git_commit,

        "model": {
            "model_family": model_family,
            "model_name": model_name,
            "weights_path": str(weights_path)
        },

        "dataset": {
            "split_manifest_path": "data/processed/splits/split_manifest.json",
            "test_index_path": "data/processed/evaluation/test_index.json",
            "data_yaml_path": "data/processed/data.yaml",
            "num_classes": 26,
            "class_names_source": "data_yaml"
        },

        "inference_settings": {
            "imgsz": 640,
            "conf_threshold": 0.25,
            "iou_threshold": 0.50,
            "max_det": 300,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "half": torch.cuda.is_available()
        },

        "evaluation_settings": {
            "metrics": ["mAP@50", "mAP@50-95", "precision", "recall", "fps"],
            "notes": "Baseline evaluation on full test set (400 images) with fixed difficulty labels"
        },

        "outputs": {
            "metrics_path": f"evaluation/metrics/baseline_{model_family}_metrics.json",
            "predictions_path": f"evaluation/metrics/baseline_{model_family}_predictions.json",
            "plots_dir": f"evaluation/plots/{model_family}_baseline"
        },

        "hardware": {
            "platform": env_info["platform"],
            "gpu": env_info["gpu"],
            "ultralytics_version": env_info["ultralytics_version"],
            "python_version": env_info["python_version"],
            "pytorch_version": env_info["pytorch_version"]
        }
    }

    # Save run metadata
    output_path = Path(output_dir) / f"baseline_{model_family}_run.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(run_metadata, f, indent=2)

    print(f"✓ Saved run metadata: {output_path}")
    return run_metadata


def evaluate_model(model, test_index_path, conf_threshold=0.25, imgsz=640):
    """
    Run inference on all test images and collect predictions.

    Returns:
        predictions: List of per-image predictions
        timing_info: Inference timing statistics
    """
    # Load test index
    with open(test_index_path, 'r') as f:
        test_index = json.load(f)

    test_images = test_index['images']
    print(f"Evaluating on {len(test_images)} test images...")

    predictions = []
    inference_times = []

    for img_data in tqdm(test_images, desc="Running inference"):
        image_id = img_data['image_id']
        image_filename = img_data['image_filename']

        # Construct image path (adjust based on your structure)
        # Assumes images are in data/raw/test/images/
        image_path = Path("data/raw/test/images") / image_filename

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Run inference with timing
        start_time = time.time()
        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False
        )[0]
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Extract detections
        detections = []
        if len(results.boxes) > 0:
            boxes = results.boxes
            for i in range(len(boxes)):
                detection = {
                    "class_id": int(boxes.cls[i].item()),
                    "class_name": results.names[int(boxes.cls[i].item())],
                    "confidence": float(boxes.conf[i].item()),
                    "bbox": boxes.xyxy[i].tolist(),  # xyxy format
                    "bbox_format": "xyxy"
                }
                detections.append(detection)

        # Store prediction for this image
        predictions.append({
            "image_id": image_id,
            "image_path": f"data/raw/test/images/{image_filename}",
            "detections": detections,
            "num_detections": len(detections),
            "inference_time_ms": inference_time * 1000
        })

    # Compute timing statistics
    total_time = sum(inference_times)
    avg_time = total_time / len(inference_times) if inference_times else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0

    timing_info = {
        "total_inference_time_seconds": total_time,
        "avg_inference_time_ms": avg_time * 1000,
        "fps": fps,
        "num_images": len(test_images)
    }

    return predictions, timing_info


def compute_validation_metrics(model, data_yaml_path, imgsz=640):
    """
    Run Ultralytics validation to get mAP, precision, recall.

    Returns dict with validation metrics.
    """
    print("Running validation to compute mAP metrics...")

    # Run validation
    metrics = model.val(
        data=str(data_yaml_path),
        imgsz=imgsz,
        split='test',  # Use test split
        verbose=False
    )

    # Extract key metrics
    # Ultralytics provides metrics.box for detection metrics
    validation_metrics = {
        "map50": float(metrics.box.map50),  # mAP@IoU=0.50
        "map50_95": float(metrics.box.map),  # mAP@IoU=0.50:0.95
        "precision": float(metrics.box.mp),  # Mean precision
        "recall": float(metrics.box.mr),  # Mean recall
    }

    return validation_metrics


def create_metrics_json(run_metadata, validation_metrics, timing_info, output_dir, model_family):
    """
    Create aggregate metrics JSON (Step 3.2 - Baseline Performance).
    """
    metrics_json = {
        "run_id": run_metadata["run_id"],
        "model_name": run_metadata["model"]["model_name"],
        "test_index_path": run_metadata["dataset"]["test_index_path"],
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "metrics": {
            "map50": validation_metrics["map50"],
            "map50_95": validation_metrics["map50_95"],
            "precision": validation_metrics["precision"],
            "recall": validation_metrics["recall"],
            "fps": timing_info["fps"]
        },

        "timing": {
            "total_inference_time_seconds": timing_info["total_inference_time_seconds"],
            "avg_inference_time_ms": timing_info["avg_inference_time_ms"],
            "fps": timing_info["fps"],
            "num_images": timing_info["num_images"]
        },

        "inference_settings": run_metadata["inference_settings"],
        "hardware": run_metadata["hardware"]
    }

    # Save metrics JSON
    output_path = Path(output_dir) / f"baseline_{model_family}_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✓ Saved metrics: {output_path}")
    return metrics_json


def create_predictions_json(predictions, run_metadata, output_dir, model_family):
    """
    Create per-image predictions JSON.
    """
    predictions_json = {
        "run_id": run_metadata["run_id"],
        "model_name": run_metadata["model"]["model_name"],
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_images": len(predictions),
        "inference_settings": run_metadata["inference_settings"],
        "bbox_format": "xyxy",
        "predictions": predictions
    }

    # Save predictions JSON
    output_path = Path(output_dir) / f"baseline_{model_family}_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(predictions_json, f, indent=2)

    print(f"✓ Saved predictions: {output_path}")
    return predictions_json


def evaluate_baseline_model(
    model_family,
    model_name,
    weights_path,
    dataset_root,
    output_dir,
    conf_threshold=0.25,
    imgsz=640
):
    """
    Complete baseline evaluation pipeline for one model.

    Generates 3 JSON files:
    1. Run metadata
    2. Aggregate metrics
    3. Per-image predictions
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} ({model_family.upper()})")
    print(f"{'='*60}\n")

    # Paths
    test_index_path = Path(dataset_root) / "processed/evaluation/test_index.json"
    data_yaml_path = Path(dataset_root) / "processed/data.yaml"

    # Step 1: Create run metadata
    print("Step 1: Creating run metadata...")
    run_metadata = create_run_metadata(
        model_family=model_family,
        model_name=model_name,
        weights_path=weights_path,
        dataset_root=dataset_root,
        output_dir=output_dir
    )

    # Step 2: Load model
    print(f"\nStep 2: Loading model from {weights_path}...")
    if model_family == "yolo":
        model = YOLO(weights_path)
    elif model_family == "rtdetr":
        model = RTDETR(weights_path)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    # Step 3: Run validation to get mAP metrics
    print("\nStep 3: Computing validation metrics (mAP, precision, recall)...")
    validation_metrics = compute_validation_metrics(
        model=model,
        data_yaml_path=data_yaml_path,
        imgsz=imgsz
    )

    # Step 4: Run inference on all test images
    print("\nStep 4: Running inference on test set...")
    predictions, timing_info = evaluate_model(
        model=model,
        test_index_path=test_index_path,
        conf_threshold=conf_threshold,
        imgsz=imgsz
    )

    # Step 5: Create metrics JSON
    print("\nStep 5: Creating metrics JSON...")
    metrics_json = create_metrics_json(
        run_metadata=run_metadata,
        validation_metrics=validation_metrics,
        timing_info=timing_info,
        output_dir=output_dir,
        model_family=model_family
    )

    # Step 6: Create predictions JSON
    print("\nStep 6: Creating predictions JSON...")
    predictions_json = create_predictions_json(
        predictions=predictions,
        run_metadata=run_metadata,
        output_dir=output_dir,
        model_family=model_family
    )

    print(f"\n{'='*60}")
    print(f"✓ {model_name} evaluation complete!")
    print(f"{'='*60}")
    print(f"\nMetrics Summary:")
    print(f"  mAP@50:     {validation_metrics['map50']:.4f}")
    print(f"  mAP@50-95:  {validation_metrics['map50_95']:.4f}")
    print(f"  Precision:  {validation_metrics['precision']:.4f}")
    print(f"  Recall:     {validation_metrics['recall']:.4f}")
    print(f"  FPS:        {timing_info['fps']:.2f}")
    print()

    return {
        "run_metadata": run_metadata,
        "metrics": metrics_json,
        "predictions": predictions_json
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models and generate JSON outputs for Step 3.2"
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="models/yolov8n_baseline.pt",
        help="Path to YOLOv8 weights"
    )
    parser.add_argument(
        "--rtdetr_weights",
        type=str,
        default="models/rtdetr_baseline.pt",
        help="Path to RT-DETR weights"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data",
        help="Dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/metrics",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo", "rtdetr", "both"],
        default="both",
        help="Which model(s) to evaluate"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION - Step 3.2")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset root:      {args.dataset_root}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  Image size:        {args.imgsz}")
    print(f"  Device:            {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    results = {}

    # Evaluate YOLO
    if args.model in ["yolo", "both"]:
        if Path(args.yolo_weights).exists():
            results["yolo"] = evaluate_baseline_model(
                model_family="yolo",
                model_name="yolov8n",
                weights_path=args.yolo_weights,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                conf_threshold=args.conf_threshold,
                imgsz=args.imgsz
            )
        else:
            print(f"\n⚠ YOLOv8 weights not found: {args.yolo_weights}")

    # Evaluate RT-DETR
    if args.model in ["rtdetr", "both"]:
        if Path(args.rtdetr_weights).exists():
            results["rtdetr"] = evaluate_baseline_model(
                model_family="rtdetr",
                model_name="rtdetr-l",
                weights_path=args.rtdetr_weights,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                conf_threshold=args.conf_threshold,
                imgsz=args.imgsz
            )
        else:
            print(f"\n⚠ RT-DETR weights not found: {args.rtdetr_weights}")

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files in {args.output_dir}/:")

    if "yolo" in results:
        print("\nYOLO:")
        print("  ✓ baseline_yolo_run.json")
        print("  ✓ baseline_yolo_metrics.json")
        print("  ✓ baseline_yolo_predictions.json")

    if "rtdetr" in results:
        print("\nRT-DETR:")
        print("  ✓ baseline_rtdetr_run.json")
        print("  ✓ baseline_rtdetr_metrics.json")
        print("  ✓ baseline_rtdetr_predictions.json")

    print("\n✓ All JSON files are ready for Step 3.3 (occlusion analysis)")
    print()


if __name__ == "__main__":
    main()
