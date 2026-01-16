"""
Before running this script make sure the weights exist in /artifacts/weights in this project.
Please refer to the README file under /experiments for more details.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import platform
import yaml  # pip install pyyaml if missing

from ultralytics import YOLO, RTDETR

# Import freeze presets from your existing file
from freezing.freeze_presets import (
    YOLOV8M_PRESETS,
    RTDETR_L_PRESETS,
    unfreeze_by_prefixes,
    count_params,
)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../ProjectRepo
E1_ROOT = Path(__file__).resolve().parent        # .../experiments/Experiment_1
sys.path.insert(0, str(REPO_ROOT))               # allow `import evaluation.*`

DATA_YAML = REPO_ROOT / "data" / "processed" / "data.yaml"
WEIGHTS_DIR = REPO_ROOT / "artifacts" / "weights"
RUNS_DIR = E1_ROOT / "runs"


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_index_images(index_path: Path) -> List[Dict[str, Any]]:
    with open(index_path, "r") as f:
        data = json.load(f)
    images = data.get("images", [])
    if not images:
        raise ValueError(f"No images found in index: {index_path}")
    return images


def load_data_yaml() -> Dict[str, Any]:
    with open(DATA_YAML, "r") as f:
        return yaml.safe_load(f)


def resolve_split_images_dir(split: str) -> Path:
    """
    Map evaluator split names ('val','test') to actual folder paths from data.yaml.
    """
    cfg = load_data_yaml()
    if split not in cfg:
        raise KeyError(f"Split '{split}' not found in {DATA_YAML}. Keys: {list(cfg.keys())}")

    rel = cfg[split]  # e.g., ../valid/images or ../test/images
    return (DATA_YAML.parent / rel).resolve()


def load_ultralytics_model(model_name: str):
    """
    model_name:
      - "yolov8m"
      - "rtdetr-l"
    """
    if model_name == "yolov8m":
        weights = WEIGHTS_DIR / "yolov8m.pt"
        return YOLO(str(weights)), "yolov8m"
    if model_name == "rtdetr-l":
        weights = WEIGHTS_DIR / "rtdetr-l.pt"
        return RTDETR(str(weights)), "rtdetr-l"
    raise ValueError(f"Unsupported model_name={model_name}")


def get_presets(model_key: str) -> Dict[str, List[str]]:
    if model_key == "yolov8m":
        return YOLOV8M_PRESETS
    if model_key == "rtdetr-l":
        return RTDETR_L_PRESETS
    raise ValueError(f"Unsupported model_key={model_key}")


def apply_freeze(ultra_model, model_key: str, freeze_id: str) -> Dict[str, Any]:
    """
    Applies requires_grad based on prefixes. Returns a small dict with param counts.
    """
    presets = get_presets(model_key)
    if freeze_id not in presets:
        raise ValueError(f"freeze_id={freeze_id} not in presets={list(presets.keys())}")

    torch_model = ultra_model.model  # the underlying nn.Module
    prefixes = presets[freeze_id]
    unfreeze_by_prefixes(torch_model, prefixes)
    return count_params(torch_model)


def train_one(ultra_model, save_dir: Path, epochs: int, imgsz: int, seed: int) -> Dict[str, Any]:
    """
    Trains with Ultralytics. Returns a minimal summary dict (best effort).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    project = str(save_dir.parent)
    name = str(save_dir.name)

    results = ultra_model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        seed=seed,
        project=project,
        name=name,
        exist_ok=True,
    )

    out: Dict[str, Any] = {"status": "ok"}
    try:
        out["results"] = getattr(results, "results_dict", None)
    except Exception:
        out["results"] = None
    return out


def eval_split_ultralytics(ultra_model, split: str, save_dir: Path, imgsz: int) -> Dict[str, Any]:
    """
    Runs Ultralytics validation on a specific split ("val" or "test").
    This is kept as a sanity/logging artifact. The primary evaluation is done
    using the project's evaluator on exported predictions JSON.
    """
    results = ultra_model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        project=str(save_dir.parent),
        name=f"{save_dir.name}_{split}",
        exist_ok=True,
    )

    out: Dict[str, Any] = {"split": split}
    try:
        out["metrics"] = results.results_dict
    except Exception:
        out["metrics"] = None
    return out


def export_predictions_json(
    ultra_model,
    model_key: str,
    freeze_id: str,
    split: str,
    index_path: Path,
    out_path: Path,
    imgsz: int,
    conf_low: float,
    iou: float = 0.5,
    max_det: int = 300,
    device: str | None = None,
    half: bool | None = None,
) -> Dict[str, Any]:
    """
    Exports evaluator-compatible predictions JSON for a given split.
    - Uses *_index.json to guarantee exact image_id list.
    - bbox format: xyxy pixels.
    """
    images_dir = resolve_split_images_dir(split)
    index_images = load_index_images(index_path)

    img_paths: List[str] = []
    for item in index_images:
        fname = item["image_filename"]
        p = images_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img_paths.append(str(p))

    t0 = time.time()
    results = ultra_model.predict(
        source=img_paths,
        imgsz=imgsz,
        conf=conf_low,
        iou=iou,
        max_det=max_det,
        device=device,
        half=half,
        verbose=False,
    )
    t1 = time.time()

    preds_out: List[Dict[str, Any]] = []
    names = getattr(ultra_model, "names", None) or getattr(ultra_model.model, "names", None) or {}

    for idx_item, res in zip(index_images, results):
        image_id = idx_item["image_id"]
        dets: List[Dict[str, Any]] = []

        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                class_id = int(k)
                class_name = names.get(class_id, f"class_{class_id}")
                dets.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(c),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        preds_out.append({"image_id": image_id, "detections": dets})

    payload: Dict[str, Any] = {
        "run_id": f"Experiment_1/{model_key}/{freeze_id}",
        "split": split,
        "model_family": "yolo" if model_key.startswith("yolo") else "rtdetr",
        "inference_settings": {
            "imgsz": imgsz,
            "conf_threshold": conf_low,
            "iou_threshold": iou,
            "max_det": max_det,
            "device": device,
            "half": half,
        },
        "timing": {
            "total_inference_time_seconds": round(t1 - t0, 4),
            "num_images": len(index_images),
            "avg_inference_time_ms": round(1000.0 * (t1 - t0) / max(1, len(index_images)), 4),
        },
        "predictions": preds_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def evaluate_with_eval_tools(
    predictions_path: Path,
    gt_index_path: Path,
    out_dir: Path,
    run_name: str,
    iou_threshold: float = 0.5,
    conf_thresholds: List[float] | None = None,
    fixed_conf_for_perclass_and_counting: float | None = None,
) -> Dict[str, Any]:
    """
    Runs the project evaluation stack:
    - threshold sweep (P/R/F1)
    - per-class + confusions (at fixed_conf)
    - counting quality (at fixed_conf)
    - saves metrics.json + summary.csv + plots/*
    """
    from evaluation.io import load_predictions, load_ground_truth, load_class_names, save_metrics, save_summary_csv
    from evaluation.metrics import eval_detection_prf_at_iou, eval_per_class_metrics_and_confusions, eval_counting_quality
    from evaluation.plots import plot_all_metrics

    preds = load_predictions(str(predictions_path))
    gts = load_ground_truth(str(gt_index_path))
    class_names = load_class_names(str(gt_index_path))

    threshold_sweep = eval_detection_prf_at_iou(
        preds, gts, iou_threshold=iou_threshold, conf_thresholds=conf_thresholds
    )

    if fixed_conf_for_perclass_and_counting is None:
        # Keys in threshold_sweep are strings like "0.3"
        best_thr = max(
            (float(k) for k in threshold_sweep.keys()),
            key=lambda t: (threshold_sweep[str(t)]["f1"], -t),
        )
        fixed_conf_for_perclass_and_counting = best_thr

    per_class_results = eval_per_class_metrics_and_confusions(
        preds,
        gts,
        iou_threshold=iou_threshold,
        conf_threshold=fixed_conf_for_perclass_and_counting,
        class_names=class_names,
    )
    counting_results = eval_counting_quality(
        preds,
        gts,
        iou_threshold=iou_threshold,
        conf_threshold=fixed_conf_for_perclass_and_counting,
        class_names=class_names,
    )

    results: Dict[str, Any] = {
        "run_name": run_name,
        "predictions_path": str(predictions_path),
        "ground_truth_index_path": str(gt_index_path),
        "iou_threshold": iou_threshold,
        "threshold_sweep": threshold_sweep,
        "selected_conf_threshold": fixed_conf_for_perclass_and_counting,
        "per_class": per_class_results,
        "counting": counting_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(results, str(out_dir / "metrics.json"))
    save_summary_csv(results, str(out_dir / "summary.csv"))

    plots_dir = out_dir / "plots"
    plot_all_metrics(
        threshold_sweep=threshold_sweep,
        per_class_results=per_class_results["per_class"],
        confusion_data=per_class_results,
        counting_results=counting_results,
        output_dir=str(plots_dir),
        run_name=run_name,
    )

    return results


def run(model_name: str, freeze_id: str, epochs: int, imgsz: int, seed: int) -> None:
    ultra_model, model_key = load_ultralytics_model(model_name)

    # Directory: experiments/Experiment_1/runs/<model>/<freeze_id>/
    run_dir = RUNS_DIR / model_key / freeze_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Apply freezing
    param_counts = apply_freeze(ultra_model, model_key=model_key, freeze_id=freeze_id)

    eval_indices_dir = REPO_ROOT / "data" / "processed" / "evaluation"
    val_index = eval_indices_dir / "val_index.json"
    test_index = eval_indices_dir / "test_index.json"

    # Evaluation contract (Step A)
    conf_low = 0.01
    iou_thr = 0.5
    conf_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Save pre-run manifest
    manifest = {
        "experiment": "Experiment_1",
        "model": model_key,
        "freeze_id": freeze_id,
        "epochs": epochs,
        "imgsz": imgsz,
        "seed": seed,
        "timestamp_utc": utc_stamp(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "data_yaml": str(DATA_YAML),
        "weights_dir": str(WEIGHTS_DIR),
        "indices": {
            "val_index": str(val_index),
            "test_index": str(test_index),
        },
        "evaluation_contract": {
            "bbox_format": "xyxy_pixels",
            "iou_threshold": iou_thr,
            "prediction_export_conf_low": conf_low,
            "threshold_sweep": conf_sweep,
            "threshold_selection_split": "val",
            "threshold_selection_metric": "f1",
            "test_reporting_policy": "test_evaluated_at_val_best_threshold",
        },
        "param_counts": param_counts,
    }
    save_json(run_dir / "run_manifest.json", manifest)

    # Train
    train_summary = train_one(ultra_model, save_dir=run_dir, epochs=epochs, imgsz=imgsz, seed=seed)
    save_json(run_dir / "train_summary.json", train_summary)

    # Optional: Ultralytics val/test summaries (sanity/logging)
    val_summary = eval_split_ultralytics(ultra_model, split="val", save_dir=run_dir, imgsz=imgsz)
    save_json(run_dir / "val_metrics.json", val_summary)

    test_summary = eval_split_ultralytics(ultra_model, split="test", save_dir=run_dir, imgsz=imgsz)
    save_json(run_dir / "test_metrics.json", test_summary)

    # Export predictions JSON
    preds_dir = run_dir / "predictions"
    val_pred_path = preds_dir / "val_predictions.json"
    test_pred_path = preds_dir / "test_predictions.json"

    export_predictions_json(
        ultra_model,
        model_key=model_key,
        freeze_id=freeze_id,
        split="val",
        index_path=val_index,
        out_path=val_pred_path,
        imgsz=imgsz,
        conf_low=conf_low,
        iou=iou_thr,
    )
    export_predictions_json(
        ultra_model,
        model_key=model_key,
        freeze_id=freeze_id,
        split="test",
        index_path=test_index,
        out_path=test_pred_path,
        imgsz=imgsz,
        conf_low=conf_low,
        iou=iou_thr,
    )

    # Evaluate on VAL -> select best threshold
    val_eval_dir = run_dir / "eval" / "val"
    val_results = evaluate_with_eval_tools(
        predictions_path=val_pred_path,
        gt_index_path=val_index,
        out_dir=val_eval_dir,
        run_name=f"{model_key}-{freeze_id} (VAL)",
        iou_threshold=iou_thr,
        conf_thresholds=conf_sweep,
        fixed_conf_for_perclass_and_counting=None,
    )
    best_thr = float(val_results["selected_conf_threshold"])

    # Evaluate on TEST at VAL-best threshold
    test_eval_dir = run_dir / "eval" / "test"
    _ = evaluate_with_eval_tools(
        predictions_path=test_pred_path,
        gt_index_path=test_index,
        out_dir=test_eval_dir,
        run_name=f"{model_key}-{freeze_id} (TEST @ val_best={best_thr})",
        iou_threshold=iou_thr,
        conf_thresholds=conf_sweep,
        fixed_conf_for_perclass_and_counting=best_thr,
    )

    # Roll-up summary
    run_summary = {
        "manifest": manifest,
        "train": train_summary,
        "ultralytics_val": val_summary,
        "ultralytics_test": test_summary,
        "val_selected_threshold": best_thr,
        "artifacts": {
            "val_predictions": str(val_pred_path),
            "test_predictions": str(test_pred_path),
            "val_eval_dir": str(val_eval_dir),
            "test_eval_dir": str(test_eval_dir),
        },
    }
    save_json(run_dir / "run_summary.json", run_summary)

    print(f"[OK] Completed {model_key} {freeze_id}. Outputs in: {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolov8m", "rtdetr-l"])
    ap.add_argument("--freeze", required=True, choices=["F0", "F1", "F2", "F3"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Fail early if required inputs are missing
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml at {DATA_YAML}")

    eval_indices_dir = REPO_ROOT / "data" / "processed" / "evaluation"
    if not (eval_indices_dir / "val_index.json").exists():
        raise FileNotFoundError(f"Missing val_index.json at {eval_indices_dir / 'val_index.json'}")
    if not (eval_indices_dir / "test_index.json").exists():
        raise FileNotFoundError(f"Missing test_index.json at {eval_indices_dir / 'test_index.json'}")

    if args.model == "yolov8m" and not (WEIGHTS_DIR / "yolov8m.pt").exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_DIR / 'yolov8m.pt'}")
    if args.model == "rtdetr-l" and not (WEIGHTS_DIR / "rtdetr-l.pt").exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_DIR / 'rtdetr-l.pt'}")

    run(args.model, args.freeze, args.epochs, args.imgsz, args.seed)


if __name__ == "__main__":
    main()
