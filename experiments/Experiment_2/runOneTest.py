"""
Experiment 2 runner: one run per (model, freeze_id, epochs) with STRICT absolute paths.

Key differences vs Experiment 1 runner:
- E2 is TEST-only (no val_index dependency).
- E2 reports metrics at a FIXED confidence threshold (--report-conf) to keep epochs as the variable.
- Adds training wall-clock timing and seconds/epoch.
- Run directory includes epoch budget: runs/<model>/<freeze>/E<epochs>/...

Ultralytics caveat:
- Ultralytics may change working directories internally. This file therefore uses absolute, resolved paths everywhere.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from ultralytics import RTDETR, YOLO

# -----------------------------------------------------------------------------
# Absolute paths (CWD-agnostic)
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
E2_ROOT = THIS_FILE.parent.resolve()              # .../experiments/Experiment_2
REPO_ROOT = THIS_FILE.parents[2].resolve()        # .../<repo_root>

RUNS_DIR = (E2_ROOT / "runs").resolve()

DATA_YAML = (REPO_ROOT / "data" / "processed" / "data.yaml").resolve()
WEIGHTS_DIR = (REPO_ROOT / "artifacts" / "weights").resolve()
EVAL_INDICES_DIR = (REPO_ROOT / "data" / "processed" / "evaluation").resolve()

# Ensure repo modules import regardless of CWD
sys.path.insert(0, str(REPO_ROOT))

# Freeze presets currently live under Experiment_1/freezing/*
E1_ROOT = (REPO_ROOT / "experiments" / "Experiment_1").resolve()
sys.path.insert(0, str(E1_ROOT))

from freezing.freeze_presets import (  # noqa: E402 # type: ignore
    RTDETR_L_PRESETS,
    YOLOV8M_PRESETS,
    count_params,
    unfreeze_by_prefixes,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_device_for_inference() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def load_index_images(index_path: Path) -> List[Dict[str, Any]]:
    index_path = index_path.resolve()
    with open(index_path, "r") as f:
        idx = json.load(f)
    images = idx.get("images", [])
    if not images:
        raise ValueError(f"No images found in index: {index_path}")
    return images


def _resolve_images_dir(split: str) -> Optional[Path]:
    s = (split or "").strip().lower()
    aliases = ["valid", "val", "validation"] if s in {"val", "valid", "validation"} else [s]

    candidates: List[Path] = []
    for a in aliases:
        candidates.extend(
            [
                (REPO_ROOT / "data" / "raw" / a / "images").resolve(),
                (REPO_ROOT / "data" / "processed" / a / "images").resolve(),
            ]
        )
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _locate_image_path(filename: str, split: str) -> Path:
    fn = (filename or "").strip()
    if not fn:
        raise FileNotFoundError("Empty image filename in index.")

    images_dir = _resolve_images_dir(split)
    if images_dir is not None:
        p = (images_dir / fn).resolve()
        if p.is_file():
            return p

    # Fallback: search under repo's data/ subtree
    data_root = (REPO_ROOT / "data").resolve()
    if data_root.is_dir():
        for p in data_root.rglob(fn):
            if p.is_file():
                return p.resolve()

    msg = [f"Image not found: {fn}"]
    if images_dir is not None:
        msg.append(f"Resolved images_dir: {images_dir}")
        msg.append(f"Tried: {(images_dir / fn).resolve()}")
    raise FileNotFoundError("\n".join(msg))


# -----------------------------------------------------------------------------
# Model loading + freezing
# -----------------------------------------------------------------------------
def load_ultralytics_model(model_name: str):
    if model_name == "yolov8m":
        weights = (WEIGHTS_DIR / "yolov8m.pt").resolve()
        return YOLO(str(weights)), "yolov8m"
    if model_name == "rtdetr-l":
        weights = (WEIGHTS_DIR / "rtdetr-l.pt").resolve()
        return RTDETR(str(weights)), "rtdetr-l"
    raise ValueError(f"Unsupported model_name={model_name}")


def get_presets(model_key: str) -> Dict[str, List[str]]:
    if model_key == "yolov8m":
        return YOLOV8M_PRESETS
    if model_key == "rtdetr-l":
        return RTDETR_L_PRESETS
    raise ValueError(f"Unsupported model_key={model_key}")


def apply_freeze(ultra_model, model_key: str, freeze_id: str) -> Dict[str, Any]:
    presets = get_presets(model_key)
    if freeze_id not in presets:
        raise ValueError(f"freeze_id={freeze_id} not in presets={list(presets.keys())}")

    torch_model = ultra_model.model
    prefixes = presets[freeze_id]
    unfreeze_by_prefixes(torch_model, prefixes)
    return count_params(torch_model)


# -----------------------------------------------------------------------------
# Train (absolute project dir)
# -----------------------------------------------------------------------------
def train_one(ultra_model, save_dir: Path, epochs: int, imgsz: int, seed: int) -> Dict[str, Any]:
    save_dir = save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics uses project/name. Use absolute project to avoid CWD issues.
    project = str(save_dir.parent.resolve())
    name = save_dir.name

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
    out["results_dict"] = getattr(results, "results_dict", None)
    return out


# -----------------------------------------------------------------------------
# Prediction export (evaluator-compatible)
# -----------------------------------------------------------------------------
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
    device: str | int | None = None,
    half: bool | None = None,
    chunk_size: int = 32,
    infer_batch: int = 8,
    run_id_prefix: str = "Experiment_2",
) -> Dict[str, Any]:
    index_path = index_path.resolve()
    out_path = out_path.resolve()

    index_images = load_index_images(index_path)

    # Build absolute image paths
    img_paths: List[str] = []
    for item in index_images:
        # honor explicit image_path if present
        if isinstance(item, dict) and item.get("image_path"):
            ip = Path(str(item["image_path"]))
            if not ip.is_absolute():
                ip = (REPO_ROOT / ip).resolve()
            if ip.is_file():
                img_paths.append(str(ip.resolve()))
                continue

        fname = item["image_filename"]
        p = _locate_image_path(fname, split)
        img_paths.append(str(p.resolve()))

    half_flag = bool(half) if half is not None else False  # never None

    t0 = time.time()
    all_results = []
    for i in range(0, len(img_paths), chunk_size):
        chunk = img_paths[i : i + chunk_size]
        try:
            chunk_results = ultra_model.predict(
                source=chunk,
                imgsz=imgsz,
                conf=conf_low,
                iou=iou,
                max_det=max_det,
                device=device,
                half=half_flag,
                batch=infer_batch,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError as e:
            cuda_cleanup()
            smaller_chunk = max(4, chunk_size // 2)
            smaller_batch = max(1, infer_batch // 2)
            if smaller_chunk == chunk_size and smaller_batch == infer_batch:
                raise e
            chunk = img_paths[i : i + smaller_chunk]
            chunk_results = ultra_model.predict(
                source=chunk,
                imgsz=imgsz,
                conf=conf_low,
                iou=iou,
                max_det=max_det,
                device=device,
                half=half_flag,
                batch=smaller_batch,
                verbose=False,
            )

        all_results.extend(chunk_results)
        cuda_cleanup()

    t1 = time.time()

    names = getattr(ultra_model, "names", None) or getattr(ultra_model.model, "names", None) or {}
    preds_out: List[Dict[str, Any]] = []

    if len(all_results) != len(index_images):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(all_results)} results for {len(index_images)} images (split={split})."
        )

    for idx_item, res in zip(index_images, all_results):
        image_id = idx_item["image_id"]
        dets: List[Dict[str, Any]] = []

        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            clss = boxes.cls.detach().cpu().numpy()

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                class_id = int(k)
                dets.append(
                    {
                        "class_id": class_id,
                        "class_name": names.get(class_id, f"class_{class_id}"),
                        "confidence": float(c),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        preds_out.append({"image_id": image_id, "detections": dets})

    payload: Dict[str, Any] = {
        "run_id": f"{run_id_prefix}/{model_key}/{freeze_id}",
        "split": split,
        "model_family": "yolo" if model_key.startswith("yolo") else "rtdetr",
        "inference_settings": {
            "imgsz": imgsz,
            "conf_threshold": conf_low,
            "iou_threshold": iou,
            "max_det": max_det,
            "device": device,
            "half": half_flag,
            "chunk_size": chunk_size,
            "infer_batch": infer_batch,
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


# -----------------------------------------------------------------------------
# Custom evaluation wrapper (same API as E1; E2 enforces fixed threshold)
# -----------------------------------------------------------------------------
def evaluate_with_eval_tools(
    predictions_path: Path,
    gt_index_path: Path,
    out_dir: Path,
    run_name: str,
    iou_threshold: float = 0.5,
    conf_thresholds: Optional[List[float]] = None,
    fixed_conf_for_perclass_and_counting: Optional[float] = None,
) -> Dict[str, Any]:
    from evaluation.io import load_predictions, load_ground_truth, load_class_names, save_metrics, save_summary_csv
    from evaluation.metrics import eval_counting_quality, eval_detection_prf_at_iou, eval_per_class_metrics_and_confusions
    from evaluation.plots import plot_all_metrics

    predictions_path = predictions_path.resolve()
    gt_index_path = gt_index_path.resolve()
    out_dir = out_dir.resolve()

    preds = load_predictions(str(predictions_path))
    gts = load_ground_truth(str(gt_index_path))
    class_names = load_class_names(str(gt_index_path))

    threshold_sweep = eval_detection_prf_at_iou(
        preds, gts, iou_threshold=iou_threshold, conf_thresholds=conf_thresholds
    )

    # E2: no selection-on-val; must be forced
    if fixed_conf_for_perclass_and_counting is None:
        raise ValueError("E2 requires fixed_conf_for_perclass_and_counting (do not select threshold on test).")

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
    save_metrics(results, str((out_dir / "metrics.json").resolve()))
    save_summary_csv(results, str((out_dir / "summary.csv").resolve()))

    plots_dir = (out_dir / "plots").resolve()
    plot_all_metrics(
        threshold_sweep=threshold_sweep,
        per_class_results=per_class_results["per_class"],
        confusion_data=per_class_results,
        counting_results=counting_results,
        output_dir=str(plots_dir),
        run_name=run_name,
    )

    return results


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
def run(model_name: str, freeze_id: str, epochs: int, imgsz: int, seed: int, report_conf: float) -> None:
    ultra_model, model_key = load_ultralytics_model(model_name)

    run_dir = (RUNS_DIR / model_key / freeze_id / f"E{epochs}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Apply freeze preset
    param_counts = apply_freeze(ultra_model, model_key=model_key, freeze_id=freeze_id)

    # Indices (E2 uses TEST only)
    test_index = (EVAL_INDICES_DIR / "test_index.json").resolve()

    # Eval knobs
    conf_low = 0.01
    iou_thr = 0.5
    conf_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    manifest = {
        "experiment": "Experiment_2",
        "option": "A_epoch_budget_sweep",
        "model": model_key,
        "freeze_id": freeze_id,
        "epochs": epochs,
        "imgsz": imgsz,
        "seed": seed,
        "timestamp_utc": utc_stamp(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "paths": {
            "repo_root": str(REPO_ROOT),
            "runs_dir": str(RUNS_DIR),
            "run_dir": str(run_dir),
        },
        "data_yaml": str(DATA_YAML),
        "weights_dir": str(WEIGHTS_DIR),
        "indices": {"test_index": str(test_index)},
        "param_counts": param_counts,
        "evaluation_contract": {
            "bbox_format": "xyxy_pixels",
            "iou_threshold": iou_thr,
            "prediction_export_conf_low": conf_low,
            "threshold_sweep": conf_sweep,
            "threshold_selection_policy": "fixed",
            "reporting_conf_threshold": report_conf,
            "test_reporting_policy": "test_evaluated_at_fixed_threshold",
        },
    }
    save_json(run_dir / "run_manifest.json", manifest)

    # --------------------------
    # Train with wall-clock time
    # --------------------------
    train_t0_iso = utc_iso()
    t0 = time.time()
    train_summary = train_one(ultra_model, save_dir=run_dir, epochs=epochs, imgsz=imgsz, seed=seed)
    t1 = time.time()
    train_t1_iso = utc_iso()

    train_wall = float(t1 - t0)
    train_summary["timing"] = {
        "train_start_utc": train_t0_iso,
        "train_end_utc": train_t1_iso,
        "train_wall_time_seconds": round(train_wall, 4),
        "train_seconds_per_epoch": round(train_wall / max(1, epochs), 4),
    }
    save_json(run_dir / "train_summary.json", train_summary)
    cuda_cleanup()

    # --------------------------
    # Export predictions JSON (TEST)
    # --------------------------
    preds_dir = (run_dir / "predictions").resolve()
    test_pred_path = (preds_dir / "test_predictions.json").resolve()

    device_for_predict = get_device_for_inference()

    # Conservative defaults to avoid OOM
    chunk_size = 16 if torch.cuda.is_available() else 64
    infer_batch = 4 if torch.cuda.is_available() else 16

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
        device=device_for_predict,
        half=False,               # never None
        chunk_size=chunk_size,    # OOM-safe
        infer_batch=infer_batch,  # OOM-safe
        run_id_prefix="Experiment_2",
    )
    cuda_cleanup()

    # --------------------------
    # Evaluate (TEST @ fixed threshold)
    # --------------------------
    test_eval_dir = (run_dir / "eval" / "test").resolve()
    eval_results = evaluate_with_eval_tools(
        predictions_path=test_pred_path,
        gt_index_path=test_index,
        out_dir=test_eval_dir,
        run_name=f"{model_key}-{freeze_id}-E{epochs} (TEST fixed_conf={report_conf})",
        iou_threshold=iou_thr,
        conf_thresholds=conf_sweep,
        fixed_conf_for_perclass_and_counting=report_conf,
    )

    # Summary artifact for resume logic + bookkeeping
    run_summary = {
        "manifest": manifest,
        "train": train_summary,
        "eval": {
            "test": {
                "selected_conf_threshold": report_conf,
                "eval_dir": str(test_eval_dir),
            }
        },
        "artifacts": {
            "test_predictions": str(test_pred_path),
            "test_eval_metrics": str((test_eval_dir / "metrics.json").resolve()),
            "test_eval_summary_csv": str((test_eval_dir / "summary.csv").resolve()),
        },
    }
    save_json(run_dir / "run_summary.json", run_summary)

    print(f"[OK] Completed E2 {model_key} {freeze_id} E{epochs}. Outputs in: {run_dir}")
    _ = eval_results  # keep variable for debugging if you breakpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolov8m", "rtdetr-l"])
    ap.add_argument("--freeze", required=True, choices=["F0", "F1", "F2", "F3"])
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--report-conf",
        type=float,
        default=0.25,
        help="Fixed confidence threshold used for per-class and counting metrics (E2 fairness).",
    )
    args = ap.parse_args()

    # Hard requirements (absolute paths)
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml at {DATA_YAML}")

    test_index = (EVAL_INDICES_DIR / "test_index.json").resolve()
    if not test_index.exists():
        raise FileNotFoundError(f"Missing test_index.json at {test_index}")

    if args.model == "yolov8m" and not (WEIGHTS_DIR / "yolov8m.pt").exists():
        raise FileNotFoundError(f"Missing weights: {(WEIGHTS_DIR / 'yolov8m.pt').resolve()}")
    if args.model == "rtdetr-l" and not (WEIGHTS_DIR / "rtdetr-l.pt").exists():
        raise FileNotFoundError(f"Missing weights: {(WEIGHTS_DIR / 'rtdetr-l.pt').resolve()}")

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if not (0.0 <= args.report_conf <= 1.0):
        raise ValueError("--report-conf must be in [0, 1]")

    run(args.model, args.freeze, args.epochs, args.imgsz, args.seed, args.report_conf)


if __name__ == "__main__":
    main()
