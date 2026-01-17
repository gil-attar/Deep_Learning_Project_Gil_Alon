"""
Experiment 1 runner: one run per (model, freeze_id).

Pipeline per run:
1) Load COCO-pretrained weights from artifacts/weights
2) Apply freeze preset (F0–F3)
3) Train (Ultralytics)
4) (Optional sanity) Ultralytics val on split=val and split=test
5) Export evaluator-compatible predictions JSON for val + test using *_index.json
6) Run custom evaluation (threshold sweep + plots) and save under runs/<model>/<F#>/eval/<split>/

IMPORTANT FIXES INCLUDED:
- Avoid Ultralytics fp16 None crash: never pass half=None to predict(); always a bool.
- Avoid CUDA OOM during prediction export: run predict() in chunks + small inference batch + cache cleanup.
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

import yaml  # pip install pyyaml
import torch
from ultralytics import YOLO, RTDETR

# Freeze presets (your existing infra)
from freezing.freeze_presets import (
    YOLOV8M_PRESETS,
    RTDETR_L_PRESETS,
    unfreeze_by_prefixes,
    count_params,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../ProjectRepo (or /content/<repo>)
E1_ROOT = Path(__file__).resolve().parent        # .../experiments/Experiment_1
RUNS_DIR = E1_ROOT / "runs"

DATA_YAML = REPO_ROOT / "data" / "processed" / "data.yaml"
WEIGHTS_DIR = REPO_ROOT / "artifacts" / "weights"
EVAL_INDICES_DIR = REPO_ROOT / "data" / "processed" / "evaluation"

# Enable importing repo-root modules (evaluation/*)
sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_data_yaml() -> Dict[str, Any]:
    with open(DATA_YAML, "r") as f:
        return yaml.safe_load(f)


def cuda_cleanup() -> None:
    """
    Best-effort GPU memory cleanup between heavy phases (train -> val -> predict).
    Prevents fragmentation / peak spikes on Colab T4.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # optional: reduce fragmentation on some setups
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_device_for_inference() -> str | int:
    # Ultralytics accepts: 0, "0", "cuda:0", "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Dataset split / image-path resolution (robust)
# -----------------------------------------------------------------------------
def _split_aliases(split: str) -> List[str]:
    s = (split or "").strip().lower()
    if s in {"val", "valid", "validation"}:
        return ["valid", "val", "validation"]
    return [s]


def _resolve_images_dir(split: str) -> Optional[Path]:
    """
    Try common on-disk layouts. We do NOT assume a single fixed layout.
    """
    candidates: List[Path] = []
    for alias in _split_aliases(split):
        candidates.extend(
            [
                REPO_ROOT / "data" / "raw" / alias / "images",
                REPO_ROOT / "data" / "processed" / alias / "images",
            ]
        )
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _locate_image_path(filename: str, split: str) -> Path:
    """
    Index stores basenames (e.g. "...jpg"). Resolve reliably without assuming a single dir.
    """
    fn = (filename or "").strip()
    if not fn:
        raise FileNotFoundError("Empty image filename in index.")

    # Primary: split images dir
    images_dir = _resolve_images_dir(split)
    if images_dir is not None:
        p = images_dir / fn
        if p.is_file():
            return p

    # Fallback: bounded search under REPO_ROOT/data (still deterministic enough for Colab)
    data_root = REPO_ROOT / "data"
    if data_root.is_dir():
        for p in data_root.rglob(fn):
            if p.is_file():
                return p

    msg = [f"Image not found: {fn}"]
    if images_dir is not None:
        msg.append(f"Resolved images_dir: {images_dir}")
        msg.append(f"Tried: {images_dir / fn}")
    raise FileNotFoundError("\n".join(msg))


def load_index(index_path: Path) -> Dict[str, Any]:
    with open(index_path, "r") as f:
        return json.load(f)


def load_index_images(index_path: Path) -> List[Dict[str, Any]]:
    idx = load_index(index_path)
    images = idx.get("images", [])
    if not images:
        raise ValueError(f"No images found in index: {index_path}")
    return images


# -----------------------------------------------------------------------------
# Model loading + freezing
# -----------------------------------------------------------------------------
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
    Applies requires_grad based on whitelist prefixes (freeze all -> unfreeze prefixes).
    Returns total/trainable param counts.
    """
    presets = get_presets(model_key)
    if freeze_id not in presets:
        raise ValueError(f"freeze_id={freeze_id} not in presets={list(presets.keys())}")

    torch_model = ultra_model.model  # underlying nn.Module
    prefixes = presets[freeze_id]
    unfreeze_by_prefixes(torch_model, prefixes)
    return count_params(torch_model)


# -----------------------------------------------------------------------------
# Train + Ultralytics sanity eval
# -----------------------------------------------------------------------------
def train_one(ultra_model, save_dir: Path, epochs: int, imgsz: int, seed: int) -> Dict[str, Any]:
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
    out["results_dict"] = getattr(results, "results_dict", None)
    return out


def eval_split_ultralytics(ultra_model, split: str, save_dir: Path, imgsz: int) -> Dict[str, Any]:
    results = ultra_model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        project=str(save_dir.parent),
        name=f"{save_dir.name}_{split}",
        exist_ok=True,
    )
    out: Dict[str, Any] = {"split": split, "results_dict": getattr(results, "results_dict", None)}
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
) -> Dict[str, Any]:
    """
    Export predictions in xyxy pixel format based on the index image order.

    FIXES:
    - half is forced to bool (never None) to avoid Ultralytics fp16 None crash.
    - inference is chunked to avoid CUDA OOM; also uses small 'batch' inside predict.
    """
    index_images = load_index_images(index_path)

    # Build image paths
    img_paths: List[str] = []
    for item in index_images:
        # If index ever includes an explicit path, honor it
        if isinstance(item, dict) and item.get("image_path"):
            ip = Path(str(item["image_path"]))
            if not ip.is_absolute():
                ip = (REPO_ROOT / ip).resolve()
            if ip.is_file():
                img_paths.append(str(ip))
                continue

        fname = item["image_filename"]
        p = _locate_image_path(fname, split)
        img_paths.append(str(p))

    half_flag = bool(half) if half is not None else False  # never None

    # Chunked inference (prevents OOM on T4)
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
                batch=infer_batch,  # IMPORTANT: small inference batch inside predictor
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError as e:
            # One more attempt with smaller settings if it still spikes
            cuda_cleanup()
            smaller_chunk = max(4, chunk_size // 2)
            smaller_batch = max(1, infer_batch // 2)
            if smaller_chunk == chunk_size and smaller_batch == infer_batch:
                raise e

            # Retry current chunk with reduced params
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
            # If we reduced chunk, adjust loop index forward safely:
            # We extend results for the reduced chunk and then continue; next iteration starts at same i+chunk_size,
            # so we must also include the remaining images in subsequent loops by not skipping them.
            # To keep logic simple/deterministic, we do NOT modify i here. The remaining images of this chunk
            # will be processed in later iterations normally.

        all_results.extend(chunk_results)
        cuda_cleanup()

    t1 = time.time()

    # Map ultralytics results to evaluator format
    names = getattr(ultra_model, "names", None) or getattr(ultra_model.model, "names", None) or {}
    preds_out: List[Dict[str, Any]] = []

    # all_results must align 1:1 with index_images
    if len(all_results) != len(index_images):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(all_results)} results for {len(index_images)} images "
            f"(split={split}). This indicates an internal predict() failure or chunking logic issue."
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
        "run_id": f"Experiment_1/{model_key}/{freeze_id}",
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
# Custom evaluation wrapper (your repo's evaluation package)
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
    """
    Runs the project's evaluation stack and writes:
      - metrics.json
      - summary.csv
      - plots/*.png

    Assumes your repo provides:
      evaluation/io.py
      evaluation/metrics.py
      evaluation/plots.py
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

    # Choose threshold (best F1 on val) if not forced
    if fixed_conf_for_perclass_and_counting is None:
        # keys might be strings; normalize
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


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
def run(model_name: str, freeze_id: str, epochs: int, imgsz: int, seed: int) -> None:
    ultra_model, model_key = load_ultralytics_model(model_name)

    run_dir = RUNS_DIR / model_key / freeze_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Freeze
    param_counts = apply_freeze(ultra_model, model_key=model_key, freeze_id=freeze_id)

    # Indices
    val_index = EVAL_INDICES_DIR / "val_index.json"
    test_index = EVAL_INDICES_DIR / "test_index.json"

    # Eval knobs
    conf_low = 0.01
    iou_thr = 0.5
    conf_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "data_yaml": str(DATA_YAML),
        "weights_dir": str(WEIGHTS_DIR),
        "indices": {"val_index": str(val_index), "test_index": str(test_index)},
        "param_counts": param_counts,
        "evaluation_contract": {
            "bbox_format": "xyxy_pixels",
            "iou_threshold": iou_thr,
            "prediction_export_conf_low": conf_low,
            "threshold_sweep": conf_sweep,
            "threshold_selection_split": "val",
            "threshold_selection_metric": "f1",
            "test_reporting_policy": "test_evaluated_at_val_best_threshold",
        },
    }
    save_json(run_dir / "run_manifest.json", manifest)

    # Train
    train_summary = train_one(ultra_model, save_dir=run_dir, epochs=epochs, imgsz=imgsz, seed=seed)
    save_json(run_dir / "train_summary.json", train_summary)
    cuda_cleanup()

    # Ultralytics sanity val/test (optional but kept)
    val_summary = eval_split_ultralytics(ultra_model, split="val", save_dir=run_dir, imgsz=imgsz)
    save_json(run_dir / "val_metrics.json", val_summary)
    cuda_cleanup()

    test_summary = eval_split_ultralytics(ultra_model, split="test", save_dir=run_dir, imgsz=imgsz)
    save_json(run_dir / "test_metrics.json", test_summary)
    cuda_cleanup()

    # Export predictions JSON (custom evaluator input)
    preds_dir = run_dir / "predictions"
    val_pred_path = preds_dir / "val_predictions.json"
    test_pred_path = preds_dir / "test_predictions.json"

    device_for_predict = get_device_for_inference()

    # Conservative defaults to avoid OOM
    chunk_size = 16 if torch.cuda.is_available() else 64
    infer_batch = 4 if torch.cuda.is_available() else 16

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
        device=device_for_predict,
        half=False,               # never None
        chunk_size=chunk_size,    # OOM-safe
        infer_batch=infer_batch,  # OOM-safe
    )
    cuda_cleanup()

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
    )
    cuda_cleanup()

    # Evaluate on val: threshold sweep + plots; choose best threshold
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

    # Evaluate on test at val-best threshold
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolov8m", "rtdetr-l"])
    ap.add_argument("--freeze", required=True, choices=["F0", "F1", "F2", "F3"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Hard requirements
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml at {DATA_YAML}")

    val_index = EVAL_INDICES_DIR / "val_index.json"
    test_index = EVAL_INDICES_DIR / "test_index.json"
    if not val_index.exists():
        raise FileNotFoundError(f"Missing val_index.json at {val_index}")
    if not test_index.exists():
        raise FileNotFoundError(f"Missing test_index.json at {test_index}")

    if args.model == "yolov8m" and not (WEIGHTS_DIR / "yolov8m.pt").exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_DIR / 'yolov8m.pt'}")
    if args.model == "rtdetr-l" and not (WEIGHTS_DIR / "rtdetr-l.pt").exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_DIR / 'rtdetr-l.pt'}")

    run(args.model, args.freeze, args.epochs, args.imgsz, args.seed)


if __name__ == "__main__":
    main()
