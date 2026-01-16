'''
    Before running this script make sure the weights exist in /artifacts/weights in this project.
    Please refer to the README file under /experiments for more details.
'''
rom __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ultralytics import YOLO, RTDETR

# Import freeze presets from your existing file
from freezing.freeze_presets import (
    YOLOV8M_PRESETS,
    RTDETR_L_PRESETS,
    unfreeze_by_prefixes,
    count_params,
)

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../ProjectRepo
E1_ROOT = Path(__file__).resolve().parent        # .../experiments/Experiment_1

DATA_YAML = REPO_ROOT / "data" / "processed" / "data.yaml"
WEIGHTS_DIR = REPO_ROOT / "artifacts" / "weights"

RUNS_DIR = E1_ROOT / "runs"

def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_ultralytics_model(model_name: str):
    """
    model_name:
      - "yolov8m"
      - "rtdetr-l"
    """
    if model_name == "yolov8m":
        weights = WEIGHTS_DIR / "yolov8m.pt"
        return YOLO(str(weights)), "yolov8m"
    elif model_name == "rtdetr-l":
        weights = WEIGHTS_DIR / "rtdetr-l.pt"
        return RTDETR(str(weights)), "rtdetr-l"
    else:
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
    Trains with Ultralytics. Returns the train result object summary as dict (best effort).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics uses `project` + `name` or `save_dir` depending on version.
    # `project`+`name` is stable.
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
        # we kept default training parameters (LR, batch size, optimizer, etc.)
    )

    # results is a Ultralytics Results/Trainer object; serialize minimally
    out = {"status": "ok"}
    try:
        # Some versions store metrics here
        out["results"] = getattr(results, "results_dict", None)
    except Exception:
        out["results"] = None
    return out


def eval_split(ultra_model, split: str, save_dir: Path, imgsz: int) -> Dict[str, Any]:
    """
    Runs Ultralytics validation on a specific split ("val" or "test").
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
    # Ultralytics exposes a dict-like metrics summary in many versions
    try:
        out["metrics"] = results.results_dict
    except Exception:
        out["metrics"] = None
    return out


def run(model_name: str, freeze_id: str, epochs: int, imgsz: int, seed: int) -> None:
    ultra_model, model_key = load_ultralytics_model(model_name)

    # Directory: experiments/Experiment_1/runs/<model>/<freeze_id>/
    run_dir = RUNS_DIR / model_key / freeze_id

    # Apply freezing
    param_counts = apply_freeze(ultra_model, model_key=model_key, freeze_id=freeze_id)

    # Save a pre-run manifest
    manifest = {
        "experiment": "Experiment_1",
        "model": model_key,
        "freeze_id": freeze_id,
        "epochs": epochs,
        "imgsz": imgsz,
        "seed": seed,
        "timestamp_utc": utc_stamp(),
        "data_yaml": str(DATA_YAML),
        "weights_dir": str(WEIGHTS_DIR),
        "param_counts": param_counts,
    }
    save_json(run_dir / "run_manifest.json", manifest)

    # Train
    train_summary = train_one(ultra_model, save_dir=run_dir, epochs=epochs, imgsz=imgsz, seed=seed)
    save_json(run_dir / "train_summary.json", train_summary)

    # Evaluate: val + test
    val_summary = eval_split(ultra_model, split="val", save_dir=run_dir, imgsz=imgsz)
    save_json(run_dir / "val_metrics.json", val_summary)

    test_summary = eval_split(ultra_model, split="test", save_dir=run_dir, imgsz=imgsz)
    save_json(run_dir / "test_metrics.json", test_summary)

    # Single roll-up (this is what you’ll later align to Gil’s JSON schema)
    run_summary = {
        "manifest": manifest,
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
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

    # Basic existence checks to fail early (helpful for Gil on Colab)
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml at {DATA_YAML}")
    if args.model == "yolov8m" and not (WEIGHTS_DIR / "yolov8m.pt").exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_DIR / 'yolov8m.pt'}")
    if args.model == "rtdetr-l" and not (WEIGHTS_DIR / "rtdetr-l.pt").exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_DIR / 'rtdetr-l.pt'}")

    run(args.model, args.freeze, args.epochs, args.imgsz, args.seed)


if __name__ == "__main__":
    main()
