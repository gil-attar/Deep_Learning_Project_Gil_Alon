#!/usr/bin/env python3
"""
Crash-Safe Runner for Experiment 3

This script runs all training and evaluation sessions with proper crash recovery.
It can be resumed after interruption - completed sessions are skipped.

Usage:
    # Smoke test (1 epoch, 1 model, 2 sessions)
    python run_experiment3.py --smoke-test

    # Full experiment
    python run_experiment3.py --epochs 50

    # Resume after crash (automatically skips completed sessions)
    python run_experiment3.py --epochs 50
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "epochs": 50,
    "models": ["yolov8n", "rtdetr-l"],
    "sessions": [
        "S1_clean_train", "S2_occ_train",
        "S3_mask_backbone_early", "S4_mask_backbone_late",
        "S5_mask_neck", "S6_mask_head"
    ],
    "p_apply": 0.5,
    "p_channels": 0.2,
    "imgsz": 640,
    "batch": -1,
    "patience": 10,
    "output_root": "runs/exp3",
    "occlusion_level": "level_040",
    "continue_on_failure": True,  # Continue to next session if one fails
}

SMOKE_TEST_CONFIG = {
    "epochs": 1,
    "models": ["yolov8n"],  # Only one model
    "sessions": ["S1_clean_train", "S2_occ_train"],  # Only 2 sessions
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log(msg: str, log_file: Path = None):
    """Print and optionally write to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(line + "\n")


def is_session_done(output_dir: Path, model: str, session: str) -> bool:
    """Check if a training session is complete."""
    run_dir = output_dir / f"{model}__{session}"
    done_marker = run_dir / "DONE"
    return done_marker.exists()


def is_eval_done(output_dir: Path, model: str, session: str, test_type: str) -> bool:
    """Check if an evaluation is complete."""
    eval_dir = output_dir / "evaluations" / f"{model}__{session}__test_{test_type}"
    metrics_file = eval_dir / "metrics.json"
    return metrics_file.exists()


def mark_session_failed(output_dir: Path, model: str, session: str, error: str):
    """Mark a session as failed."""
    run_dir = output_dir / f"{model}__{session}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "FAILED").write_text(f"{datetime.now()}\n{error}")


def setup_environment():
    """Setup data and indices if needed."""
    log("Checking environment setup...")

    # Check dataset
    if not Path("data/raw/train/images").exists():
        log("ERROR: Dataset not found at data/raw/")
        log("Please run: python scripts/download_dataset.py --output_dir data/raw")
        return False

    # Build evaluation indices if needed
    if not Path("data/processed/evaluation/test_index.json").exists():
        log("Building evaluation indices...")
        result = subprocess.run([
            sys.executable, "scripts/build_evaluation_indices.py",
            "--dataset_root", "data/raw",
            "--output_dir", "data/processed/evaluation"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            log(f"ERROR building indices: {result.stderr}")
            return False

    # Generate occluded training data if needed
    if not Path("data/occluded_train_040/level_040").exists():
        log("Generating occluded training data (40%)...")
        result = subprocess.run([
            sys.executable, "scripts/generate_synthetic_occlusions.py",
            "--test_index", "data/processed/evaluation/train_index.json",
            "--images_dir", "data/raw/train/images",
            "--labels_dir", "data/raw/train/labels",
            "--output_dir", "data/occluded_train_040",
            "--levels", "0.4",
            "--seed", "42"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            log(f"ERROR generating occluded train data: {result.stderr}")
            return False

    # Generate occluded test data if needed
    if not Path("data/synthetic_occlusion/level_040").exists():
        log("Generating occluded test data (40%)...")
        result = subprocess.run([
            sys.executable, "scripts/generate_synthetic_occlusions.py",
            "--test_index", "data/processed/evaluation/test_index.json",
            "--images_dir", "data/raw/test/images",
            "--labels_dir", "data/raw/test/labels",
            "--output_dir", "data/synthetic_occlusion",
            "--levels", "0.4",
            "--seed", "42"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            log(f"ERROR generating occluded test data: {result.stderr}")
            return False

    # Create data YAML files
    create_data_yamls()

    log("Environment setup complete!")
    return True


def create_data_yamls():
    """Create data.yaml files for clean and occluded training."""
    import yaml

    # Load original config
    with open('data/raw/data.yaml', 'r') as f:
        original_config = yaml.safe_load(f)

    Path('data/processed').mkdir(parents=True, exist_ok=True)

    # Clean training
    clean_config = {
        'path': str(Path('data/raw').resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': original_config['names'],
        'nc': len(original_config['names'])
    }
    with open('data/processed/data_clean.yaml', 'w') as f:
        yaml.dump(clean_config, f, default_flow_style=False)

    # Occluded training
    occ_config = {
        'path': str(Path('data').resolve()),
        'train': 'occluded_train_040/level_040/images',
        'val': 'raw/valid/images',
        'test': 'raw/test/images',
        'names': original_config['names'],
        'nc': len(original_config['names'])
    }
    with open('data/processed/data_occ_train.yaml', 'w') as f:
        yaml.dump(occ_config, f, default_flow_style=False)


# =============================================================================
# TRAINING
# =============================================================================

def train_single_session(model_name: str, session_name: str, config: dict, log_file: Path) -> bool:
    """Train a single session. Returns True on success."""
    from ultralytics import YOLO, RTDETR
    from experiments.Experiment_3.mask_presets import get_mask_prefixes, get_session_config
    from experiments.Experiment_3.channel_masking import MaskingManager

    output_dir = Path(config['output_root'])
    run_name = f"{model_name}__{session_name}"
    run_dir = output_dir / run_name

    log(f"Training: {run_name}", log_file)

    # Get session config
    session_config = get_session_config(session_name)

    # Select data.yaml
    if session_config['train_data'] == 'occluded':
        data_yaml = 'data/processed/data_occ_train.yaml'
    else:
        data_yaml = 'data/processed/data_clean.yaml'

    log(f"  Data: {data_yaml}", log_file)
    log(f"  Epochs: {config['epochs']}", log_file)

    # Load model
    if 'yolo' in model_name.lower():
        model = YOLO(f"{model_name}.pt")
        model_type = 'yolo'
    else:
        model = RTDETR(f"{model_name}.pt")
        model_type = 'rtdetr'

    # Setup masking if needed
    masking_manager = None
    if session_config['mask_location'] is not None:
        mask_location = session_config['mask_location']
        layer_prefixes = get_mask_prefixes(model_type, mask_location)

        log(f"  Masking: {mask_location} -> {layer_prefixes}", log_file)

        masking_manager = MaskingManager(
            model.model,
            config['p_apply'],
            config['p_channels']
        )
        num_hooks = masking_manager.add_masking_to_layers(layer_prefixes)
        log(f"  Added {num_hooks} masking hooks", log_file)

    try:
        # Train
        results = model.train(
            data=data_yaml,
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            patience=config['patience'],
            save=True,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42
        )

        # Mark as done
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "DONE").touch()

        # Save masking summary
        if masking_manager:
            with open(run_dir / "masking_summary.json", 'w') as f:
                json.dump(masking_manager.get_summary(), f, indent=2)

        log(f"  SUCCESS: {run_name}", log_file)
        return True

    except Exception as e:
        log(f"  FAILED: {run_name} - {e}", log_file)
        mark_session_failed(output_dir, model_name, session_name, str(e))
        return False

    finally:
        if masking_manager:
            masking_manager.remove_all_hooks()


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_single_session(model_name: str, session_name: str, test_type: str,
                           config: dict, log_file: Path) -> dict:
    """Evaluate a single session on a test set."""
    from ultralytics import YOLO, RTDETR
    from tqdm import tqdm
    from evaluation.io import load_ground_truth, load_class_names
    from evaluation.metrics import (
        eval_detection_prf_at_iou,
        eval_per_class_metrics_and_confusions,
        eval_counting_quality
    )
    from evaluation.plots import plot_all_metrics

    output_dir = Path(config['output_root'])
    run_name = f"{model_name}__{session_name}"
    eval_name = f"{run_name}__test_{test_type}"
    eval_dir = output_dir / "evaluations" / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    log(f"Evaluating: {eval_name}", log_file)

    # Find weights
    weights_path = output_dir / run_name / "weights" / "best.pt"
    if not weights_path.exists():
        # Search for it
        found = list(output_dir.rglob(f"*{run_name}*/weights/best.pt"))
        if found:
            weights_path = found[0]
        else:
            log(f"  ERROR: Weights not found for {run_name}", log_file)
            return None

    # Load model
    if 'rtdetr' in model_name.lower():
        model = RTDETR(str(weights_path))
    else:
        model = YOLO(str(weights_path))

    # Select test images
    if test_type == 'clean':
        test_images_dir = Path("data/raw/test/images")
    else:
        test_images_dir = Path(f"data/synthetic_occlusion/{config['occlusion_level']}/images")

    # Load test index
    with open("data/processed/evaluation/test_index.json") as f:
        test_index = json.load(f)

    # Generate predictions
    predictions = []
    for img_data in tqdm(test_index['images'], desc="Inference", leave=False):
        image_path = test_images_dir / img_data['image_filename']
        if not image_path.exists():
            continue

        results = model.predict(str(image_path), conf=0.01, imgsz=640, verbose=False)[0]

        detections = []
        if len(results.boxes) > 0:
            for i in range(len(results.boxes)):
                detections.append({
                    "class_id": int(results.boxes.cls[i].item()),
                    "class_name": results.names[int(results.boxes.cls[i].item())],
                    "confidence": float(results.boxes.conf[i].item()),
                    "bbox": results.boxes.xyxy[i].tolist(),
                    "bbox_format": "xyxy"
                })

        predictions.append({
            "image_id": img_data['image_id'],
            "detections": detections
        })

    # Save predictions
    with open(eval_dir / "predictions.json", 'w') as f:
        json.dump({
            "run_id": eval_name,
            "test_type": test_type,
            "predictions": predictions
        }, f)

    # Run evaluation metrics
    gts = load_ground_truth("data/processed/evaluation/test_index.json")
    class_names = load_class_names("data/processed/evaluation/test_index.json")

    threshold_sweep = eval_detection_prf_at_iou(
        predictions, gts, iou_threshold=0.5,
        conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )

    best_thr = max(threshold_sweep.keys(), key=lambda k: threshold_sweep[k]['f1'])
    best_metrics = threshold_sweep[best_thr]

    per_class = eval_per_class_metrics_and_confusions(
        predictions, gts, conf_threshold=float(best_thr), class_names=class_names
    )

    counting = eval_counting_quality(
        predictions, gts, conf_threshold=float(best_thr), class_names=class_names
    )

    # Generate plots
    plot_all_metrics(
        threshold_sweep=threshold_sweep,
        per_class_results=per_class['per_class'],
        confusion_data=per_class,
        counting_results=counting,
        output_dir=str(eval_dir),
        run_name=eval_name
    )

    # Save metrics
    metrics = {
        "run_name": eval_name,
        "model": model_name,
        "session": session_name,
        "test_type": test_type,
        "best_threshold": float(best_thr),
        "precision": best_metrics['precision'],
        "recall": best_metrics['recall'],
        "f1": best_metrics['f1'],
        "tp": best_metrics['tp'],
        "fp": best_metrics['fp'],
        "fn": best_metrics['fn'],
        "count_mae_matched": counting['matched_only']['global_mae'],
        "count_mae_all": counting['all_predictions']['global_mae']
    }

    with open(eval_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    log(f"  F1: {metrics['f1']:.4f} @ conf={best_thr}", log_file)
    return metrics


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_experiment(config: dict):
    """Run the full experiment with crash recovery."""
    output_dir = Path(config['output_root'])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "experiment_log.txt"

    log("=" * 60, log_file)
    log("EXPERIMENT 3: Internal Masking vs Occlusion Training", log_file)
    log("=" * 60, log_file)
    log(f"Configuration:", log_file)
    log(f"  Epochs: {config['epochs']}", log_file)
    log(f"  Models: {config['models']}", log_file)
    log(f"  Sessions: {config['sessions']}", log_file)
    log(f"  Masking: p_apply={config['p_apply']}, p_channels={config['p_channels']}", log_file)
    log("=" * 60, log_file)

    # Setup environment
    if not setup_environment():
        log("Environment setup failed. Exiting.", log_file)
        return False

    # Track results
    all_metrics = []
    training_results = []

    # Phase 1: Training
    log("\n" + "=" * 40, log_file)
    log("PHASE 1: TRAINING", log_file)
    log("=" * 40, log_file)

    total_train = len(config['models']) * len(config['sessions'])
    current = 0

    for model_name in config['models']:
        for session_name in config['sessions']:
            current += 1
            run_name = f"{model_name}__{session_name}"

            # Check if already done
            if is_session_done(output_dir, model_name, session_name):
                log(f"[{current}/{total_train}] SKIP (done): {run_name}", log_file)
                training_results.append({
                    "model": model_name, "session": session_name,
                    "status": "skipped"
                })
                continue

            log(f"[{current}/{total_train}] Training: {run_name}", log_file)

            try:
                success = train_single_session(model_name, session_name, config, log_file)
                training_results.append({
                    "model": model_name, "session": session_name,
                    "status": "success" if success else "failed"
                })

                if not success and not config['continue_on_failure']:
                    log("Stopping due to failure.", log_file)
                    break

            except Exception as e:
                log(f"EXCEPTION: {e}", log_file)
                log(traceback.format_exc(), log_file)
                training_results.append({
                    "model": model_name, "session": session_name,
                    "status": "failed", "error": str(e)
                })
                if not config['continue_on_failure']:
                    break

    # Save training results
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(training_results, f, indent=2)

    # Phase 2: Evaluation
    log("\n" + "=" * 40, log_file)
    log("PHASE 2: EVALUATION", log_file)
    log("=" * 40, log_file)

    for model_name in config['models']:
        for session_name in config['sessions']:
            # Skip if training failed
            if not is_session_done(output_dir, model_name, session_name):
                log(f"SKIP eval (no weights): {model_name}__{session_name}", log_file)
                continue

            for test_type in ['clean', 'occluded']:
                # Check if already done
                if is_eval_done(output_dir, model_name, session_name, test_type):
                    log(f"SKIP (done): {model_name}__{session_name}__test_{test_type}", log_file)
                    # Load existing metrics
                    eval_dir = output_dir / "evaluations" / f"{model_name}__{session_name}__test_{test_type}"
                    with open(eval_dir / "metrics.json") as f:
                        all_metrics.append(json.load(f))
                    continue

                try:
                    metrics = evaluate_single_session(
                        model_name, session_name, test_type, config, log_file
                    )
                    if metrics:
                        all_metrics.append(metrics)
                except Exception as e:
                    log(f"EVAL FAILED: {model_name}__{session_name}__test_{test_type}: {e}", log_file)
                    log(traceback.format_exc(), log_file)

    # Save all metrics
    with open(output_dir / "all_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Phase 3: Generate summary
    log("\n" + "=" * 40, log_file)
    log("PHASE 3: SUMMARY", log_file)
    log("=" * 40, log_file)

    generate_summary(all_metrics, output_dir, log_file)

    log("\n" + "=" * 60, log_file)
    log("EXPERIMENT 3 COMPLETE!", log_file)
    log(f"Results saved to: {output_dir}", log_file)
    log("=" * 60, log_file)

    return True


def generate_summary(all_metrics: list, output_dir: Path, log_file: Path):
    """Generate summary CSV and comparison plots."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if not all_metrics:
        log("No metrics to summarize.", log_file)
        return

    df = pd.DataFrame(all_metrics)

    # Summary table
    summary_clean = df[df['test_type'] == 'clean'][['model', 'session', 'f1', 'precision', 'recall']]
    summary_clean = summary_clean.rename(columns={'f1': 'F1_clean', 'precision': 'P_clean', 'recall': 'R_clean'})

    summary_occ = df[df['test_type'] == 'occluded'][['model', 'session', 'f1', 'precision', 'recall']]
    summary_occ = summary_occ.rename(columns={'f1': 'F1_occ', 'precision': 'P_occ', 'recall': 'R_occ'})

    summary = pd.merge(summary_clean, summary_occ, on=['model', 'session'], how='outer')
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    log("\nRESULTS SUMMARY:", log_file)
    log(summary.to_string(index=False), log_file)

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for test_type in ['clean', 'occluded']:
        data = df[df['test_type'] == test_type]
        if data.empty:
            continue

        sessions = data['session'].unique()
        models = data['model'].unique()

        x = np.arange(len(sessions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, model in enumerate(models):
            model_data = data[data['model'] == model]
            values = []
            for s in sessions:
                v = model_data[model_data['session'] == s]['f1'].values
                values.append(v[0] if len(v) > 0 else 0)

            offset = width * (i - len(models)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=model)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Session')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Experiment 3: F1 Score on {test_type.title()} Test Set')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in sessions], fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f"comparison_f1_{test_type}.png", dpi=150, bbox_inches='tight')
        plt.close()

    log(f"Plots saved to: {plots_dir}", log_file)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Crash-safe runner for Experiment 3")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--smoke-test', action='store_true', help='Run quick smoke test')
    parser.add_argument('--models', nargs='+', default=None, help='Models to run')
    parser.add_argument('--sessions', nargs='+', default=None, help='Sessions to run')
    parser.add_argument('--output', type=str, default='runs/exp3', help='Output directory')
    parser.add_argument('--stop-on-failure', action='store_true', help='Stop if any session fails')

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()

    if args.smoke_test:
        config.update(SMOKE_TEST_CONFIG)
        print("Running SMOKE TEST mode")
    else:
        config['epochs'] = args.epochs

    if args.models:
        config['models'] = args.models
    if args.sessions:
        config['sessions'] = args.sessions
    if args.output:
        config['output_root'] = args.output
    if args.stop_on_failure:
        config['continue_on_failure'] = False

    # Run
    success = run_experiment(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
