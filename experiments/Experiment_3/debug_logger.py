"""
Debug Logger for Experiment 3

Comprehensive logging system that tracks:
1. Masking hook activity (are hooks firing? how often?)
2. Data paths and sample images (is the right data being used?)
3. Training losses per epoch (is training progressing normally?)
4. Model state (is model in training mode when expected?)
5. Ground truth statistics (are labels loaded correctly?)
6. Prediction samples (what is the model actually predicting?)
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn


class ExperimentDebugLogger:
    """
    Comprehensive debug logger for Experiment 3.

    Writes to both a JSON file (for programmatic analysis) and
    a human-readable text file. Both are saved to Google Drive
    for crash safety.
    """

    def __init__(self, output_dir: Path, run_id: str):
        """
        Args:
            output_dir: Directory to save log files (should be on Drive)
            run_id: Experiment run ID for identification
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id
        self.log_path = self.output_dir / "debug_log.json"
        self.txt_path = self.output_dir / "debug_log.txt"

        # Initialize log structure
        self.log_data = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "sessions": {},
            "errors": [],
            "warnings": []
        }

        # Write initial log
        self._save_log()
        self._write_txt(f"{'='*70}\n")
        self._write_txt(f"EXPERIMENT 3 DEBUG LOG\n")
        self._write_txt(f"Run ID: {run_id}\n")
        self._write_txt(f"Started: {self.log_data['start_time']}\n")
        self._write_txt(f"{'='*70}\n\n")

    def _save_log(self):
        """Save JSON log to disk (called frequently for crash safety)."""
        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2, default=str)

    def _write_txt(self, text: str):
        """Append to human-readable text log."""
        with open(self.txt_path, 'a') as f:
            f.write(text)

    def start_session(self, model_name: str, session_name: str):
        """Initialize logging for a training session."""
        session_key = f"{model_name}__{session_name}"

        self.current_session = session_key
        self.log_data["sessions"][session_key] = {
            "model": model_name,
            "session": session_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",

            # Data paths (to verify correct data is used)
            "data_yaml_path": None,
            "train_images_dir": None,
            "sample_train_images": [],

            # Masking info
            "masking_enabled": False,
            "masking_config": {},
            "hooked_layers": [],
            "masking_activations_per_epoch": [],

            # Training progress
            "epochs_completed": 0,
            "losses_per_epoch": [],
            "model_training_mode_checks": [],

            # Validation results during training
            "val_metrics_per_epoch": [],

            # Final evaluation
            "final_weights_path": None,
            "evaluation_results": {}
        }

        self._write_txt(f"\n{'='*70}\n")
        self._write_txt(f"SESSION: {session_key}\n")
        self._write_txt(f"Started: {datetime.now().isoformat()}\n")
        self._write_txt(f"{'='*70}\n")

        self._save_log()
        return session_key

    def log_data_config(
        self,
        data_yaml_path: str,
        train_images_dir: str,
        sample_images: List[str],
        num_train_images: int,
        num_val_images: int
    ):
        """Log data configuration for current session."""
        session = self.log_data["sessions"][self.current_session]
        session["data_yaml_path"] = data_yaml_path
        session["train_images_dir"] = train_images_dir
        session["sample_train_images"] = sample_images[:10]  # First 10
        session["num_train_images"] = num_train_images
        session["num_val_images"] = num_val_images

        self._write_txt(f"\nDATA CONFIG:\n")
        self._write_txt(f"  data.yaml: {data_yaml_path}\n")
        self._write_txt(f"  train_dir: {train_images_dir}\n")
        self._write_txt(f"  num_train: {num_train_images}\n")
        self._write_txt(f"  num_val: {num_val_images}\n")
        self._write_txt(f"  sample_images:\n")
        for img in sample_images[:5]:
            self._write_txt(f"    - {img}\n")

        self._save_log()

    def log_masking_config(
        self,
        enabled: bool,
        mask_location: Optional[str],
        layer_prefixes: List[str],
        p_apply: float,
        p_channels: float,
        num_hooks_added: int,
        hooked_layer_names: List[str]
    ):
        """Log masking configuration."""
        session = self.log_data["sessions"][self.current_session]
        session["masking_enabled"] = enabled
        session["masking_config"] = {
            "mask_location": mask_location,
            "layer_prefixes": layer_prefixes,
            "p_apply": p_apply,
            "p_channels": p_channels,
            "num_hooks_added": num_hooks_added
        }
        session["hooked_layers"] = hooked_layer_names

        self._write_txt(f"\nMASKING CONFIG:\n")
        self._write_txt(f"  enabled: {enabled}\n")
        self._write_txt(f"  location: {mask_location}\n")
        self._write_txt(f"  layer_prefixes: {layer_prefixes}\n")
        self._write_txt(f"  p_apply: {p_apply}\n")
        self._write_txt(f"  p_channels: {p_channels}\n")
        self._write_txt(f"  num_hooks: {num_hooks_added}\n")

        if hooked_layer_names:
            self._write_txt(f"  hooked_layers ({len(hooked_layer_names)}):\n")
            for layer in hooked_layer_names[:20]:  # First 20
                self._write_txt(f"    - {layer}\n")
            if len(hooked_layer_names) > 20:
                self._write_txt(f"    ... and {len(hooked_layer_names) - 20} more\n")

        self._save_log()

    def log_epoch_start(self, epoch: int, model_training_mode: bool):
        """Log start of an epoch with model state."""
        session = self.log_data["sessions"][self.current_session]
        session["model_training_mode_checks"].append({
            "epoch": epoch,
            "time": datetime.now().isoformat(),
            "model_training": model_training_mode
        })

        self._write_txt(f"\n  Epoch {epoch}: model.training={model_training_mode}\n")
        self._save_log()

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        mask_activations: int,
        additional_metrics: Optional[Dict] = None
    ):
        """Log end of epoch with losses and masking stats."""
        session = self.log_data["sessions"][self.current_session]
        session["epochs_completed"] = epoch
        session["losses_per_epoch"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        session["masking_activations_per_epoch"].append({
            "epoch": epoch,
            "mask_activations": mask_activations
        })

        if additional_metrics:
            session["val_metrics_per_epoch"].append({
                "epoch": epoch,
                **additional_metrics
            })

        self._write_txt(f"    train_loss={train_loss}, val_loss={val_loss}, ")
        self._write_txt(f"mask_activations={mask_activations}\n")

        self._save_log()

    def log_masking_summary(self, total_activations: int, hooked_layers_summary: List[Dict]):
        """Log final masking summary after training."""
        session = self.log_data["sessions"][self.current_session]
        session["masking_final_summary"] = {
            "total_activations": total_activations,
            "per_layer_activations": hooked_layers_summary
        }

        self._write_txt(f"\nMASKING SUMMARY:\n")
        self._write_txt(f"  total_activations: {total_activations}\n")

        if total_activations == 0 and session["masking_enabled"]:
            self._write_txt(f"  *** WARNING: Masking was enabled but NEVER activated! ***\n")
            self.log_warning("Masking enabled but never activated - hooks may not be working!")

        self._save_log()

    def log_training_complete(self, weights_path: str, training_time_seconds: float):
        """Log training completion."""
        session = self.log_data["sessions"][self.current_session]
        session["status"] = "trained"
        session["final_weights_path"] = weights_path
        session["training_time_seconds"] = training_time_seconds

        self._write_txt(f"\nTRAINING COMPLETE:\n")
        self._write_txt(f"  weights: {weights_path}\n")
        self._write_txt(f"  time: {training_time_seconds:.1f}s\n")

        self._save_log()

    def log_evaluation_result(
        self,
        test_type: str,  # 'clean' or 'occluded'
        metrics: Dict
    ):
        """Log evaluation results."""
        session = self.log_data["sessions"][self.current_session]
        session["evaluation_results"][test_type] = metrics

        self._write_txt(f"\nEVALUATION ({test_type}):\n")
        self._write_txt(f"  F1: {metrics.get('f1', 'N/A'):.4f}\n")
        self._write_txt(f"  Precision: {metrics.get('precision', 'N/A'):.4f}\n")
        self._write_txt(f"  Recall: {metrics.get('recall', 'N/A'):.4f}\n")
        self._write_txt(f"  TP/FP/FN: {metrics.get('tp', 0)}/{metrics.get('fp', 0)}/{metrics.get('fn', 0)}\n")

        self._save_log()

    def log_sample_predictions(
        self,
        test_type: str,
        sample_predictions: List[Dict]
    ):
        """Log sample predictions for debugging."""
        session = self.log_data["sessions"][self.current_session]
        if "sample_predictions" not in session:
            session["sample_predictions"] = {}
        session["sample_predictions"][test_type] = sample_predictions

        self._write_txt(f"\nSAMPLE PREDICTIONS ({test_type}):\n")
        for i, pred in enumerate(sample_predictions[:5]):
            self._write_txt(f"  Image {i+1}: {pred.get('image_id', 'unknown')}\n")
            self._write_txt(f"    num_detections: {len(pred.get('detections', []))}\n")
            self._write_txt(f"    num_ground_truth: {pred.get('num_gt', 0)}\n")
            if pred.get('detections'):
                det = pred['detections'][0]
                self._write_txt(f"    first_det: {det.get('class_name', '?')} @ {det.get('confidence', 0):.3f}\n")

        self._save_log()

    def log_ground_truth_stats(
        self,
        test_type: str,
        num_images: int,
        num_objects: int,
        objects_per_class: Dict[str, int]
    ):
        """Log ground truth statistics."""
        session = self.log_data["sessions"][self.current_session]
        if "ground_truth_stats" not in session:
            session["ground_truth_stats"] = {}
        session["ground_truth_stats"][test_type] = {
            "num_images": num_images,
            "num_objects": num_objects,
            "top_classes": dict(sorted(objects_per_class.items(), key=lambda x: -x[1])[:10])
        }

        self._write_txt(f"\nGROUND TRUTH ({test_type}):\n")
        self._write_txt(f"  images: {num_images}, objects: {num_objects}\n")

        self._save_log()

    def log_warning(self, message: str):
        """Log a warning."""
        warning = {
            "time": datetime.now().isoformat(),
            "session": getattr(self, 'current_session', None),
            "message": message
        }
        self.log_data["warnings"].append(warning)

        self._write_txt(f"\n*** WARNING: {message} ***\n")
        self._save_log()

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log an error."""
        error = {
            "time": datetime.now().isoformat(),
            "session": getattr(self, 'current_session', None),
            "message": message,
            "exception": str(exception) if exception else None
        }
        self.log_data["errors"].append(error)

        session_key = getattr(self, 'current_session', None)
        if session_key and session_key in self.log_data["sessions"]:
            self.log_data["sessions"][session_key]["status"] = "error"

        self._write_txt(f"\n*** ERROR: {message} ***\n")
        if exception:
            self._write_txt(f"    {exception}\n")

        self._save_log()

    def log_model_architecture_check(
        self,
        model_name: str,
        model: nn.Module,
        max_layers: int = 50
    ):
        """Log model architecture for verification of layer names."""
        layers = []
        for name, module in model.named_modules():
            layers.append({
                "name": name,
                "type": type(module).__name__
            })

        session = self.log_data["sessions"][self.current_session]
        session["model_architecture"] = {
            "model_name": model_name,
            "total_layers": len(layers),
            "layer_sample": layers[:max_layers]
        }

        self._write_txt(f"\nMODEL ARCHITECTURE ({model_name}):\n")
        self._write_txt(f"  total_layers: {len(layers)}\n")
        self._write_txt(f"  first {min(30, len(layers))} layers:\n")
        for layer in layers[:30]:
            self._write_txt(f"    {layer['name']}: {layer['type']}\n")

        self._save_log()

    def log_environment(self, env_info: Dict):
        """Log environment/GPU info."""
        self.log_data["environment"] = env_info

        self._write_txt(f"\nENVIRONMENT:\n")
        self._write_txt(f"  GPU: {env_info.get('gpu_name', 'N/A')}\n")
        self._write_txt(f"  CUDA: {env_info.get('cuda_version', 'N/A')}\n")
        self._write_txt(f"  PyTorch: {env_info.get('pytorch_version', 'N/A')}\n")
        self._write_txt(f"  Ultralytics: {env_info.get('ultralytics_version', 'N/A')}\n")

        self._save_log()

    def log_labels_check(self, check_name: str, labels_info: Dict):
        """Log labels verification results."""
        session = self.log_data["sessions"][self.current_session]
        if "labels_checks" not in session:
            session["labels_checks"] = {}
        session["labels_checks"][check_name] = labels_info

        self._write_txt(f"\nLABELS CHECK ({check_name}):\n")
        self._write_txt(f"  images_dir: {labels_info.get('images_dir', 'N/A')}\n")
        self._write_txt(f"  labels_dir_exists: {labels_info.get('labels_dir_exists', 'N/A')}\n")
        self._write_txt(f"  num_images: {labels_info.get('num_images', 0)}\n")
        self._write_txt(f"  num_missing_labels: {labels_info.get('num_missing_labels', 0)}\n")
        self._write_txt(f"  total_boxes: {labels_info.get('total_boxes', 0)}\n")
        self._write_txt(f"  avg_boxes_per_image: {labels_info.get('avg_boxes_per_image', 0):.2f}\n")

        if labels_info.get('num_missing_labels', 0) > 0:
            self._write_txt(f"  *** WARNING: {labels_info['num_missing_labels']} images missing labels! ***\n")
            self.log_warning(f"{check_name}: {labels_info['num_missing_labels']} images missing labels")

        self._save_log()

    def log_confidence_analysis(self, test_type: str, conf_info: Dict):
        """Log confidence distribution analysis."""
        session = self.log_data["sessions"][self.current_session]
        if "confidence_analysis" not in session:
            session["confidence_analysis"] = {}
        session["confidence_analysis"][test_type] = conf_info

        self._write_txt(f"\nCONFIDENCE ANALYSIS ({test_type}):\n")
        self._write_txt(f"  total_predictions: {conf_info.get('num_predictions', 0)}\n")
        self._write_txt(f"  images_with_detections: {conf_info.get('images_with_detections', 0)}\n")
        self._write_txt(f"  images_without_detections: {conf_info.get('images_without_detections', 0)}\n")

        if conf_info.get('mean_conf') is not None:
            self._write_txt(f"  mean_confidence: {conf_info['mean_conf']:.4f}\n")
            self._write_txt(f"  median_confidence: {conf_info.get('median_conf', 0):.4f}\n")
            self._write_txt(f"  conf >= 0.5: {conf_info.get('conf_above_0.5', 0)}\n")
            self._write_txt(f"  conf >= 0.3: {conf_info.get('conf_above_0.3', 0)}\n")
            self._write_txt(f"  conf >= 0.1: {conf_info.get('conf_above_0.1', 0)}\n")

            # Warning if confidence is very low
            if conf_info['mean_conf'] < 0.1:
                self._write_txt(f"  *** WARNING: Very low mean confidence! Model may not have learned. ***\n")
                self.log_warning(f"{test_type}: Very low mean confidence ({conf_info['mean_conf']:.4f})")
        else:
            self._write_txt(f"  *** WARNING: No predictions made! ***\n")
            self.log_warning(f"{test_type}: No predictions made at all")

        self._save_log()

    def log_ultralytics_results(self, results_summary: Dict):
        """Log Ultralytics training results summary."""
        session = self.log_data["sessions"][self.current_session]
        session["ultralytics_results"] = results_summary

        self._write_txt(f"\nULTRALYTICS TRAINING RESULTS:\n")
        for key, value in results_summary.items():
            if isinstance(value, float):
                self._write_txt(f"  {key}: {value:.4f}\n")
            else:
                self._write_txt(f"  {key}: {value}\n")

        self._save_log()

    def end_session(self, success: bool = True):
        """Mark session as complete."""
        session = self.log_data["sessions"][self.current_session]
        session["end_time"] = datetime.now().isoformat()
        session["status"] = "success" if success else "failed"

        self._write_txt(f"\nSESSION ENDED: {'SUCCESS' if success else 'FAILED'}\n")
        self._write_txt(f"{'='*70}\n")

        self._save_log()

    def finalize(self):
        """Finalize the log with summary statistics."""
        self.log_data["end_time"] = datetime.now().isoformat()

        # Generate summary
        summary = {
            "total_sessions": len(self.log_data["sessions"]),
            "successful_sessions": sum(1 for s in self.log_data["sessions"].values() if s["status"] == "success"),
            "failed_sessions": sum(1 for s in self.log_data["sessions"].values() if s["status"] == "failed"),
            "total_warnings": len(self.log_data["warnings"]),
            "total_errors": len(self.log_data["errors"])
        }
        self.log_data["summary"] = summary

        self._write_txt(f"\n{'='*70}\n")
        self._write_txt(f"EXPERIMENT COMPLETE\n")
        self._write_txt(f"{'='*70}\n")
        self._write_txt(f"Sessions: {summary['total_sessions']} ({summary['successful_sessions']} success, {summary['failed_sessions']} failed)\n")
        self._write_txt(f"Warnings: {summary['total_warnings']}\n")
        self._write_txt(f"Errors: {summary['total_errors']}\n")

        # Key findings
        self._write_txt(f"\nKEY FINDINGS TO CHECK:\n")

        # Check masking activation
        for session_key, session in self.log_data["sessions"].items():
            if session.get("masking_enabled"):
                total_activations = session.get("masking_final_summary", {}).get("total_activations", 0)
                if total_activations == 0:
                    self._write_txt(f"  [PROBLEM] {session_key}: Masking enabled but 0 activations!\n")
                else:
                    self._write_txt(f"  [OK] {session_key}: {total_activations} mask activations\n")

        self._save_log()

        print(f"\nDebug logs saved to:")
        print(f"  JSON: {self.log_path}")
        print(f"  Text: {self.txt_path}")


def verify_data_yaml(yaml_path: str) -> Dict:
    """Load and verify a data.yaml file, returning key info."""
    import yaml
    from pathlib import Path

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Resolve paths
    base_path = Path(config.get('path', ''))
    train_path = base_path / config.get('train', '')
    val_path = base_path / config.get('val', '')

    # Count images
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))

    return {
        "yaml_path": yaml_path,
        "base_path": str(base_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_exists": train_path.exists(),
        "val_exists": val_path.exists(),
        "num_train_images": len(train_images),
        "num_val_images": len(val_images),
        "sample_train_images": [str(p) for p in train_images[:5]],
        "num_classes": config.get('nc', len(config.get('names', []))),
        "class_names": config.get('names', [])[:10]
    }


def verify_labels_exist(images_dir: str, labels_dir: str = None) -> Dict:
    """
    Verify that label files exist for images.

    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory (if None, assumes ../labels relative to images)

    Returns:
        Dict with verification results
    """
    from pathlib import Path

    images_path = Path(images_dir)
    if labels_dir is None:
        labels_path = images_path.parent / "labels"
    else:
        labels_path = Path(labels_dir)

    # Get all images
    images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))

    # Check for corresponding labels
    missing_labels = []
    empty_labels = []
    labels_with_boxes = []
    total_boxes = 0

    for img in images:
        label_file = labels_path / (img.stem + '.txt')
        if not label_file.exists():
            missing_labels.append(img.name)
        else:
            content = label_file.read_text().strip()
            if not content:
                empty_labels.append(img.name)
            else:
                num_boxes = len(content.split('\n'))
                labels_with_boxes.append((img.name, num_boxes))
                total_boxes += num_boxes

    return {
        "images_dir": str(images_path),
        "labels_dir": str(labels_path),
        "labels_dir_exists": labels_path.exists(),
        "num_images": len(images),
        "num_missing_labels": len(missing_labels),
        "num_empty_labels": len(empty_labels),
        "num_labels_with_boxes": len(labels_with_boxes),
        "total_boxes": total_boxes,
        "avg_boxes_per_image": total_boxes / max(1, len(labels_with_boxes)),
        "missing_labels_sample": missing_labels[:5],
        "empty_labels_sample": empty_labels[:5]
    }


def verify_occluded_test_data(occluded_test_dir: str) -> Dict:
    """
    Verify that occluded test data exists and has correct structure.

    Args:
        occluded_test_dir: Path to occluded test directory (e.g., data/synthetic_occlusion/level_040)

    Returns:
        Dict with verification results
    """
    from pathlib import Path

    test_path = Path(occluded_test_dir)
    images_path = test_path / "images"
    labels_path = test_path / "labels"

    result = {
        "path": str(test_path),
        "exists": test_path.exists(),
        "images_dir_exists": images_path.exists(),
        "labels_dir_exists": labels_path.exists(),
    }

    if images_path.exists():
        images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        result["num_images"] = len(images)
        result["sample_images"] = [img.name for img in images[:5]]
    else:
        result["num_images"] = 0
        result["sample_images"] = []

    if labels_path.exists():
        labels = list(labels_path.glob('*.txt'))
        result["num_labels"] = len(labels)

        # Check a few labels
        total_boxes = 0
        for label in labels[:100]:
            content = label.read_text().strip()
            if content:
                total_boxes += len(content.split('\n'))
        result["sample_total_boxes"] = total_boxes
    else:
        result["num_labels"] = 0
        result["sample_total_boxes"] = 0

    return result


def get_environment_info() -> Dict:
    """Get GPU and environment info for debugging."""
    import sys
    import platform

    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    try:
        import torch
        env_info["pytorch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        env_info["pytorch_version"] = "not installed"

    try:
        import ultralytics
        env_info["ultralytics_version"] = ultralytics.__version__
    except (ImportError, AttributeError):
        env_info["ultralytics_version"] = "unknown"

    return env_info


def analyze_prediction_confidences(predictions: List[Dict]) -> Dict:
    """
    Analyze confidence distribution of predictions.

    Args:
        predictions: List of prediction dicts with 'detections' key

    Returns:
        Dict with confidence statistics
    """
    all_confidences = []
    images_with_detections = 0
    images_without_detections = 0

    for pred in predictions:
        detections = pred.get('detections', [])
        if detections:
            images_with_detections += 1
            for det in detections:
                conf = det.get('confidence', 0)
                all_confidences.append(conf)
        else:
            images_without_detections += 1

    if not all_confidences:
        return {
            "num_predictions": 0,
            "images_with_detections": images_with_detections,
            "images_without_detections": images_without_detections,
            "min_conf": None,
            "max_conf": None,
            "mean_conf": None,
            "conf_above_0.5": 0,
            "conf_above_0.3": 0,
            "conf_above_0.1": 0
        }

    import statistics

    return {
        "num_predictions": len(all_confidences),
        "images_with_detections": images_with_detections,
        "images_without_detections": images_without_detections,
        "min_conf": min(all_confidences),
        "max_conf": max(all_confidences),
        "mean_conf": statistics.mean(all_confidences),
        "median_conf": statistics.median(all_confidences),
        "conf_above_0.5": sum(1 for c in all_confidences if c >= 0.5),
        "conf_above_0.3": sum(1 for c in all_confidences if c >= 0.3),
        "conf_above_0.1": sum(1 for c in all_confidences if c >= 0.1),
        "conf_percentiles": {
            "p10": sorted(all_confidences)[len(all_confidences)//10] if len(all_confidences) > 10 else None,
            "p50": sorted(all_confidences)[len(all_confidences)//2] if len(all_confidences) > 2 else None,
            "p90": sorted(all_confidences)[int(len(all_confidences)*0.9)] if len(all_confidences) > 10 else None,
        }
    }


if __name__ == "__main__":
    # Test the logger
    print("Testing ExperimentDebugLogger...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentDebugLogger(Path(tmpdir), "TEST_RUN")

        logger.start_session("yolov8m", "S1_clean_train")
        logger.log_data_config(
            data_yaml_path="data/processed/data_clean.yaml",
            train_images_dir="data/raw/train/images",
            sample_images=["img1.jpg", "img2.jpg"],
            num_train_images=1384,
            num_val_images=200
        )
        logger.log_masking_config(
            enabled=False,
            mask_location=None,
            layer_prefixes=[],
            p_apply=0.5,
            p_channels=0.2,
            num_hooks_added=0,
            hooked_layer_names=[]
        )
        logger.log_epoch_start(1, True)
        logger.log_epoch_end(1, train_loss=0.5, val_loss=0.6, mask_activations=0)
        logger.log_training_complete("/path/to/weights.pt", 100.0)
        logger.end_session(success=True)
        logger.finalize()

        # Show output
        with open(logger.txt_path) as f:
            print(f.read())
