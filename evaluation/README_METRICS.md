# Evaluation System Guide

This document explains the evaluation system for the CNN vs Transformer object detection project.

---

## Overview: JSON Files & Workflow

### Complete Workflow

```
1. BUILD INDICES (run ONCE before experiments)
   python scripts/build_evaluation_indices.py

2. TRAIN MODEL (your experiment)
   → saves weights to models/

3. GENERATE PREDICTIONS (run after training)
   Run inference, save detections to JSON

4. EVALUATE (run to get metrics & plots)
   python scripts/evaluate_run.py
```

### JSON Files Created

| File | Location | When Created | Purpose |
|------|----------|--------------|---------|
| `train_index.json` | `data/processed/evaluation/` | Once (Step 1) | Ground truth for train set |
| `val_index.json` | `data/processed/evaluation/` | Once (Step 1) | Ground truth for val set |
| `test_index.json` | `data/processed/evaluation/` | Once (Step 1) | Ground truth for test set |
| `{experiment}_predictions.json` | `evaluation/metrics/` | Per experiment (Step 3) | Model predictions |
| `metrics.json` | `evaluation/results/{experiment}/` | Per evaluation (Step 4) | Computed metrics |
| `summary.csv` | `evaluation/results/{experiment}/` | Per evaluation (Step 4) | Quick table view |

### What About Legacy Files?

**We do NOT use legacy files.** The following are from old notebooks (02, 04) and use Ultralytics' built-in evaluation:

```
evaluation/metrics/baseline_yolo_*.json    ← Legacy, don't use
evaluation/metrics/baseline_rtdetr_*.json  ← Legacy, don't use
evaluation/metrics/e3_*.json               ← Legacy, don't use
```

Our new system computes metrics ourselves for more control and transparency.

---

## Step 1: Build Ground Truth Indices

Run **once** before starting experiments:

```bash
python scripts/build_evaluation_indices.py \
    --dataset_root data/raw \
    --output_dir data/processed/evaluation
```

**Creates:**
- `data/processed/evaluation/train_index.json` (1384 images)
- `data/processed/evaluation/val_index.json` (200 images)
- `data/processed/evaluation/test_index.json` (400 images)

**Index Format:**
```json
{
  "metadata": {
    "split": "test",
    "num_images": 400,
    "total_objects": 856,
    "num_classes": 26,
    "class_names": {"0": "Asparagus", "1": "Avocado", ...}
  },
  "images": [
    {
      "image_id": "image_001",
      "image_filename": "image_001.jpg",
      "image_width": 640,
      "image_height": 480,
      "ground_truth": [
        {
          "class_id": 9,
          "class_name": "Capsicum",
          "bbox_xyxy": [100, 150, 300, 400],
          "bbox_yolo": [0.31, 0.57, 0.31, 0.52]
        }
      ],
      "difficulty": "easy"
    }
  ]
}
```

**Important:** Ground truth bboxes are stored in **actual pixel coordinates** (not normalized). The script reads each image to get its real dimensions.

---

## Step 2: Train Your Model

Train however you want (notebook, script, etc.). Save the best weights.

---

## Step 3: Generate Predictions

After training, run inference and save predictions to JSON.

**Example code:**

```python
from ultralytics import YOLO  # or RTDETR
from pathlib import Path
import json

# Load trained model
model = YOLO('path/to/your/trained_weights.pt')

# Load test index
with open('data/processed/evaluation/test_index.json') as f:
    test_index = json.load(f)

# Generate predictions
predictions = []
for img_data in test_index['images']:
    image_path = Path('data/raw/test/images') / img_data['image_filename']

    # IMPORTANT: Use very low conf threshold (0.01)
    # We filter by threshold during evaluation, not here
    results = model.predict(
        source=str(image_path),
        conf=0.01,  # Save almost everything
        imgsz=640,
        verbose=False
    )[0]

    # Extract detections
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

# Save predictions JSON
pred_json = {
    "run_id": "YOUR_EXPERIMENT_NAME",
    "split": "test",
    "model_family": "yolo",  # or "rtdetr"
    "model_name": "yolov8n",
    "inference_settings": {
        "conf_threshold": 0.01,
        "iou_threshold": 0.50,
        "imgsz": 640
    },
    "predictions": predictions
}

output_path = "evaluation/metrics/YOUR_EXPERIMENT_test_predictions.json"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(pred_json, f, indent=2)

print(f"Saved predictions to {output_path}")
```

**Predictions Format:**
```json
{
  "run_id": "e1_freeze_backbone_50epochs",
  "split": "test",
  "model_family": "yolo",
  "model_name": "yolov8n_frozen_backbone",
  "inference_settings": {
    "conf_threshold": 0.01,
    "iou_threshold": 0.50,
    "imgsz": 640
  },
  "predictions": [
    {
      "image_id": "image_001",
      "detections": [
        {
          "class_id": 9,
          "class_name": "Capsicum",
          "confidence": 0.87,
          "bbox": [105, 148, 298, 395],
          "bbox_format": "xyxy"
        }
      ]
    }
  ]
}
```

---

## Step 4: Run Evaluation

### Option A: CLI Script

```bash
python scripts/evaluate_run.py \
    --predictions evaluation/metrics/YOUR_EXPERIMENT_test_predictions.json \
    --ground_truth data/processed/evaluation/test_index.json \
    --output_dir evaluation/results/YOUR_EXPERIMENT/test/ \
    --run_name "Your Experiment Name" \
    --conf_thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
```

### Option B: Notebook

```python
from evaluation.io import load_predictions, load_ground_truth, load_class_names
from evaluation.metrics import (
    eval_detection_prf_at_iou,
    eval_per_class_metrics_and_confusions,
    eval_counting_quality
)
from evaluation.plots import plot_all_metrics

# Load data
preds = load_predictions("evaluation/metrics/YOUR_EXPERIMENT_test_predictions.json")
gts = load_ground_truth("data/processed/evaluation/test_index.json")
class_names = load_class_names("data/processed/evaluation/test_index.json")

# Run all 3 metrics
threshold_sweep = eval_detection_prf_at_iou(preds, gts, iou_threshold=0.5)
per_class = eval_per_class_metrics_and_confusions(preds, gts, conf_threshold=0.5, class_names=class_names)
counting = eval_counting_quality(preds, gts, conf_threshold=0.5, class_names=class_names)

# Generate plots
plot_all_metrics(
    threshold_sweep=threshold_sweep,
    per_class_results=per_class['per_class'],
    confusion_data=per_class,
    counting_results=counting,
    output_dir="evaluation/results/YOUR_EXPERIMENT/test/",
    run_name="Your Experiment"
)
```

### Output Files

After evaluation, you'll find in `evaluation/results/YOUR_EXPERIMENT/`:

| File | Description |
|------|-------------|
| `metrics.json` | All computed metrics in JSON format |
| `summary.csv` | Quick table for copy-paste into reports |
| `threshold_sweep.png` | P/R/F1 vs confidence threshold plot |
| `per_class_f1.png` | Bar chart of F1 per class |
| `confusion_matrix.png` | Heatmap of class confusions |
| `count_mae_comparison.png` | Counting accuracy comparison |

---

## The 3 Evaluation Metrics

### Metric 1: Detection P/R/F1 at IoU Threshold

**Purpose:** Evaluate box-level correctness across confidence thresholds.

**Definitions:**
- **True Positive (TP):** Prediction matched to GT with IoU ≥ 0.5
- **False Positive (FP):** Prediction with no matching GT
- **False Negative (FN):** GT with no matching prediction

**Formulas:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × P × R / (P + R)

**Output:** Results at multiple confidence thresholds to find the optimal operating point.

### Metric 2: Per-Class Metrics + Confusion Matrix

**Purpose:** Identify which classes are hard and what confusions occur.

**Per-Class:** Precision, Recall, F1, Support for each of the 26 classes.

**Confusion Matrix:**
- Rows = True class, Columns = Predicted class
- Only includes matched detections (IoU ≥ threshold)
- Diagonal = correct class predictions
- Off-diagonal = class confusions

### Metric 3: Counting Quality (MAE)

**Purpose:** Measure counting accuracy for images with multiple objects.

**Two Methods:**
- **Matched-Only:** Count = number of TPs (robust to false positives)
- **All-Predictions:** Count = all boxes above threshold (simpler)

**MAE:** Mean Absolute Error between predicted count and GT count per image.

---

## Threshold Selection Protocol

**IMPORTANT:** Use validation set for threshold selection, test set for final reporting only.

### Correct Workflow:

```python
# 1. Evaluate validation set at multiple thresholds
val_results = eval_detection_prf_at_iou(val_preds, val_gts,
    conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# 2. Find best threshold on validation
best_threshold = max(val_results.keys(), key=lambda k: val_results[k]['f1'])

# 3. Report test set metrics at that threshold
test_results = eval_detection_prf_at_iou(test_preds, test_gts,
    conf_thresholds=[best_threshold])
```

**Never** pick threshold based on test set performance!

---

## Matching Algorithm

We use **greedy one-to-one matching** (per image, per class):

1. Compute IoU between all prediction-GT pairs
2. Sort pairs by IoU descending
3. Greedily assign matches (highest IoU first)
4. Each prediction matches at most one GT
5. Each GT matches at most one prediction
6. Match valid only if IoU ≥ threshold (default: 0.5)

---

## Directory Structure

```
evaluation/
├── __init__.py
├── io.py                 # Load/save predictions and ground truth
├── matching.py           # IoU computation and greedy matching
├── metrics.py            # 3 core evaluation functions
├── plots.py              # Visualization functions
├── README_METRICS.md     # This file
├── QUICK_START.md        # Quick reference guide
├── metrics/              # Prediction JSONs go here
│   └── {experiment}_predictions.json
└── results/              # Evaluation outputs go here
    └── {experiment}/
        ├── metrics.json
        ├── summary.csv
        └── *.png (plots)

data/processed/evaluation/
├── train_index.json      # Ground truth indices
├── val_index.json
└── test_index.json

scripts/
├── build_evaluation_indices.py   # Create GT indices (run once)
└── evaluate_run.py               # CLI evaluation script
```

---

## FAQ

**Q: Why not use Ultralytics' built-in `model.val()`?**
A: We want more control over metrics. Our system lets us compute P/R/F1 at multiple thresholds, per-class breakdowns, confusion matrices, and counting metrics - all with consistent matching logic.

**Q: Why save predictions with conf=0.01?**
A: We filter by confidence during evaluation, not inference. This lets us evaluate the same predictions at different thresholds without re-running inference.

**Q: Why greedy matching instead of Hungarian algorithm?**
A: Greedy is simpler, faster, and standard in detection evaluation (COCO, PASCAL VOC).

**Q: How do I compare YOLO vs RT-DETR fairly?**
A: Run both on the **same test set** with the **same evaluation code**. Use validation to pick best threshold for each model separately.

---

## Contact

For questions about the evaluation system, open an issue in the repo.
