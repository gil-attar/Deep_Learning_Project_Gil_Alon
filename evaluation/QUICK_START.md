# Evaluation System - Quick Start

## What's Included

This evaluation system provides comprehensive metrics for object detection models (YOLOv8 and RT-DETR).

**3 Metrics:**
1. **Detection P/R/F1** - Box-level correctness at multiple confidence thresholds
2. **Per-Class Metrics** - Class-specific performance + confusion matrix
3. **Counting Quality** - Accuracy for duplicate objects (MAE)

**4 Plots:**
1. `threshold_sweep.png` - P/R/F1 vs confidence threshold
2. `per_class_f1.png` - Bar chart of F1 per class
3. `confusion_matrix.png` - Heatmap of class confusions
4. `count_mae_comparison.png` - Counting accuracy (matched-only vs all-predictions)

---

## Complete Workflow

```
1. BUILD INDICES (once)     → Ground truth JSONs
2. TRAIN MODEL              → Weights file
3. GENERATE PREDICTIONS     → Predictions JSON
4. EVALUATE                 → Metrics + Plots
```

---

## Step 1: Build Ground Truth Indices (ONE TIME)

```bash
python scripts/build_evaluation_indices.py \
    --dataset_root data/raw \
    --output_dir data/processed/evaluation
```

**Creates:**
```
data/processed/evaluation/
├── train_index.json   (1384 images)
├── val_index.json     (200 images)
└── test_index.json    (400 images)
```

---

## Step 2: Train Your Model

Train however you want. Save the best weights.

---

## Step 3: Generate Predictions

```python
from ultralytics import YOLO
from pathlib import Path
import json

# Load model
model = YOLO('path/to/weights.pt')

# Load test index
with open('data/processed/evaluation/test_index.json') as f:
    test_index = json.load(f)

# Generate predictions
predictions = []
for img_data in test_index['images']:
    image_path = Path('data/raw/test/images') / img_data['image_filename']

    results = model.predict(source=str(image_path), conf=0.01, imgsz=640, verbose=False)[0]

    detections = []
    for i in range(len(results.boxes)):
        detections.append({
            "class_id": int(results.boxes.cls[i].item()),
            "class_name": results.names[int(results.boxes.cls[i].item())],
            "confidence": float(results.boxes.conf[i].item()),
            "bbox": results.boxes.xyxy[i].tolist(),
            "bbox_format": "xyxy"
        })

    predictions.append({"image_id": img_data['image_id'], "detections": detections})

# Save
pred_json = {
    "run_id": "my_experiment",
    "split": "test",
    "model_family": "yolo",
    "predictions": predictions
}

with open('evaluation/metrics/my_experiment_test_predictions.json', 'w') as f:
    json.dump(pred_json, f, indent=2)
```

**Important:** Use `conf=0.01` to save all predictions. Filter during evaluation.

---

## Step 4: Run Evaluation

### Option A: CLI

```bash
python scripts/evaluate_run.py \
    --predictions evaluation/metrics/my_experiment_test_predictions.json \
    --ground_truth data/processed/evaluation/test_index.json \
    --output_dir evaluation/results/my_experiment/test/ \
    --run_name "My Experiment" \
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

# Load
preds = load_predictions("evaluation/metrics/my_experiment_test_predictions.json")
gts = load_ground_truth("data/processed/evaluation/test_index.json")
class_names = load_class_names("data/processed/evaluation/test_index.json")

# Evaluate
threshold_sweep = eval_detection_prf_at_iou(preds, gts, iou_threshold=0.5)
per_class = eval_per_class_metrics_and_confusions(preds, gts, conf_threshold=0.5, class_names=class_names)
counting = eval_counting_quality(preds, gts, conf_threshold=0.5, class_names=class_names)

# Plot
plot_all_metrics(
    threshold_sweep=threshold_sweep,
    per_class_results=per_class['per_class'],
    confusion_data=per_class,
    counting_results=counting,
    output_dir="evaluation/results/my_experiment/test/",
    run_name="My Experiment"
)
```

---

## Output Files

```
evaluation/results/my_experiment/test/
├── metrics.json              # All metrics
├── summary.csv               # Quick table
├── threshold_sweep.png       # P/R/F1 vs confidence
├── per_class_f1.png          # Per-class performance
├── confusion_matrix.png      # Class confusions
└── count_mae_comparison.png  # Counting accuracy
```

---

## File Locations Summary

| What | Where |
|------|-------|
| Ground truth indices | `data/processed/evaluation/*_index.json` |
| Prediction JSONs | `evaluation/metrics/{experiment}_predictions.json` |
| Results & plots | `evaluation/results/{experiment}/` |

---

## Important Rules

1. **Build indices once** - Don't regenerate after experiments start
2. **Save predictions with conf=0.01** - Filter during evaluation
3. **Use validation for threshold selection** - Never tune on test set
4. **Same test set for all experiments** - Fair comparison

---

## Threshold Selection Protocol

```python
# 1. Evaluate validation set
val_results = eval_detection_prf_at_iou(val_preds, val_gts,
    conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# 2. Find best threshold
best_thr = max(val_results.keys(), key=lambda k: val_results[k]['f1'])

# 3. Report test set at that threshold
test_results = eval_detection_prf_at_iou(test_preds, test_gts,
    conf_thresholds=[best_thr])
```

---

## Test the System

Run the test notebook:
```bash
# In Google Colab or locally
notebooks/test_evaluation_system.ipynb
```

This downloads data, trains a small model, and runs the full evaluation.

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: predictions` | Run inference first, save predictions JSON |
| `Predictions count != GT count` | Check image_ids match between files |
| `Import error: evaluation` | Run from project root directory |
| `All zeros in results` | Check bbox format (must be xyxy pixels, not normalized) |

---

## Full Documentation

See [README_METRICS.md](README_METRICS.md) for:
- Detailed metric definitions
- Matching algorithm explanation
- JSON format specifications
- FAQ

---

Good luck with your experiments!
