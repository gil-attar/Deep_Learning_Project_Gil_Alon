```markdown
# Evaluation Metrics Guide

This document explains the three core evaluation metrics used in this project for object detection evaluation.

## Overview

We implement three complementary metrics to evaluate detection quality:

1. **Detection P/R/F1 at IoU Threshold** - Box-level correctness
2. **Per-Class Metrics + Confusion Matrix** - Class-specific performance
3. **Counting Quality (MAE)** - Accuracy for duplicate objects

All metrics use **greedy one-to-one matching** with IoU thresholds for consistency.

---

## 1. Detection P/R/F1 at IoU Threshold

### Purpose
Evaluate box-level correctness: Does the model detect objects correctly?

### Definitions

**True Positive (TP):**
- Predicted box matched to a ground truth box with IoU ≥ threshold (default: 0.5)
- One-to-one matching: each GT can match at most one prediction

**False Positive (FP):**
- Predicted box that doesn't match any ground truth

**False Negative (FN):**
- Ground truth box that doesn't match any prediction

**Metrics:**
- **Precision** = TP / (TP + FP) — What fraction of predictions are correct?
- **Recall** = TP / (TP + FN) — What fraction of GT objects are detected?
- **F1 Score** = 2 × (P × R) / (P + R) — Harmonic mean of P and R

### Matching Strategy

**Greedy Matching (per image, per class):**
1. Compute IoU between all predicted and GT boxes
2. Sort pairs by IoU descending
3. Greedily assign matches (highest IoU first)
4. Each prediction can match at most one GT
5. Each GT can match at most one prediction
6. Match is valid only if IoU ≥ threshold

### Threshold Sweep

We evaluate at **multiple confidence thresholds** (e.g., 0.0, 0.1, ..., 0.9):
- Filter predictions by confidence before matching
- Plot P/R/F1 vs confidence to find optimal threshold
- Use **validation set** to select best threshold
- Report test set metrics at that threshold

### Usage

```python
from evaluation.metrics import eval_detection_prf_at_iou
from evaluation.io import load_predictions, load_ground_truth

preds = load_predictions("path/to/predictions.json")
gts = load_ground_truth("path/to/test_index.json")

results = eval_detection_prf_at_iou(
    preds, gts,
    iou_threshold=0.5,
    conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
)

# Find best threshold by F1
best_thr = max(results.keys(), key=lambda k: results[k]['f1'])
print(f"Best F1={results[best_thr]['f1']:.3f} at conf={best_thr}")
```

---

## 2. Per-Class Metrics + Confusion Matrix

### Purpose
Identify which ingredient classes are hard to detect and what confusions occur.

### Per-Class Metrics

For each class, compute:
- **Precision**: Of all predictions for this class, what fraction are correct?
- **Recall**: Of all GT objects of this class, what fraction are detected?
- **F1 Score**: Harmonic mean
- **Support**: Number of GT objects of this class

### Confusion Matrix

**What it shows:**
- Rows = True class
- Cols = Predicted class
- Cell (i, j) = number of times true class i was predicted as class j

**Important:**
- Only includes **matched detections** (IoU ≥ threshold)
- Unmatched predictions → FP (not in confusion matrix)
- Unmatched GT → FN (not in confusion matrix)

**Top Confusions:**
- Extract off-diagonal elements (true ≠ pred)
- Sort by frequency
- Example: "Garlic → Onion: 5 times"

### Usage

```python
from evaluation.metrics import eval_per_class_metrics_and_confusions

results = eval_per_class_metrics_and_confusions(
    preds, gts,
    iou_threshold=0.5,
    conf_threshold=0.5,  # Use best threshold from threshold sweep
    class_names=class_names
)

# Per-class F1
for class_name, metrics in results['per_class'].items():
    print(f"{class_name}: F1={metrics['f1']:.3f}, support={metrics['support']}")

# Top confusions
for conf in results['top_confusions'][:5]:
    print(f"{conf['true_class']} → {conf['pred_class']}: {conf['count']} times")
```

---

## 3. Counting Quality (MAE)

### Purpose
Evaluate accuracy for images with **duplicate objects** (e.g., "2 carrots, 3 potatoes").

Standard detection metrics (P/R/F1) are box-level and don't capture counting errors well.

### Why Counting Matters

Example:
- GT: 3 carrots
- Pred: 1 carrot (correctly detected)
- Standard metrics: TP=1, FN=2 (doesn't emphasize counting)
- Counting metric: |1 - 3| = 2 error

### Two Counting Methods

We compute **both methods** for robustness:

#### Method 1: Matched-Only (Recommended)
- **pred_count** = number of **TPs** (matched detections)
- Robust to false positive spam
- Conservative: only counts correctly detected objects

#### Method 2: All-Predictions
- **pred_count** = all predicted boxes above conf_threshold
- May be inflated by false positives
- Simpler, more direct

### MAE (Mean Absolute Error)

For each image:
1. Count GT objects per class: `gt_count[class]`
2. Count predicted objects per class: `pred_count[class]`
3. Compute error: `sum over classes |pred_count[class] - gt_count[class]|`

**Global MAE** = average error over all images

### Usage

```python
from evaluation.metrics import eval_counting_quality

results = eval_counting_quality(
    preds, gts,
    iou_threshold=0.5,
    conf_threshold=0.5,
    class_names=class_names
)

print(f"Matched-only MAE: {results['matched_only']['global_mae']:.4f}")
print(f"All-predictions MAE: {results['all_predictions']['global_mae']:.4f}")

# Per-class MAE
for class_name, mae in results['matched_only']['per_class_mae'].items():
    print(f"{class_name}: MAE={mae:.4f}")
```

---

## Threshold Selection Protocol

**IMPORTANT:** Use validation set for hyperparameter selection, test set for final reporting.

### Step 1: Evaluate Validation Set

```python
# Evaluate on validation set at multiple thresholds
val_preds = load_predictions("path/to/val_predictions.json")
val_gts = load_ground_truth("path/to/val_index.json")

val_results = eval_detection_prf_at_iou(
    val_preds, val_gts,
    conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)

# Select best threshold by F1 on validation
best_conf_thr = max(val_results.keys(), key=lambda k: val_results[k]['f1'])
```

### Step 2: Evaluate Test Set

```python
# Evaluate test set at the chosen threshold
test_preds = load_predictions("path/to/test_predictions.json")
test_gts = load_ground_truth("path/to/test_index.json")

test_results = eval_detection_prf_at_iou(
    test_preds, test_gts,
    conf_thresholds=[best_conf_thr]  # Only the selected threshold
)

# Report test F1
print(f"Test F1 at conf={best_conf_thr}: {test_results[best_conf_thr]['f1']:.4f}")
```

### Step 3: Report Both

**In your report/paper:**
- "Best threshold selected on validation: 0.5 (F1=0.78)"
- "Test set performance at this threshold: F1=0.76"

**Never:**
- Pick threshold based on test set
- Report only test metrics without validation justification

---

## Train/Val/Test Usage

### When to Evaluate Each Split

**Train Set:**
- Check for overfitting
- Expected: Very high metrics (model should fit training data)
- If train F1 << val F1: underfitting

**Validation Set:**
- Hyperparameter selection (conf_threshold, freeze depth, epochs, etc.)
- Model checkpointing (save best val epoch)
- Expected: Lower than train (generalization gap)

**Test Set:**
- Final reporting only
- Never use for hyperparameter tuning
- Expected: Similar to val (slight drop is normal)

### Typical Pattern

```
Train F1:  0.95  ← Model fits training data well
Val F1:    0.78  ← Some generalization gap (normal)
Test F1:   0.76  ← Similar to val (good!)
```

**Red flags:**
- Val << Train: Overfitting
- Test << Val: Lucky val split or data distribution shift

---

## File Formats

### Predictions JSON

```json
{
  "run_id": "baseline_yolo",
  "split": "test",
  "model_family": "yolo",
  "inference_settings": {
    "conf_threshold": 0.01,
    "iou_threshold": 0.50
  },
  "predictions": [
    {
      "image_id": "img_001",
      "detections": [
        {
          "class_id": 9,
          "class_name": "Capsicum",
          "confidence": 0.89,
          "bbox": [236, 1, 534, 170],
          "bbox_format": "xyxy"
        }
      ]
    }
  ]
}
```

### Ground Truth JSON

```json
{
  "metadata": {
    "split": "test",
    "num_images": 400,
    "class_names": {"0": "Asparagus", "1": "Avocado", ...}
  },
  "images": [
    {
      "image_id": "img_001",
      "image_filename": "img_001.jpg",
      "ground_truth": [
        {
          "class_id": 9,
          "class_name": "Capsicum",
          "bbox_xyxy": [236, 1, 534, 170]
        }
      ]
    }
  ]
}
```

---

## CLI Usage

### Standalone Evaluation

```bash
# Evaluate from saved predictions
python scripts/evaluate_run.py \
  --predictions evaluation/metrics/baseline_yolo_test_predictions.json \
  --ground_truth data/processed/evaluation/test_index.json \
  --output_dir evaluation/results/baseline_yolo/test/ \
  --run_name "Baseline YOLO" \
  --conf_thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

This generates:
- `metrics.json` - All metrics in one file
- `summary.csv` - Quick table for reports
- `threshold_sweep.png` - P/R/F1 vs confidence
- `per_class_f1.png` - Bar chart per class
- `confusion_matrix.png` - Confusion heatmap
- `count_mae_comparison.png` - Counting comparison

---

## Notebook Usage

```python
# Import evaluation module
from evaluation.metrics import (
    eval_detection_prf_at_iou,
    eval_per_class_metrics_and_confusions,
    eval_counting_quality
)
from evaluation.io import load_predictions, load_ground_truth
from evaluation.plots import plot_all_metrics

# Load data
preds = load_predictions("path/to/predictions.json")
gts = load_ground_truth("path/to/index.json")

# Run metrics
prf = eval_detection_prf_at_iou(preds, gts)
per_class = eval_per_class_metrics_and_confusions(preds, gts, conf_threshold=0.5)
counting = eval_counting_quality(preds, gts, conf_threshold=0.5)

# Generate plots
plot_all_metrics(
    threshold_sweep=prf,
    per_class_results=per_class['per_class'],
    confusion_data=per_class,
    counting_results=counting,
    output_dir="results/my_run/",
    run_name="My Experiment"
)
```

---

## FAQ

**Q: Why greedy matching instead of Hungarian algorithm?**
A: Greedy is simpler, faster, and standard in detection evaluation. Hungarian gives marginally better global matching but is overkill for one-to-one constraints.

**Q: Why IoU=0.5 as default?**
A: Standard in COCO and PASCAL VOC. Higher IoU (0.75) is stricter, lower (0.25) is lenient. We use 0.5 for fair comparison.

**Q: Should I use matched-only or all-predictions for counting?**
A: **Matched-only is recommended** (robust to FP spam). Report both for completeness.

**Q: How do I compare YOLO vs RT-DETR?**
A: Run evaluation on both using the **same test set** and **same thresholds**. Use validation to pick best conf_threshold for each model separately.

**Q: Can I change the evaluation code after running experiments?**
A: **No!** Freeze evaluation code before experiments. Changing metrics mid-project invalidates comparisons.

---

## References

- COCO Detection Evaluation: https://cocodataset.org/#detection-eval
- PASCAL VOC: http://host.robots.ox.ac.uk/pascal/VOC/
- Greedy vs Hungarian Matching: https://arxiv.org/abs/1406.4729

---

## Contact

For questions or issues with the evaluation system, open an issue in the repo or contact the project maintainers.
```
