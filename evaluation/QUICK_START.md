# Evaluation System - Quick Start

## 📦 What's Included

This evaluation system provides comprehensive metrics for object detection models (YOLOv8 and RT-DETR).

**Metrics:**
1. **Detection P/R/F1** - Box-level correctness at multiple confidence thresholds
2. **Per-Class Metrics** - Class-specific performance + confusion matrix
3. **Counting Quality** - Accuracy for duplicate objects (e.g., "2 carrots")

**Features:**
- ✅ Train/Val/Test split support
- ✅ Threshold sweep for hyperparameter selection
- ✅ Automatic plot generation
- ✅ CLI and notebook-friendly
- ✅ Consistent across YOLO and RT-DETR

---

## 🚀 Quick Start (3 Steps)

### Step 1: Build Ground Truth Indices

```bash
# Generate train/val/test index files (ONE TIME ONLY)
python scripts/build_evaluation_indices.py \
  --dataset_root data/raw \
  --output_dir data/processed/evaluation
```

This creates:
- `data/processed/evaluation/train_index.json`
- `data/processed/evaluation/val_index.json`
- `data/processed/evaluation/test_index.json`

### Step 2: Generate Predictions

Train your model and save predictions in this format:

```json
{
  "run_id": "my_experiment",
  "split": "test",
  "model_family": "yolo",
  "predictions": [
    {
      "image_id": "img_001",
      "detections": [
        {
          "class_id": 9,
          "class_name": "Capsicum",
          "confidence": 0.89,
          "bbox": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

**Important:** Save predictions with **low confidence threshold** (e.g., 0.01) so you can filter post-hoc.

### Step 3: Run Evaluation

```bash
# CLI
python scripts/evaluate_run.py \
  --predictions evaluation/metrics/my_run_test_predictions.json \
  --ground_truth data/processed/evaluation/test_index.json \
  --output_dir evaluation/results/my_run/test/
```

Or in notebook:

```python
from evaluation.metrics import *
from evaluation.io import *
from evaluation.plots import plot_all_metrics

# Load data
preds = load_predictions("path/to/predictions.json")
gts = load_ground_truth("path/to/test_index.json")

# Run metrics
prf = eval_detection_prf_at_iou(preds, gts)
per_class = eval_per_class_metrics_and_confusions(preds, gts, conf_threshold=0.5)
counting = eval_counting_quality(preds, gts, conf_threshold=0.5)

# Plot
plot_all_metrics(prf, per_class['per_class'], per_class, counting, "results/", "My Run")
```

---

## 📊 Outputs

After running evaluation, you'll get:

```
evaluation/results/my_run/test/
├── metrics.json              # All metrics in one file
├── summary.csv               # Quick table for reports
├── threshold_sweep.png       # P/R/F1 vs confidence
├── per_class_f1.png          # Bar chart per class
├── confusion_matrix.png      # Confusion heatmap
└── count_mae_comparison.png  # Counting accuracy
```

---

## 🧪 Test the System

Run the test notebook to make sure everything works:

```bash
jupyter notebook notebooks/test_evaluation_system.ipynb
```

This trains a tiny model (5 epochs) and runs the full evaluation pipeline.

---

## 📖 Full Documentation

See [README_METRICS.md](README_METRICS.md) for:
- Detailed metric definitions
- Matching algorithms
- Threshold selection protocol
- Train/Val/Test usage guidelines
- FAQ

---

## 🎯 Typical Workflow

### For Experiments:

1. **Train on train set**
2. **Generate predictions on all 3 splits** (train/val/test)
3. **Evaluate val set** → pick best conf_threshold
4. **Evaluate test set** at that threshold → final reporting
5. **Evaluate train set** → check for overfitting

### Example:

```bash
# Val set: find best threshold
python scripts/evaluate_run.py \
  --predictions val_predictions.json \
  --ground_truth data/processed/evaluation/val_index.json \
  --output_dir results/my_model/val/

# Test set: final reporting (use best threshold from val)
python scripts/evaluate_run.py \
  --predictions test_predictions.json \
  --ground_truth data/processed/evaluation/test_index.json \
  --output_dir results/my_model/test/ \
  --conf_thresholds 0.5  # Best threshold from val
```

---

## 🔧 Module Structure

```
evaluation/
├── __init__.py           # Module entry point
├── io.py                 # Load/save predictions and ground truth
├── matching.py           # IoU computation + greedy matching
├── metrics.py            # 3 core metric functions
├── plots.py              # Visualization functions
├── README_METRICS.md     # Full documentation
└── QUICK_START.md        # This file
```

---

## ⚠️ Important Rules

1. **Freeze evaluation code before experiments** - Don't change metrics mid-project!
2. **Use validation for hyperparameter selection** - Never pick thresholds based on test
3. **Save predictions with low conf_threshold** - Filter post-hoc for flexibility
4. **Same test set for all experiments** - Ensures fair comparison

---

## 💡 Tips

- Use `matched_only` counting method (robust to FP spam)
- Evaluate all 3 splits to detect overfitting
- Compare models at their individually-optimal thresholds (picked on val)
- Check confusion matrix to identify problematic class pairs

---

## 🐛 Troubleshooting

**"FileNotFoundError: predictions file not found"**
→ Make sure you've run inference and saved predictions first

**"Predictions count != GT count"**
→ Check that image_ids match between predictions and ground truth

**"No predictions found"**
→ Check your prediction JSON format (see Step 2 above)

**"Import error: evaluation module"**
→ Make sure you're running from project root, or add to sys.path

---

## 📞 Need Help?

- Check [README_METRICS.md](README_METRICS.md) for detailed explanations
- Run [test_evaluation_system.ipynb](../notebooks/test_evaluation_system.ipynb) to debug
- Open an issue in the repo

---

Good luck with your experiments! 🚀
