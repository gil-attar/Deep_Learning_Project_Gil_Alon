# Evaluation Directory

This directory contains all evaluation artifacts for the baseline models.

## Structure

```
evaluation/
├── metrics/           # JSON files with evaluation results
│   ├── baseline_yolo_run.json          # YOLOv8 run metadata
│   ├── baseline_yolo_metrics.json      # YOLOv8 aggregate metrics
│   ├── baseline_yolo_predictions.json  # YOLOv8 per-image predictions
│   ├── baseline_rtdetr_run.json        # RT-DETR run metadata
│   ├── baseline_rtdetr_metrics.json    # RT-DETR aggregate metrics
│   └── baseline_rtdetr_predictions.json # RT-DETR per-image predictions
└── plots/             # Visualizations and graphs
    ├── yolo_baseline/
    └── rtdetr_baseline/
```

## JSON File Descriptions

### 1. Run Metadata Files (`*_run.json`)

**Purpose:** Document exactly what was run for reproducibility.

**Contains:**
- Model configuration (family, name, weights path)
- Dataset references (split manifest, test index, data.yaml)
- Inference settings (imgsz, conf_threshold, iou_threshold)
- Hardware/environment info (GPU, Python version, Ultralytics version)
- Output paths

**Example:** [baseline_yolo_run.json](metrics/baseline_yolo_run.json)

### 2. Metrics Files (`*_metrics.json`)

**Purpose:** Store aggregate performance metrics for architecture comparison.

**Contains:**
- mAP@50, mAP@50-95
- Precision, Recall
- FPS (frames per second)
- Timing breakdown (total inference time, avg time per image)

**Example:** [baseline_yolo_metrics.json](metrics/baseline_yolo_metrics.json)

### 3. Predictions Files (`*_predictions.json`)

**Purpose:** Store per-image detections for occlusion analysis and threshold sweeps.

**Contains:**
- For each test image:
  - Image ID and path
  - List of detections (class_id, class_name, confidence, bbox in xyxy format)
  - Inference time

**Example:** [baseline_yolo_predictions.json](metrics/baseline_yolo_predictions.json)

**Note:** Predictions files enable Step 3.3 (occlusion difficulty slicing) and Step 3.4 (threshold sweep) without re-running inference.

## How to Generate These Files

### In Google Colab (Recommended for Training)

Run the [02_train_models.ipynb](../notebooks/02_train_models.ipynb) notebook which:

1. Trains both models
2. Saves weights to `models/`
3. Runs evaluation script automatically
4. Generates all 6 JSON files

### Standalone Evaluation (After Training)

If you already have trained weights:

```bash
# Create data.yaml first
python scripts/create_data_yaml.py \
    --dataset_root data/raw \
    --output data/processed/data.yaml

# Run evaluation for both models
python scripts/evaluate_baseline.py \
    --yolo_weights models/yolov8n_baseline.pt \
    --rtdetr_weights models/rtdetr_baseline.pt \
    --dataset_root data \
    --output_dir evaluation/metrics \
    --conf_threshold 0.25 \
    --imgsz 640 \
    --model both
```

### Evaluate Only One Model

```bash
# Evaluate only YOLO
python scripts/evaluate_baseline.py \
    --yolo_weights models/yolov8n_baseline.pt \
    --model yolo

# Evaluate only RT-DETR
python scripts/evaluate_baseline.py \
    --rtdetr_weights models/rtdetr_baseline.pt \
    --model rtdetr
```

## Verification

After running evaluation, verify all files were created:

```python
import os

required_files = [
    "evaluation/metrics/baseline_yolo_run.json",
    "evaluation/metrics/baseline_rtdetr_run.json",
    "evaluation/metrics/baseline_yolo_metrics.json",
    "evaluation/metrics/baseline_rtdetr_metrics.json",
    "evaluation/metrics/baseline_yolo_predictions.json",
    "evaluation/metrics/baseline_rtdetr_predictions.json"
]

for f in required_files:
    print(f"{'✓' if os.path.exists(f) else '✗'} {f}")
```

## Next Steps

Once all 6 JSON files are generated:

1. **Step 3.3:** Occlusion difficulty analysis
   - Slice predictions by Easy/Medium/Hard using `test_index.json`
   - Compute per-difficulty metrics (Recall, Precision, False Negatives)

2. **Step 3.4:** Confidence threshold sweep
   - Use saved confidences in predictions JSONs
   - Test thresholds: 0.1, 0.25, 0.5, 0.75
   - Find optimal threshold for each model

3. **Step 4:** Generate visualizations
   - Performance comparison graphs
   - Precision-Recall curves
   - Occlusion difficulty breakdown

## Important Notes

- **Do NOT commit** `.pt` weight files to Git (they're large)
- **DO commit** all JSON files (they're lightweight and essential for reproducibility)
- All JSON files use the **same test set** defined in `data/processed/evaluation/test_index.json`
- Bounding boxes are in **xyxy format** for consistency
- `conf_threshold=0.25` is the baseline; other thresholds tested in Step 3.4
