# Experiment 1 Run Contract (Frozen Protocol)

This document defines the invariants for Experiment 1 ("Freeze-Ladder Fine-Tuning") so that all runs are comparable and compatible with the project's evaluation system.

## Scope
- Models: YOLOv8m (COCO pretrained), RT-DETR-L (COCO pretrained)
- Freeze regimes: F0, F1, F2 (K=2), F3
- One run per (model, freeze_id): each run must start from fresh pretrained weights.

## Frozen inputs (do not change during Experiment 1)
The dataset split and evaluation indices are treated as immutable for Experiment 1.

- Split manifest:
  - `data/processed/splits/split_manifest.json`
- Evaluation indices (ground truth):
  - `data/processed/evaluation/train_index.json`
  - `data/processed/evaluation/val_index.json`
  - `data/processed/evaluation/test_index.json`

Policy:
- Do not regenerate split files or indices unless the dataset itself changes. If regeneration is required, bump the experiment contract version and re-run all configurations.

## Canonical run directory layout
Each configuration writes to:

`experiments/Experiment_1/runs/<model>/<freeze_id>/`

Required structure:

experiments/Experiment_1/runs/<model>/<freeze_id>/
run_manifest.json
predictions/
val_predictions.json
test_predictions.json
eval/
val/
metrics.json
summary.csv
plots/
threshold_sweep.png
per_class_f1.png
confusion_matrix.png
count_mae_comparison.png
test/
metrics.json
summary.csv
plots/
threshold_sweep.png
per_class_f1.png
confusion_matrix.png
count_mae_comparison.png
ultralytics/
args.yaml (or a copied snapshot of Ultralytics training args)


Notes:
- If Ultralytics writes its own run folder elsewhere, record that path in `run_manifest.json` and optionally copy its `args.yaml` into `ultralytics/`.

## Evaluation contract
The evaluation system consumes:
1) Predictions JSON (per image detections) and
2) Ground truth index JSON (`*_index.json`).

### Bounding box format
- Predictions must use **xyxy pixel coordinates**:
  - `bbox: [x1, y1, x2, y2]`
- `image_id` in predictions must exactly match the `image_id` strings in the corresponding index.

### Matching rule
- IoU threshold: **0.5**

### Confidence thresholding policy
- Predictions must be exported using a **low confidence cutoff** so that thresholding is performed post-hoc by the evaluator:
  - `prediction_export_conf_low = 0.01`

### Threshold selection and reporting
- Select the best confidence threshold on **VAL** by maximizing **F1**.
- Report **TEST** metrics at that selected validation threshold.
- Do not choose thresholds based on test results.

### Fixed threshold sweep grid (VAL selection)
- `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`

## What varies across runs
Only these items vary across Experiment 1 runs:
- `model` (yolov8m vs rtdetr-l)
- `freeze_id` (F0–F3)

All other settings are treated as fixed by this contract (split, indices, evaluation rules).
