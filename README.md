# CNN vs Transformer for Occluded Object Detection

Deep Learning course project comparing YOLOv8 (CNN) vs RT-DETR (Transformer) architectures for ingredient detection, with focus on occlusion robustness.

**Authors:** Gil & Alon

---

## Project Overview

This project investigates two research questions:

1. **Experiment 1 (Freezing Ladder):** How does the number of trainable layers during fine-tuning affect detection performance?
2. **Experiment 3 (Channel Masking):** Can internal feature masking during training improve occlusion robustness?

### Models
- **YOLOv8m** - CNN-based single-stage detector
- **RT-DETR-L** - Transformer-based real-time detector

### Dataset
- **Source:** Roboflow Food Ingredients Dataset (26 classes)
- **Split:** 1384 train / 200 val / 400 test images
- **Format:** YOLO (images + bounding box labels)

---

## Repository Structure

```
Deep_Learning_Gil_Alon/
├── data/
│   ├── raw/                      # Original dataset (downloaded via script)
│   └── processed/
│       ├── evaluation/           # Ground truth indices (train/val/test_index.json)
│       └── splits/               # Split manifest for reproducibility
│
├── evaluation/                   # Custom evaluation pipeline
│   ├── __init__.py
│   ├── io.py                     # Load predictions and ground truth
│   ├── matching.py               # IoU-based prediction-to-GT matching
│   ├── metrics.py                # P/R/F1, per-class metrics, counting MAE
│   └── plots.py                  # Visualization functions
│
├── experiments/
│   ├── Experiment_1/             # Freezing Ladder (see README inside)
│   │   ├── README.md
│   │   ├── freeze_presets.py
│   │   ├── runOneTest.py
│   │   └── eval_contract.json
│   │
│   └── Experiment_3/             # Channel Masking (see README inside)
│       ├── README.md
│       ├── mask_presets.py
│       ├── channel_masking.py
│       └── debug_logger.py
│
├── notebooks/
│   └── test_evaluation_system.ipynb
│
├── scripts/
│   ├── download_dataset.py       # Download from Roboflow
│   ├── build_evaluation_indices.py
│   └── generate_synthetic_occlusions.py
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
export ROBOFLOW_API_KEY="your_api_key"
python scripts/download_dataset.py --output_dir data/raw
```

### 3. Build Evaluation Indices

```bash
python scripts/build_evaluation_indices.py \
    --dataset_root data/raw \
    --output_dir data/processed/evaluation
```

### 4. Run Experiments

Each experiment has its own README with detailed instructions:

- **Experiment 1:** See [experiments/Experiment_1/README.md](experiments/Experiment_1/README.md)
- **Experiment 3:** See [experiments/Experiment_3/README.md](experiments/Experiment_3/README.md)

Experiments are designed to run in **Google Colab** with GPU acceleration.

---

## Evaluation Pipeline

We use a custom evaluation system (not Ultralytics' built-in `model.val()`) for consistency across experiments.

### Metrics Computed
- **Threshold Sweep:** P/R/F1 at confidence thresholds 0.0-0.9
- **Per-Class Metrics:** F1 score per ingredient class
- **Confusion Matrix:** Classification errors for matched detections
- **Counting MAE:** How accurately the model counts objects

### Usage

```python
from evaluation.io import load_predictions, load_ground_truth
from evaluation.metrics import eval_detection_prf_at_iou

predictions = load_predictions("path/to/predictions.json")
ground_truth = load_ground_truth("data/processed/evaluation/test_index.json")

results = eval_detection_prf_at_iou(predictions, ground_truth, iou_threshold=0.5)
print(f"Best F1: {max(r['f1'] for r in results.values())}")
```

---

## Experiments Summary

### Experiment 1: Freezing Ladder

**Question:** How many layers should we fine-tune?

| Preset | Layers Trained | Description |
|--------|----------------|-------------|
| F0 | Head only | Minimal fine-tuning |
| F1 | Head + Neck | Moderate fine-tuning |
| F2 | Head + Neck + Late Backbone | Recommended |
| F3 | All layers | Full fine-tuning |

**Key Finding:** F2 (partial fine-tuning) achieved best balance of performance and generalization.

### Experiment 3: Channel Masking vs Occlusion Training

**Question:** Can masking feature channels simulate occlusion robustness?

| Session | Training Data | Masking Location |
|---------|---------------|------------------|
| S1 | Clean | None (baseline) |
| S2 | 40% Occluded | None |
| S3 | Clean | Backbone Early |
| S4 | Clean | Backbone Late |
| S5 | Clean | Neck |
| S6 | Clean | Head |

**Key Finding:** Channel masking does NOT improve occlusion robustness. S2 (occluded training) achieved 81% F1 on occluded images but exhibited catastrophic forgetting on clean images.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Ultralytics 8.0+
- See `requirements.txt` for full list

---

## License

Academic project for Deep Learning course.
