# which detector “understands the scene” better? CNN detector (YOLOv8m)   VS   Transformer detector (RT-DETR-L)

Deep Learning course project comparing YOLOv8 (CNN) vs RT-DETR (Transformer) architectures for ingredient detection, with focus on occlusion robustness.

**Authors:** Gil Attar & Alon Shorr

---

## Project Overview

This project investigates two research questions:

1. **Experiment 1 (Freezing Ladder):** How does the number of trainable layers during fine-tuning affect detection performance?
2. **Experiment 2 (Training Duration):** How does training duration affects performance?
3. **Experiment 3 (Channel Masking):** Can internal feature masking during training improve occlusion robustness?

### Models
- **YOLOv8m** - CNN-based single-stage detector
- **RT-DETR-L** - Transformer-based real-time detector

### Dataset
- **Source:** Roboflow Food Ingredients Dataset (26 classes)
- **Split:** 1384 train / 200 val / 400 test images
- **Format:** YOLO (images + bounding box labels)
- **VERY IMPORTANT NOTE:** the current code runs on dataset activated automaticly by our own private Roboflow API key. This key is going to be DELETED after our project is over. in case you want to run the code yourself, you need to set your own API key (pay to roboflow, download the database from https://universe.roboflow.com/samuels/food-ingredients-dataset-2/browse?queryText=&pageSize=50&startingIndex=650&browseQuery=true, create a new project from it, and make a version from it with train/val/test splits with the augmentations you want). To run, simply change in each experiment the data download line: os.environ["ROBOFLOW_API_KEY"] = "<enter your key here>"


---

## Repository Structure

```
Deep_Learning_Gil_Alon/
├── artifacts/                    # Run data can be stored here (unless storing it on google drive)
│
├── data/
│   ├── raw/                      # Original dataset (downloaded when running experiments)
│   └── processed/
│       ├── evaluation/           # Ground truth indices (train/val/test_index.json)
│       └── splits/               # Split manifest for reproducibility
│
├── evaluation/                   # Custom evaluation pipeline
│   ├── __init__.py               # Python package initialization
│   ├── io.py                     # Load/save predictions and ground truth
│   ├── matching.py               # IoU computation and greedy matching
│   ├── metrics.py                # P/R/F1, per-class metrics, counting MAE
│   ├── plots.py                  # Visualization functions
│   ├── plots/                    # Empty placeholder (plots saved to results dir)
│   ├── QUICK_START.md           # Updated - quick reference guide
│   ├── README_METRICS.md        # Updated - full documentation
│   └── evaluation_summery.txt   # Updated - module overview
│
├── experiments/
│   ├── Experiment_1/             # Freezing Ladder (see README inside)
│   │   ├── README.md
│   │   ├── freeze_presets.py
│   │   ├── runOneTest.py
│   │   └── eval_contract.json
│   │
│   ├── Experiment_2/             # Training Duration (see README inside)
│   │   ├── README.md
│   │   ├── E2_run_evaluate.ipynb
│   │   └── runOneTest.py
│   │
│   └── Experiment_3/             # Channel Masking (see README inside)
│       ├── README.md
│       ├── E3_full_run/
│       │   ├── E3_run_evaluate_FINAL_RUN.ipynb       # Final full run
│       │   └── E3_run_evaluate_S2_DEBUGCHECK.ipynb   # Debug run (S2 only - proves domain shift)
│       ├── E3_run_evaluate.ipynb
│       ├── mask_presets.py
│       ├── channel_masking.py
│       └── debug_logger.py
│
├── legacy/                       # Legacy dir, only for documantation purposes
│
├── notebooks/
│   └── evaluation_system_mockup.ipynb  # general pipeline to test data pull & evaluation system
│
├── scripts/
│   ├── download_dataset.py           # Download dataset from Roboflow
│   ├── build_evaluation_indices.py   # Create train/val/test index JSONs
│   ├── create_data_yaml.py           # Create data.yaml for Ultralytics
│   ├── generate_synthetic_occlusions.py  # Create occluded test sets (E3)
│   ├── evaluate_run.py               # Standalone evaluation from predictions
│   └── fetch_weights.sh              # Download pretrained model weights
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
- **Experiment 2:** See [experiments/Experiment_2/README.md](experiments/Experiment_2/README.md)
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

### Experiment 2: Training Duration

**Question:** How does training duration affects performance?

| Epochs | Purpose |
|--------|---------|
| 5, 10, 20, 40, 80 | Compare convergence and generalization |

**Key Finding:** Transformer can reach the same results as CNN but it takes more epochs.

### Experiment 3: Channel Masking vs Occlusion Training

**Question:** Can masking feature channels simulate occlusion robustness?

| Session | Training Data | Masking Location |
|---------|---------------|------------------|
| S1      | Clean         | None (baseline)  |
| S2      | 40% Occluded  | None             |
| S3      | Clean         | Backbone Early   |
| S4      | Clean         | Backbone Late    |
| S5      | Clean         | Neck             |
| S6      | Clean         | Head             |

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
