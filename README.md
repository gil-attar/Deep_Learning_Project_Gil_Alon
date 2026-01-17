# The Attention-Based Chef 🍳

Deep Learning course project comparing CNN (YOLOv8) vs Transformer (RT-DETR) architectures for ingredient detection from food images, with recipe generation via OpenAI API.

**Authors:** Gil & Alon

## Project Overview

This project addresses the question: **Do Transformers handle occlusion better than CNNs in object detection?**

We compare:
- **YOLOv8** (CNN-based, single-stage detector)
- **RT-DETR** (Transformer-based, real-time DETR)

### Key Novelties
1. **Occlusion Difficulty Analysis** - Test images classified as Easy/Medium/Hard based on bounding box overlap
2. **Confidence Calibration Study** - Analyzing prediction confidence across difficulty levels
3. **Attention Map Visualization** - How Transformers attend to occluded objects

---

## Step 2: Data Pipeline & Evaluation Foundations (FROZEN)

This section documents the data protocol. **Do not modify after Step 2 is complete.**

### Dataset

- **Source:** Roboflow - `gaworkspace-utcbg/food-ingredients-dataset-2-rewtd` (version 1)
- **Format:** YOLOv8 (images + YOLO format labels)
- **Location:** `data/raw/` (read-only, immutable)

### Data Split (Fixed)

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 1384   | 70%        |
| Valid | 200    | 10%        |
| Test  | 400    | 20%        |

The test set is **fixed forever**. All experiments evaluate on the same 396 test images.

### Occlusion Difficulty Definition (Core Novelty)

Difficulty is assigned based on **MAXIMUM pairwise IoU** between ground-truth bounding boxes:

| Difficulty | Criterion | Meaning |
|------------|-----------|---------|
| **Easy**   | max_iou < 0.05 | No significant overlap |
| **Medium** | 0.05 ≤ max_iou < 0.15 | Partial overlap |
| **Hard**   | max_iou ≥ 0.15 | Significant occlusion |

#### IoU Computation
```
For each test image:
  1. Get all ground-truth bounding boxes
  2. Compute pairwise IoU for all box pairs
  3. Take the MAXIMUM IoU value
  4. Assign difficulty based on thresholds above
```

**These thresholds are frozen and must not change.**

### Step 2 Artifacts

All artifacts are in `data/processed/`:

| File | Purpose |
|------|---------|
| `splits/split_manifest.json` | Lists all filenames per split (reproducibility) |
| `evaluation/test_index.json` | Ground truth + difficulty for each test image |
| `evaluation/difficulty_summary.csv` | Statistics per difficulty level |

### Script Usage

```bash
# Download dataset to data/raw/
python scripts/download_dataset.py --output_dir data/raw

# Build evaluation artifacts (creates all Step 2 outputs)
python scripts/build_evaluation_index.py --dataset_root data/raw --output_dir data/processed --seed 42
```

### Repository Structure

```
Deep_Learning_Gil_Alon/
├── data/
│   ├── raw/                    # Immutable dataset (not committed)
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml
│   └── processed/
│       ├── splits/
│       │   └── split_manifest.json     # ✓ Committed
│       └── evaluation/
│           ├── test_index.json         # ✓ Committed
│           └── difficulty_summary.csv  # ✓ Committed
├── scripts/
│   ├── download_dataset.py
│   ├── build_evaluation_index.py
│   └── merge_valid_into_train.py       # Phase 2 only (not Step 2)
├── notebooks/
│   └── 01_data_pipeline.ipynb
├── models/                     # Trained model weights (not committed)
├── requirements.txt
└── README.md
```

---

## Step 3: Baseline Model Training & Evaluation (READY TO RUN)

This section implements baseline training and evaluation for YOLOv8 and RT-DETR.

### Workflow

#### Option A: Google Colab (Recommended)

1. Open [notebooks/02_train_models.ipynb](notebooks/02_train_models.ipynb) in Colab
2. Run all cells sequentially:
   - Downloads dataset
   - Trains YOLOv8n (50 epochs)
   - Trains RT-DETR-l (50 epochs)
   - Generates all 6 evaluation JSON files
   - Saves weights and JSONs to Google Drive

**Expected outputs:**
```
models/
├── yolov8n_baseline.pt       # ~6 MB
└── rtdetr_baseline.pt        # ~100 MB

evaluation/metrics/
├── baseline_yolo_run.json          # Run metadata
├── baseline_yolo_metrics.json      # Aggregate metrics
├── baseline_yolo_predictions.json  # Per-image predictions
├── baseline_rtdetr_run.json
├── baseline_rtdetr_metrics.json
└── baseline_rtdetr_predictions.json
```

#### Option B: Local/WSL Training (If you have GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
export ROBOFLOW_API_KEY="your_api_key"
python scripts/download_dataset.py --output_dir data/raw

# Create data.yaml
python scripts/create_data_yaml.py --dataset_root data/raw --output data/processed/data.yaml

# Train models (requires GPU)
# ... (not recommended for 3-week timeline - use Colab's free GPU instead)
```

#### Option C: Evaluation Only (If you already have weights)

If you've already trained models and just need to generate evaluation JSONs:

```bash
python scripts/evaluate_baseline.py \
    --yolo_weights models/yolov8n_baseline.pt \
    --rtdetr_weights models/rtdetr_baseline.pt \
    --dataset_root data \
    --output_dir evaluation/metrics \
    --model both
```

### What Gets Generated

The evaluation script creates **6 JSON files** (see [evaluation/README.md](evaluation/README.md)):

1. **Run Metadata** (`*_run.json`): Reproducibility info (model config, dataset refs, hardware)
2. **Metrics** (`*_metrics.json`): Aggregate performance (mAP@50, precision, recall, FPS)
3. **Predictions** (`*_predictions.json`): Per-image detections (for occlusion analysis)

### Verification

After training completes:

```python
import json

# Check metrics comparison
with open("evaluation/metrics/baseline_yolo_metrics.json") as f:
    yolo = json.load(f)
with open("evaluation/metrics/baseline_rtdetr_metrics.json") as f:
    rtdetr = json.load(f)

print(f"YOLOv8   mAP@50: {yolo['metrics']['map50']:.4f}")
print(f"RT-DETR  mAP@50: {rtdetr['metrics']['map50']:.4f}")
```

### Next: Occlusion Analysis (Step 3.3)

Once JSON files are generated, your friend can proceed with occlusion difficulty analysis without re-running inference.

---

## Environment Setup

```bash
pip install -r requirements.txt
```

Required environment variable:
```bash
export ROBOFLOW_API_KEY="your_api_key"
```

---

## Experiments

### Experiment 1: Freezing Ladder
Measures how the number of trainable parameters during fine-tuning affects performance when adapting COCO-pretrained detectors to our ingredient dataset.

- **Location:** `experiments/Experiment_1/`
- **Notebook:** `notebooks/E1_run_evaluate.ipynb`
- **Presets:** F0 (head only) → F3 (full fine-tune)

### Experiment 3: Internal Masking vs Occlusion Training
Tests whether internal channel masking during training can improve robustness to occlusions.

- **Location:** `experiments/Experiment_3/`
- **Notebook:** `experiments/Experiment_3/experiment3_run.ipynb`
- **Sessions:** 6 per model (clean baseline, occluded training, 4 masking locations)
- **Models:** YOLOv8m, RT-DETR-L
- **See:** [Experiment 3 README](experiments/Experiment_3/README.md)

---

## Project Roadmap

- [x] Step 1: Repository & Environment Setup
- [x] Step 2: Data Pipeline & Evaluation Foundations (FROZEN)
- [x] Step 3: Baseline Model Training & Evaluation
  - [x] 3.1: Protocol Freezing (run metadata JSONs)
  - [x] 3.2: Training & Baseline Performance (metrics + predictions JSONs)
  - [ ] 3.3: Occlusion Difficulty Analysis (slice by Easy/Medium/Hard)
  - [ ] 3.4: Confidence Threshold Sweep
- [x] Experiment 1: Freezing Ladder (YOLOv8m vs RT-DETR-L)
- [ ] Experiment 3: Internal Masking vs Occlusion Training
- [ ] Step 4: Performance Visualization
- [ ] Step 5: Hyperparameter Tuning (Phase 1)
- [ ] Step 6: Final Training (Phase 2 - 80/20 split)
- [ ] Step 7: Confidence Calibration Study
- [ ] Step 8: Attention Map Visualization
- [ ] Step 9: Recipe Generation Pipeline (Logic Gate + OpenAI API)
- [ ] Step 10: Final Report & Ethics Statement
