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
| Train | 1386   | 70%        |
| Valid | 198    | 10%        |
| Test  | 396    | 20%        |

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

## Environment Setup

```bash
pip install -r requirements.txt
```

Required environment variable:
```bash
export ROBOFLOW_API_KEY="your_api_key"
```

---

## Project Roadmap

- [x] Step 1: Repository & Environment Setup
- [x] Step 2: Data Pipeline & Evaluation Foundations (FROZEN)
- [ ] Step 3: Baseline Model Training (YOLOv8 & RT-DETR)
- [ ] Step 4: Evaluation Script
- [ ] Step 5: Occlusion Difficulty Analysis
- [ ] Step 6: Hyperparameter Tuning (Phase 1)
- [ ] Step 7: Final Training (Phase 2 - 80/20 split)
- [ ] Step 8: Confidence Calibration Study
- [ ] Step 9: Attention Map Visualization
- [ ] Step 10: Performance Graphs & Analysis
- [ ] Step 11: Recipe Generation (OpenAI API)
- [ ] Step 12: Final Report
