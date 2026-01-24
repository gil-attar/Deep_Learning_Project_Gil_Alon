# Experiment 2: Training Duration (Epoch Budget Sweep)

## Research Question

**How does the number of training epochs affect model performance and generalization?**

We investigate when models enter overfitting/underfitting regimes by training with different epoch budgets.

## Method

- **Models:** YOLOv8m, RT-DETR-L
- **Freezing:** Fixed at F2 (Head + Neck + Late Backbone)
- **Epoch Budgets:** 5, 10, 20, 40, 80 epochs
- **No Early Stopping:** Models train for the full epoch budget to observe convergence behavior

## Files

| File | Description |
|------|-------------|
| `E2_run_evaluate.ipynb` | Main notebook to run in Google Colab |
| `runOneTest.py` | Training script for a single (model, epochs) configuration |
| `run_experiment2.sh` | Bash script to sweep all epoch budgets |

## How to Run

### 1. Open in Google Colab

Upload `E2_run_evaluate.ipynb` to Google Colab.

### 2. Configure

In cell 24, set:
```python
DRY_RUN = False  # Set to False for full sweep
FREEZE = "F2"    # Fixed freezing preset
EPOCHS_LIST = "5 10 20 40 80"  # Epoch budgets to test
```

### 3. Run All Cells

The notebook will:
1. Download dataset and build evaluation indices
2. Train both models at each epoch budget
3. Evaluate on test set using our custom evaluation pipeline
4. Generate comparison plots

## Outputs

Results are saved to Google Drive under:
```
/content/drive/MyDrive/Colab_Outputs/Deep_Learning_Project_Gil_Alon/E2_<timestamp>/
├── E2_runs/
│   ├── yolov8m/F2/
│   │   ├── E5/   # 5 epochs
│   │   ├── E10/  # 10 epochs
│   │   └── ...
│   └── rtdetr-l/F2/
│       ├── E5/
│       └── ...
└── _plots/
    ├── perf_vs_epochs.png
    └── time_vs_epochs.png
```

Each run directory contains:
- `run_manifest.json` - Configuration and parameter counts
- `weights/best.pt` - Best checkpoint
- `results.csv` - Per-epoch training metrics
- `eval/test/metrics.json` - Test set evaluation
- `eval/test/plots/` - Threshold sweep, confusion matrix, etc.

## Key Metrics

- **Best Validation mAP50:** Peak performance during training
- **Best Epoch:** When peak performance was achieved
- **Generalization Gap:** Difference between train and val loss
- **AUC of mAP50 curve:** Overall learning efficiency

## Expected Results

- Short training (5-10 epochs): Underfitting, model hasn't converged
- Optimal training (20-50 epochs): Best generalization
- Long training (80+ epochs): Potential overfitting, diminishing returns

## Notes

- Uses the same evaluation pipeline as Experiments 1 and 3 for consistency
- All runs use seed=42 for reproducibility
- Training is done WITHOUT early stopping to observe full convergence curves
