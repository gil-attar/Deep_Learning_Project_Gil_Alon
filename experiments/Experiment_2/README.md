# Experiment 2: Training Duration (Epoch Budget Sweep)

## Research Question

**How does the number of training epochs affect model detection performance and generalization?**

## Hypothesis
Under a fixed transfer-learning configuration, CNN-based detectors are expected to achieve a large fraction of their final performance with relatively few epochs, whereas transformer-based detectors may require substantially larger training budgets to reach comparable performance due to optimization and representation adaptation demands.

## Method

- **Models:** YOLOv8m, RT-DETR-L
- **Freezing:** Fixed at F2 (Head + Neck + Late Backbone) from E1
- **Epoch Budgets:** 5, 10, 20, 40, 80 epochs. This yields 2 models × 5 budgets = 10 runs.
- **No Early Stopping:** Models train for the full epoch budget to observe convergence behavior

### Evaluation
The unified project evaluation protocol is used:
- Threshold selection uses the validation split
- Final reporting uses the held-out test split
- Primary metrics: mAP@0.5 and F1 (IoU-based matching)

## Files

| File | Purpose |
|------|---------|
| `E2_run_evaluate.ipynb` | Main Colab notebook: runs the epoch-budget sweep, evaluates, aggregates, and saves plots |
| `run_experiment2.sh` (or equivalent) | Orchestrates the 10-run sweep (2 models × 5 epoch budgets) |
| `runOneTest.py` (shared) | Single-configuration runner: train → export predictions → evaluate |
| `eval_contract.json` / `RUN_CONTRACT.md` (shared) | Defines required artifacts and evaluation compatibility |

## Run Contract

### Colab-First Orchestration
Experiment 2 is executed end-to-end via a Colab notebook with Drive-backed persistence:
1. Mount Google Drive for persistent storage
2. Download the dataset and pretrained weights into the runtime
3. Link the experiment output directory to a Drive-backed folder
4. Run one configuration per (model × epoch budget)
5. Aggregate results and export combined plots/summaries to Drive

### Run Sequence
1. Open `E2_run_evaluate.ipynb` in Google Colab
2. Set the configuration cell (names may differ slightly, but intent is identical):
   - `EPOCH_LIST = [5, 10, 20, 40, 80]`
   - `RUN_ID = ""` for a fresh run (or set to an existing `E2_...` to resume)
   - `DRY_RUN = True` for a quick smoke test (then set `False` for the full run)
3. Run all notebook cells

### Resume After Disconnect
1. Locate `LATEST_E2_RUN_ID.txt` on Google Drive
2. Set `RUN_ID = "E2_..."` in the notebook configuration cell
3. Re-run; completed budgets should be skipped based on run completeness checks

## Output Structure
Drive-backed output root:
```
/content/drive/MyDrive/Colab_Outputs/Deep_Learning_Project_Gil_Alon/<RUN_ID>/
└── E2_runs/
    ├── yolov8m/
    │   ├── epochs_5/
    │   ├── epochs_10/
    │   ├── epochs_20/
    │   ├── epochs_40/
    │   └── epochs_80/
    ├── rtdetr-l/
    │   ├── epochs_5/
    │   ├── epochs_10/
    │   ├── epochs_20/
    │   ├── epochs_40/
    │   └── epochs_80/
    └── _plots/                  
```

Each run directory contains:
- `run_manifest.json` — Run configuration + parameter counts
- `weights/best.pt` — Best checkpoint (per the selected validation criterion)
- `results.csv` — Per-epoch training log (if exported by the trainer)
- `eval/test/metrics.json` — Test-set metrics
- `eval/test/plots/` — Evaluation plots (e.g., threshold sweep, confusion matrix), if plot generation is enabled

## Key Metrics

- **Best Validation mAP50:** Peak performance during training
- **Best Epoch:** When peak performance was achieved
- **Generalization Gap:** Difference between train and val loss
- **AUC of mAP50 curve:** Overall learning efficiency

# Key Results (Summary)
- YOLOv8m converges quickly: performance improves rapidly at small epoch budgets, with a large fraction of final performance typically achieved by 10–20 epochs.
- RT-DETR-L is more budget-sensitive: performance is limited at small budgets and improves substantially only after extended training, with major gains typically appearing between 20–40 epochs and continuing up to 80 epochs.

## Conclusion
Under our specific fine-tuning regime and experimental setup, we found CNNs to likely be more training-budget efficient, reaching good performance in few epochs due to convolutional inductive bias, and thus are preferable to transformers when the epoch budget is tight. However, this conclusion is limited to our particular models and use case, dataset, and hyperparameter choices. Broader validation across additional architectures, datasets, training schedules and tasks is required before generalizing this trend.
