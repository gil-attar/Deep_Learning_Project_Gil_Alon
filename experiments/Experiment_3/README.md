# Experiment 3: Internal Masking vs Occlusion Training

## Goal
Test whether **internal channel masking during training** can improve robustness to occlusions, as an alternative to training on occluded images.

## Hypothesis
Randomly zeroing feature channels during training may force the network to learn redundant representations, improving robustness to partial occlusions at test time — similar to how Dropout improves generalization.

## Experimental Design

### Sessions (6 per model)
| Session | Training Data | Masking Location | Description |
|---------|---------------|------------------|-------------|
| S1 | Clean | None | Baseline (no augmentation) |
| S2 | Occluded (40%) | None | Standard practice (train on occluded images) |
| S3 | Clean | backbone_early | Mask early backbone layers |
| S4 | Clean | backbone_late | Mask late backbone layers |
| S5 | Clean | neck | Mask neck/FPN layers |
| S6 | Clean | head | Mask detection head |

### Models
- **YOLOv8m** (CNN-based detector)
- **RT-DETR-L** (Transformer-based detector)

### Evaluation
All models tested on:
- `test_clean` — original test images
- `test_occluded` — same test images with 40% synthetic occlusion

## Implementation Details

### Channel Masking
- **Type**: Channel-wise zeroing (not element-wise dropout)
- **p_apply = 0.5**: 50% of batches have masking applied
- **p_channels = 0.2**: 20% of channels zeroed when masking is active
- **Training only**: Masking disabled during evaluation (`model.eval()`)

### Layer Boundaries
Based on architecture analysis from Experiment 1:

**YOLOv8:**
- backbone_early: `model.0` – `model.4`
- backbone_late: `model.5` – `model.9`
- neck: `model.10` – `model.21`
- head: `model.22`

**RT-DETR:**
- backbone_early: `model.0` – `model.5`
- backbone_late: `model.6` – `model.11`
- neck: `model.12` – `model.27`
- head: `model.28`

### Dataset Consistency
All sessions use:
- Same train/val/test split
- Same occluded images (seed=42)
- Same training seed (42)

## Files

| File | Purpose |
|------|---------|
| `experiment3_run.ipynb` | Main notebook (run in Colab) |
| `channel_masking.py` | Forward hook implementation |
| `mask_presets.py` | Layer definitions & session configs |

## Running the Experiment

### Smoke Test (1 epoch)
1. Open `experiment3_run.ipynb` in Google Colab
2. Keep `EPOCHS = 1` in config cell
3. Run all cells

### Full Experiment
1. Open `experiment3_run.ipynb` in Google Colab
2. Set `EPOCHS = 50` in config cell
3. Run all cells (takes several hours)

### Resume After Disconnect
1. Check `LATEST_E3_RUN_ID.txt` on your Google Drive for the RUN_ID
2. Uncomment and set `RUN_ID = "E3_..."` in the config cell
3. Re-run all cells — completed sessions are automatically skipped

### RUN_ID Behavior
| Scenario | RUN_ID | Behavior |
|----------|--------|----------|
| Smoke test (EPOCHS=1) | `E3_SMOKE_TEST` (fixed) | Separate folder, won't interfere with full run |
| Full run (EPOCHS=50) | `E3_20240117_...` (timestamp) | New unique folder each time |
| Resume (manual) | Whatever you set | Looks for existing DONE markers |

## Output Structure
```
/content/drive/MyDrive/Colab_Outputs/Deep_Learning_Project_Gil_Alon/<RUN_ID>/
├── RUN_ID.txt                    # Contains RUN_ID for resume
├── E3_runs/
│   ├── yolov8m__S1_clean_train/
│   │   ├── weights/best.pt
│   │   ├── DONE
│   │   └── results.csv
│   ├── yolov8m__S2_occ_train/
│   │   └── ...
│   ├── evaluations/
│   │   ├── yolov8m__S1_clean_train__test_clean/
│   │   │   └── metrics.json
│   │   └── ...
│   ├── all_metrics.json
│   ├── summary_metrics.csv
│   └── plots/
│       ├── comparison_f1_clean.png
│       └── comparison_f1_occluded.png
└── LATEST_E3_RUN_ID.txt          # Easy lookup for resume
```

## Key Metrics
- **F1 Score** at optimal confidence threshold
- **Precision / Recall**
- **Count MAE** (ingredient counting error)

## Expected Insights
1. Does masking at any location match or exceed training on occluded images?
2. Which masking location (backbone vs neck vs head) is most effective?
3. Do CNN (YOLO) and Transformer (RT-DETR) respond differently to masking?
