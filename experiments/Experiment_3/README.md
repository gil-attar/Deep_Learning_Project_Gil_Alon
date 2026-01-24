# Experiment 3: Channel Masking vs Occlusion Training

## Research Question
**Can internal feature channel masking during training improve robustness to occlusions, as an alternative to training on occluded images?**

## Hypothesis
Randomly zeroing feature channels during training may force the network to learn redundant representations, improving robustness to partial occlusions at test time — similar to how Dropout improves generalization.

## Method

### Sessions (6 per model)
| Session | Training Data | Masking Location | Description                     |
|---------|---------------|------------------|---------------------------------|
| S1      | Clean         | None             | Baseline (no augmentation)      |
| S2      | Occluded (40%)| None             | Standard occlusion augmentation |
| S3      | Clean         | backbone_early   | Mask early backbone layers      |
| S4      | Clean         | backbone_late    | Mask late backbone layers       |
| S5      | Clean         | neck             | Mask neck/FPN layers            |
| S6      | Clean         | head             | Mask detection head             |

### Models
- **YOLOv8m** (CNN-based detector)
- **RT-DETR-L** (Transformer-based detector)

### Evaluation
All models evaluated on:
- **test_clean** — original test images
- **test_occluded** — same images with 40% synthetic occlusion

## Implementation Details

### Channel Masking Parameters
- **p_apply = 0.3**: 30% of batches have masking applied
- **p_channels = 0.1**: 10% of channels zeroed when masking is active
- **Training only**: Masking disabled during evaluation
**stronger masking may cause 0 detections**

### Layer Boundaries
**YOLOv8m:**
- backbone_early: `model.0` – `model.4`
- backbone_late: `model.5` – `model.9`
- neck: `model.10` – `model.21`
- head: `model.22`

**RT-DETR-L:**
- backbone_early: `model.0` – `model.5`
- backbone_late: `model.6` – `model.11`
- neck: `model.12` – `model.27`
- head: `model.28`

## Files

| File | Purpose |
|-------------------------|---------|
| `E3_run_evaluate.ipynb` | Main notebook (run in Google Colab) |
| `channel_masking.py` | Forward hook implementation |
| `mask_presets.py` | Layer definitions & session configs |
| `debug_logger.py` | Debug logging utilities |

### Completed Run Notebooks
| File | Purpose |
|------|---------|
| `E3_full_run/E3_run_evaluate_FINAL_RUN.ipynb` | Full experiment run with all sessions (S1-S6) |
| `E3_full_run/E3_run_evaluate_S2_DEBUGCHECK.ipynb` | S2-only re-run to verify domain shift behavior |

## Running the Experiment

1. Open `E3_run_evaluate.ipynb` in Google Colab
2. Configure parameters in the config cell:
   - `EPOCHS = 50` for full experiment
   - `DRY_RUN = False`
3. Run all cells

### Resume After Disconnect
1. Check `LATEST_E3_RUN_ID.txt` on your Google Drive
2. Set `PREVIOUS_RUN_ID = "E3_..."` in the config cell
3. Re-run — completed sessions are automatically skipped

## Output Structure
```
/content/drive/MyDrive/Colab_Outputs/Deep_Learning_Project_Gil_Alon/<RUN_ID>/
├── E3_runs/
│   ├── yolov8m__S1_clean_train/
│   │   ├── weights/best.pt
│   │   └── results.csv
│   ├── yolov8m__S2_occ_train/
│   │   └── ...
│   └── ...
├── _plots/
│   ├── comparison_f1_clean.png
│   └── comparison_f1_occluded.png
└── all_results.json
```

## Key Results

### Channel Masking Does NOT Improve Occlusion Robustness

**Key Findings:**
1. **Channel masking (S3-S6)** provides no improvement over baseline (S1) on occluded images
2. **Occlusion training (S2)** dramatically improves occluded performance (81% F1) but causes **catastrophic forgetting** on clean images (3% F1)
3. **Domain shift confirmed**: S2 models detect 15x more objects on occluded vs clean images

### Why S2 Shows Domain Shift
- Model trained exclusively on 40% occluded images
- Learns occlusion-specific features (black grid patterns)
- Completely fails on clean images (different visual domain)
- This is expected behavior, not a bug

## Conclusion
Channel masking at any network depth does not simulate occlusion robustness. True occlusion robustness requires exposure to occluded training data, but this creates a trade-off with clean image performance. A mixed training approach (clean + occluded) would be needed to maintain both capabilities.
