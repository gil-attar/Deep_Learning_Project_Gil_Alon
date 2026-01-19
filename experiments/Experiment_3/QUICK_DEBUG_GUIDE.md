# Quick Debug Guide for Experiment 3

## Step 1: Pull the code in Colab

```python
!cd /content/Deep_Learning_Gil_Alon && git pull
```

## Step 2: Add these 4 code blocks to your notebook

### BLOCK 1: Add AFTER cell-6 (after output_dir is defined, NOT after cell-9!)

**IMPORTANT**: This cell uses `output_dir` and `RUN_ID` which are defined in cell-6.
Place this IMMEDIATELY after cell-6, before any other imports.

```python
# ============================================================
# DEBUG IMPORTS
# ============================================================
from experiments.Experiment_3.debug_logger import (
    ExperimentDebugLogger,
    verify_data_yaml,
    verify_labels_exist,
    verify_occluded_test_data,
    get_environment_info,
    analyze_prediction_confidences
)

# Initialize debug logger
DEBUG_LOG_DIR = output_dir / "debug_logs"
debug_logger = ExperimentDebugLogger(DEBUG_LOG_DIR, RUN_ID)
print(f"Debug logs will be saved to: {DEBUG_LOG_DIR}")

# Log environment
env_info = get_environment_info()
debug_logger.log_environment(env_info)
print(f"GPU: {env_info.get('gpu_name', 'N/A')}")
```

### BLOCK 2: Add AFTER cell-17 (after creating data.yaml files)

```python
# ============================================================
# DATA VERIFICATION - RUN THIS BEFORE TRAINING!
# ============================================================
print("="*70)
print("DATA VERIFICATION")
print("="*70)

# 1. Clean training data
print("\n1. CLEAN TRAINING DATA:")
clean_info = verify_data_yaml("data/processed/data_clean.yaml")
print(f"   Path: {clean_info['train_path']}")
print(f"   Exists: {clean_info['train_exists']}")
print(f"   Images: {clean_info['num_train_images']}")

clean_labels = verify_labels_exist("data/raw/train/images")
print(f"   Labels: {clean_labels['total_boxes']} boxes")

# 2. Occluded training data (S2)
print("\n2. OCCLUDED TRAINING DATA (S2):")
occ_info = verify_data_yaml("data/processed/data_occ_train.yaml")
print(f"   Path: {occ_info['train_path']}")
print(f"   Exists: {occ_info['train_exists']}")
print(f"   Images: {occ_info['num_train_images']}")

occ_labels = verify_labels_exist("data/occluded_train_040/level_040/images")
print(f"   Labels dir exists: {occ_labels['labels_dir_exists']}")
print(f"   Total boxes: {occ_labels['total_boxes']}")
print(f"   Missing labels: {occ_labels['num_missing_labels']}")

if occ_labels['num_missing_labels'] > 0:
    print(f"   *** CRITICAL: {occ_labels['num_missing_labels']} images missing labels! ***")

# 3. Occluded test data
print("\n3. OCCLUDED TEST DATA:")
occ_test = verify_occluded_test_data("data/synthetic_occlusion/level_040")
print(f"   Exists: {occ_test['exists']}")
print(f"   Images: {occ_test.get('num_images', 0)}")

# Summary
print("\n" + "="*70)
issues = []
if occ_labels.get('num_missing_labels', 0) > 0:
    issues.append(f"Occluded training missing {occ_labels['num_missing_labels']} labels!")
if not occ_test.get('exists', False):
    issues.append("Occluded test data not found!")

if issues:
    print("*** ISSUES FOUND: ***")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("All checks passed!")
print("="*70)
```

### BLOCK 3: REPLACE cell-19 (train_session function)

Replace the entire `train_session` function with this (uses debug logging):

```python
def train_session(
    model_name: str,
    session_name: str,
    epochs: int,
    output_dir: Path,
    p_apply: float = 0.5,
    p_channels: float = 0.2
) -> dict:
    """Train a single session with debug logging."""
    import time
    session_config = get_session_config(session_name)
    run_name = f"{model_name}__{session_name}"
    run_dir = output_dir / run_name

    print(f"\n{'='*60}")
    print(f"TRAINING: {run_name}")
    print(f"{'='*60}")

    # Start debug logging
    debug_logger.start_session(model_name, session_name)

    # Check if already completed
    if is_session_complete(output_dir, model_name, session_name):
        print(f"Session already completed. Skipping.")
        debug_logger.end_session(success=True)
        return {"status": "skipped", "run_dir": str(run_dir)}

    # Select data.yaml
    if session_config['train_data'] == 'occluded':
        data_yaml = 'data/processed/data_occ_train.yaml'
    else:
        data_yaml = 'data/processed/data_clean.yaml'

    print(f"Data: {data_yaml}")

    # Log data config
    try:
        data_info = verify_data_yaml(data_yaml)
        debug_logger.log_data_config(
            data_yaml_path=data_yaml,
            train_images_dir=data_info['train_path'],
            sample_images=data_info['sample_train_images'],
            num_train_images=data_info['num_train_images'],
            num_val_images=data_info['num_val_images']
        )
    except Exception as e:
        debug_logger.log_error(f"Failed to verify data: {e}")

    # Load model
    model = get_model(model_name)

    # Setup masking if needed
    masking_manager = None
    mask_location = session_config['mask_location']

    if mask_location is not None:
        model_type = get_model_type(model_name)
        layer_prefixes = get_mask_prefixes(model_type, mask_location)

        print(f"Masking: {mask_location} -> {layer_prefixes}")

        # VERBOSE=True to see if masking fires!
        masking_manager = MaskingManager(model.model, p_apply, p_channels, verbose=True)
        num_hooks = masking_manager.add_masking_to_layers(layer_prefixes)
        print(f"Added {num_hooks} masking hooks")

        hooked_names = [hook.name for _, hook in masking_manager.hooks]
        debug_logger.log_masking_config(
            enabled=True, mask_location=mask_location,
            layer_prefixes=layer_prefixes, p_apply=p_apply,
            p_channels=p_channels, num_hooks_added=num_hooks,
            hooked_layer_names=hooked_names
        )

        if num_hooks == 0:
            debug_logger.log_warning(f"No hooks added for {mask_location}!")
    else:
        debug_logger.log_masking_config(
            enabled=False, mask_location=None, layer_prefixes=[],
            p_apply=p_apply, p_channels=p_channels,
            num_hooks_added=0, hooked_layer_names=[]
        )

    # Train
    abs_output_dir = output_dir.resolve()
    start_time = time.time()

    try:
        results = model.train(
            data=data_yaml, epochs=epochs, imgsz=IMGSZ, batch=BATCH,
            patience=PATIENCE, save=True, project=str(abs_output_dir),
            name=run_name, exist_ok=True, pretrained=True,
            optimizer='auto', verbose=True, seed=SEED
        )

        training_time = time.time() - start_time

        # Log masking stats
        if masking_manager:
            masking_manager.print_debug_summary()
            stats = masking_manager.get_detailed_stats()
            debug_logger.log_masking_summary(
                total_activations=stats['aggregate']['total_mask_applications'],
                hooked_layers_summary=stats['per_hook']
            )

        # Mark done
        run_dir = abs_output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "DONE").touch()

        if masking_manager:
            with open(run_dir / "masking_summary.json", 'w') as f:
                json.dump(masking_manager.get_summary(), f, indent=2)

        debug_logger.log_training_complete(str(run_dir / "weights" / "best.pt"), training_time)
        debug_logger.end_session(success=True)

        return {"status": "success", "run_dir": str(run_dir), "weights_path": str(run_dir / "weights" / "best.pt")}

    except Exception as e:
        debug_logger.log_error(f"Training failed: {e}", e)
        debug_logger.end_session(success=False)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "FAILED").write_text(str(e))
        return {"status": "failed", "error": str(e), "run_dir": str(run_dir)}

    finally:
        if masking_manager:
            masking_manager.remove_all_hooks()

print("Training function defined with debug logging!")
```

### BLOCK 4: Add at END of training loop (cell-21)

Add this at the end of cell-21, after the training loop completes:

```python
# Finalize debug log
debug_logger.finalize()

print(f"\n*** Debug logs saved to: {DEBUG_LOG_DIR} ***")
print("Check debug_log.txt for human-readable summary")
```

## Step 3: Run Smoke Test (EPOCHS=1)

1. Set `EPOCHS = 1` in config cell
2. Use a NEW RUN_ID (not your existing 50-epoch run): comment out the RUN_ID line
3. Run all cells
4. Check `debug_logs/debug_log.txt` on your Drive

## What to Look For in debug_log.txt

1. **Masking activations** - Should see `total_activations: XXXX` (not 0!) for S3-S6
2. **Data paths** - S2 should show path containing "occluded"
3. **Labels** - Should NOT say "missing labels"
4. **Warnings** - Look for any `*** WARNING ***` messages

## If Smoke Test Looks Good

1. Set `EPOCHS = 50`
2. Set a new RUN_ID or let it auto-generate
3. Run full experiment overnight
