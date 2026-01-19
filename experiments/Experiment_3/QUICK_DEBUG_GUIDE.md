# Quick Debug Guide for Experiment 3

## ROOT CAUSE FOUND - MASKING FIX

The smoke test revealed that **masking hooks were never firing** (0 activations for all S3-S6 sessions).

### The Problem
When we do:
```python
model = YOLO("yolov8m.pt")
masking_manager = MaskingManager(model.model, ...)  # Add hooks here
model.train(...)  # But training uses a DIFFERENT model internally!
```

Ultralytics creates a **separate model object internally** (`trainer.model`) that is used for actual forward passes. Our hooks on `model.model` are never called because training doesn't go through that object.

### The Fix
We now use **callback-based masking** that adds hooks to `trainer.model` AFTER the trainer is initialized:

```python
from experiments.Experiment_3.channel_masking import MaskingCallbacks

model = YOLO("yolov8m.pt")
callbacks = MaskingCallbacks(
    layer_prefixes=["model.5", "model.6"],
    p_apply=0.5,
    p_channels=0.2
)
callbacks.register(model)  # Register callbacks
model.train(...)  # Hooks added via on_pretrain_routine_start callback
```

## Step 1: Pull the updated code in Colab

```python
!cd /content/Deep_Learning_Gil_Alon && git pull
```

## Step 2: Replace cell-21 with the FIXED training function

The training function has been completely rewritten to use callback-based masking.

Copy this ENTIRE cell to replace your existing `train_session_debug` function:

```python
# Import the new callback-based masking
from experiments.Experiment_3.channel_masking import MaskingCallbacks

def train_session_debug(
    model_name: str,
    session_name: str,
    epochs: int,
    output_dir: Path,
    debug_logger: ExperimentDebugLogger,
    p_apply: float = 0.5,
    p_channels: float = 0.2,
    verbose_masking: bool = True
) -> dict:
    """
    Train a single session with comprehensive debug logging.

    FIXED: Uses callback-based masking that hooks into trainer.model
    (the actual model used for forward passes during training).
    """
    import time
    session_config = get_session_config(session_name)
    run_name = f"{model_name}__{session_name}"
    run_dir = output_dir / run_name

    print(f"\n{'='*60}")
    print(f"TRAINING: {run_name}")
    print(f"{'='*60}")
    print(f"Description: {session_config['description']}")
    print(f"Epochs: {epochs}")

    # Start debug logging
    debug_logger.start_session(model_name, session_name)

    # Check if already completed
    if is_session_complete(output_dir, model_name, session_name):
        print(f"Session already completed. Skipping.")
        debug_logger.log_warning("Session skipped - already complete")
        debug_logger.end_session(success=True)
        return {"status": "skipped", "run_dir": str(run_dir)}

    # Select data.yaml based on session
    if session_config['train_data'] == 'occluded':
        data_yaml = 'data/processed/data_occ_train.yaml'
    else:
        data_yaml = 'data/processed/data_clean.yaml'

    print(f"Data: {data_yaml}")

    # Log data configuration
    try:
        data_info = verify_data_yaml(data_yaml)
        debug_logger.log_data_config(
            data_yaml_path=data_yaml,
            train_images_dir=data_info['train_path'],
            sample_images=data_info['sample_train_images'],
            num_train_images=data_info['num_train_images'],
            num_val_images=data_info['num_val_images']
        )

        if session_config['train_data'] == 'occluded':
            if 'occluded' not in data_info['train_path'].lower() and 'occ' not in data_info['train_path'].lower():
                debug_logger.log_warning(f"S2 should use occluded data but train_path is: {data_info['train_path']}")
    except Exception as e:
        debug_logger.log_error(f"Failed to verify data.yaml: {e}")

    # Load model
    model = get_model(model_name)

    # Log model architecture (first time only)
    if session_name == "S1_clean_train":
        debug_logger.log_model_architecture_check(model_name, model.model)

    # Setup CALLBACK-BASED masking if needed
    # This is the FIX - hooks are added via callbacks to trainer.model
    masking_callbacks = None
    mask_location = session_config['mask_location']

    if mask_location is not None:
        model_type = get_model_type(model_name)
        layer_prefixes = get_mask_prefixes(model_type, mask_location)

        print(f"Masking: {mask_location} -> layers {layer_prefixes}")
        print(f"Masking params: p_apply={p_apply}, p_channels={p_channels}")
        print(f"Using CALLBACK-BASED masking (hooks trainer.model)")

        # Create callbacks and register with model
        masking_callbacks = MaskingCallbacks(
            layer_prefixes=layer_prefixes,
            p_apply=p_apply,
            p_channels=p_channels,
            verbose=verbose_masking
        )
        masking_callbacks.register(model)

        debug_logger.log_masking_config(
            enabled=True,
            mask_location=mask_location,
            layer_prefixes=layer_prefixes,
            p_apply=p_apply,
            p_channels=p_channels,
            num_hooks_added=-1,
            hooked_layer_names=[]
        )
    else:
        print("Masking: None")
        debug_logger.log_masking_config(
            enabled=False,
            mask_location=None,
            layer_prefixes=[],
            p_apply=p_apply,
            p_channels=p_channels,
            num_hooks_added=0,
            hooked_layer_names=[]
        )

    # Train with timing
    abs_output_dir = output_dir.resolve()
    start_time = time.time()

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            save=True,
            project=str(abs_output_dir),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=SEED
        )

        training_time = time.time() - start_time

        # Get masking stats from callbacks
        run_dir = abs_output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if masking_callbacks:
            stats = masking_callbacks.get_stats()
            if stats:
                debug_logger.log_masking_summary(
                    total_activations=stats['aggregate']['total_mask_applications'],
                    hooked_layers_summary=stats['per_hook']
                )

                with open(run_dir / "masking_detailed_stats.json", 'w') as f:
                    json.dump(stats, f, indent=2)

                manager = masking_callbacks.get_manager()
                if manager:
                    with open(run_dir / "masking_summary.json", 'w') as f:
                        json.dump(manager.get_summary(), f, indent=2)
            else:
                debug_logger.log_warning("Masking callbacks did not return stats!")

        (run_dir / "DONE").touch()

        debug_logger.log_training_complete(
            weights_path=str(run_dir / "weights" / "best.pt"),
            training_time_seconds=training_time
        )
        debug_logger.end_session(success=True)

        print(f"\nTraining complete: {run_name}")
        print(f"Time: {training_time:.1f}s")
        print(f"Saved to: {run_dir}")

        return {
            "status": "success",
            "run_dir": str(run_dir),
            "weights_path": str(run_dir / "weights" / "best.pt")
        }

    except Exception as e:
        debug_logger.log_error(f"Training failed: {e}", e)
        debug_logger.end_session(success=False)

        print(f"\nTraining FAILED: {run_name}")
        print(f"Error: {e}")

        run_dir = abs_output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "FAILED").write_text(str(e))

        return {"status": "failed", "error": str(e), "run_dir": str(run_dir)}

    finally:
        if masking_callbacks:
            masking_callbacks.remove_hooks()

print("Training function defined (CALLBACK-BASED masking)!")
```

## Step 3: Run Another Smoke Test (EPOCHS=1)

1. Set `EPOCHS = 1` in config cell
2. Use a NEW RUN_ID: `RUN_ID = "E3_SMOKE_TEST_FIX"`
3. Delete the old E3_SMOKE_TEST folder on Drive (or rename it)
4. Run all cells
5. Look for this output during S3-S6 training:
   ```
   [MaskingCallbacks] on_pretrain_routine_start
   [MaskingCallbacks] trainer.model type: <class '...'>
   [MaskingCallbacks] Added XX masking hooks to trainer.model
   ```

## What Success Looks Like

In the POST-TRAINING DEBUG ANALYSIS, you should now see:

```
1. MASKING ACTIVATION CHECK:
[N/A]  yolov8m__S1_clean_train: No masking (baseline or S2)
[N/A]  yolov8m__S2_occ_train: No masking (baseline or S2)
[OK]   yolov8m__S3_mask_backbone_early: 12,345 mask activations
[OK]   yolov8m__S4_mask_backbone_late: 23,456 mask activations
[OK]   yolov8m__S5_mask_neck: 34,567 mask activations
[OK]   yolov8m__S6_mask_head: 5,678 mask activations
```

Instead of the previous failure:
```
[FAIL] yolov8m__S3_mask_backbone_early: Masking enabled but 0 activations!
```

## If Smoke Test Shows Masking Working

1. Set `EPOCHS = 50`
2. Set a new RUN_ID for the full run (or let it auto-generate with timestamp)
3. Run overnight

## Files Changed

1. `experiments/Experiment_3/channel_masking.py` - Added `MaskingCallbacks` class
2. `experiments/Experiment_3/E3_debug_cells.py` - Updated `train_session_debug` to use callbacks
3. `experiments/Experiment_3/QUICK_DEBUG_GUIDE.md` - This file

## Technical Details

The key insight is that Ultralytics' training pipeline creates a new model object:
- `YOLO("model.pt").model` → The model you get when loading
- `trainer.model` → The model actually used during training forward passes

These can be different objects, especially with features like:
- Distributed Data Parallel (DDP)
- Model compilation
- Internal model copying

By using the `on_pretrain_routine_start` callback, we add hooks to `trainer.model` which is guaranteed to be the model used for forward passes.
