"""
Debug-enhanced cells for E3_run_evaluate.ipynb

COPY THESE CELLS INTO YOUR NOTEBOOK TO ENABLE COMPREHENSIVE DEBUGGING

This file contains the updated training and evaluation functions with
full debug logging that saves to Google Drive.
"""

# ============================================================
# CELL: Import debug logger and setup
# ADD THIS AFTER YOUR IMPORTS
# ============================================================
DEBUG_CELL_IMPORTS = '''
# Define output_dir first
output_dir = Path("runs/exp3").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

# Import debug logger and utilities
from experiments.Experiment_3.debug_logger import (
    ExperimentDebugLogger,
    verify_data_yaml,
    verify_labels_exist,
    verify_occluded_test_data,
    get_environment_info,
    analyze_prediction_confidences
)
from experiments.Experiment_3.channel_masking import MaskingManager
from PIL import Image
import glob

DEBUG_LOG_DIR = output_dir / "debug_logs"
debug_logger = ExperimentDebugLogger(DEBUG_LOG_DIR, RUN_ID)
print(f"Debug logs will be saved to: {DEBUG_LOG_DIR}")

env_info = get_environment_info()
debug_logger.log_environment(env_info)
print(f"GPU: {env_info.get('gpu_name', 'N/A')}")
'''


# ============================================================
# CELL: Verify data before training
# ADD THIS BEFORE TRAINING LOOP
# ============================================================
VERIFY_DATA_CELL = '''
print("="*70)
print("DATA VERIFICATION (CRITICAL - CHECK THESE BEFORE RUNNING FULL EXPERIMENT)")
print("="*70)

# Verify clean training data
print("\\n1. CLEAN TRAINING DATA:")
clean_info = verify_data_yaml("data/processed/data_clean.yaml")
print(f"   Path: {clean_info['train_path']}")
print(f"   Exists: {clean_info['train_exists']}")
print(f"   Images: {clean_info['num_train_images']}")
print(f"   Sample: {clean_info['sample_train_images'][:3]}")

# Verify labels for clean training data
clean_labels = verify_labels_exist("data/raw/train/images")
print(f"   Labels dir exists: {clean_labels['labels_dir_exists']}")
print(f"   Total boxes: {clean_labels['total_boxes']}")
print(f"   Missing labels: {clean_labels['num_missing_labels']}")

# Verify occluded training data
print("\\n2. OCCLUDED TRAINING DATA (S2):")
occ_info = verify_data_yaml("data/processed/data_occ_train.yaml")
print(f"   Path: {occ_info['train_path']}")
print(f"   Exists: {occ_info['train_exists']}")
print(f"   Images: {occ_info['num_train_images']}")
print(f"   Sample: {occ_info['sample_train_images'][:3]}")

# Verify labels for occluded training data
occ_labels = verify_labels_exist("data/occluded_train_040/level_040/images")
print(f"   Labels dir exists: {occ_labels['labels_dir_exists']}")
print(f"   Total boxes: {occ_labels['total_boxes']}")
print(f"   Missing labels: {occ_labels['num_missing_labels']}")

if occ_labels['num_missing_labels'] > 0:
    print(f"   *** CRITICAL: S2 has {occ_labels['num_missing_labels']} images without labels! ***")

# Verify occluded test data
print("\\n3. OCCLUDED TEST DATA:")
occ_test = verify_occluded_test_data("data/synthetic_occlusion/level_040")
print(f"   Path: {occ_test['path']}")
print(f"   Exists: {occ_test['exists']}")
print(f"   Images: {occ_test.get('num_images', 0)}")
print(f"   Labels: {occ_test.get('num_labels', 0)}")

# Check if occluded images LOOK occluded (display a few)
print("\\n4. VISUAL INSPECTION OF OCCLUDED TRAINING IMAGES:")
occ_train_images = glob.glob("data/occluded_train_040/level_040/images/*.jpg")[:3]
if occ_train_images:
    for img_path in occ_train_images:
        print(f"   Displaying: {img_path}")
        display(Image.open(img_path).resize((300, 300)))
else:
    print("   *** WARNING: No occluded training images found! ***")

# Compare clean vs occluded image of same file
print("\\n5. SIDE-BY-SIDE COMPARISON (Clean vs Occluded):")
if occ_info['sample_train_images']:
    occ_sample = Path(occ_info['sample_train_images'][0])
    clean_sample = Path("data/raw/train/images") / occ_sample.name
    if clean_sample.exists():
        print(f"   Clean: {clean_sample}")
        print(f"   Occluded: {occ_sample}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(Image.open(clean_sample))
        axes[0].set_title("Clean")
        axes[0].axis('off')
        axes[1].imshow(Image.open(occ_sample))
        axes[1].set_title("40% Occluded")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

print("\\n" + "="*70)
print("SANITY CHECK SUMMARY:")
print("="*70)
issues = []
if not clean_info['train_exists']:
    issues.append("Clean training images not found!")
if not occ_info['train_exists']:
    issues.append("Occluded training images not found!")
if occ_labels['num_missing_labels'] > 0:
    issues.append(f"Occluded training missing {occ_labels['num_missing_labels']} labels!")
if not occ_test['exists']:
    issues.append("Occluded test data not found!")

if issues:
    print("*** ISSUES FOUND - DO NOT PROCEED UNTIL FIXED: ***")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("All checks passed! Safe to proceed with training.")
print("="*70)
'''


# ============================================================
# CELL: Updated train_session function with CALLBACK-BASED masking
# REPLACE YOUR EXISTING train_session FUNCTION WITH THIS
#
# THE FIX: Ultralytics creates a DIFFERENT model object internally
# during training (trainer.model). Adding hooks to model.model
# before calling model.train() doesn't work because the hooks
# are on the wrong object!
#
# SOLUTION: Use Ultralytics callbacks to add hooks to trainer.model
# AFTER the trainer is initialized.
# ============================================================
TRAIN_SESSION_DEBUG = '''
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

    print(f"\\n{'='*60}")
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

        # CRITICAL CHECK: Are we using the right images?
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

        # Log config (hooks will be added when training starts)
        debug_logger.log_masking_config(
            enabled=True,
            mask_location=mask_location,
            layer_prefixes=layer_prefixes,
            p_apply=p_apply,
            p_channels=p_channels,
            num_hooks_added=-1,  # Will be set when callbacks fire
            hooked_layer_names=[]  # Will be populated when callbacks fire
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
            # Callbacks print summary automatically via on_train_end
            stats = masking_callbacks.get_stats()
            if stats:
                debug_logger.log_masking_summary(
                    total_activations=stats['aggregate']['total_mask_applications'],
                    hooked_layers_summary=stats['per_hook']
                )

                # Save detailed stats to file
                with open(run_dir / "masking_detailed_stats.json", 'w') as f:
                    json.dump(stats, f, indent=2)

                # Save summary
                manager = masking_callbacks.get_manager()
                if manager:
                    with open(run_dir / "masking_summary.json", 'w') as f:
                        json.dump(manager.get_summary(), f, indent=2)
            else:
                debug_logger.log_warning("Masking callbacks did not return stats!")

        # Mark as done
        (run_dir / "DONE").touch()

        debug_logger.log_training_complete(
            weights_path=str(run_dir / "weights" / "best.pt"),
            training_time_seconds=training_time
        )
        debug_logger.end_session(success=True)

        print(f"\\nTraining complete: {run_name}")
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

        print(f"\\nTraining FAILED: {run_name}")
        print(f"Error: {e}")

        run_dir = abs_output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "FAILED").write_text(str(e))

        return {"status": "failed", "error": str(e), "run_dir": str(run_dir)}

    finally:
        # Cleanup hooks
        if masking_callbacks:
            masking_callbacks.remove_hooks()
'''


# ============================================================
# CELL: Updated training loop
# REPLACE YOUR EXISTING TRAINING LOOP WITH THIS
# ============================================================
TRAINING_LOOP_DEBUG = '''
# Run all training sessions with debug logging
training_results = []
total_sessions = len(MODELS) * len(SESSIONS_TO_RUN)
current = 0

for model_name in MODELS:
    for session_name in SESSIONS_TO_RUN:
        current += 1
        print(f"\\n[{current}/{total_sessions}] {model_name} - {session_name}")

        result = train_session_debug(
            model_name=model_name,
            session_name=session_name,
            epochs=EPOCHS,
            output_dir=output_dir,
            debug_logger=debug_logger,
            p_apply=P_APPLY,
            p_channels=P_CHANNELS,
            verbose_masking=(EPOCHS <= 5)  # Only verbose for smoke tests
        )

        result['model'] = model_name
        result['session'] = session_name
        training_results.append(result)

# Finalize debug log
debug_logger.finalize()

# Summary
print("\\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
for r in training_results:
    status_icon = "OK" if r['status'] == 'success' else "SKIP" if r['status'] == 'skipped' else "FAIL"
    print(f"[{status_icon}] {r['model']}__{r['session']}")

print(f"\\nDebug logs saved to: {DEBUG_LOG_DIR}")
print(f"  - debug_log.json (for analysis)")
print(f"  - debug_log.txt (human readable)")

# Save results
with open(output_dir / "training_results.json", 'w') as f:
    json.dump(training_results, f, indent=2)
'''


# ============================================================
# CELL: Post-training debug analysis
# ADD THIS AFTER TRAINING COMPLETES
# ============================================================
POST_TRAINING_ANALYSIS = '''
print("="*70)
print("POST-TRAINING DEBUG ANALYSIS")
print("="*70)

# Load and analyze debug log
with open(DEBUG_LOG_DIR / "debug_log.json") as f:
    debug_data = json.load(f)

print("\\n1. MASKING ACTIVATION CHECK:")
print("-"*50)
for session_key, session in debug_data["sessions"].items():
    if session.get("masking_enabled"):
        total_activations = session.get("masking_final_summary", {}).get("total_activations", 0)
        if total_activations == 0:
            print(f"  [FAIL] {session_key}: Masking enabled but 0 activations!")
        else:
            print(f"  [OK]   {session_key}: {total_activations:,} mask activations")
    else:
        print(f"  [N/A]  {session_key}: No masking (baseline or S2)")

print("\\n2. DATA PATH CHECK:")
print("-"*50)
for session_key, session in debug_data["sessions"].items():
    train_dir = session.get("train_images_dir", "unknown")
    num_images = session.get("num_train_images", 0)
    is_s2 = "S2" in session_key

    if is_s2:
        if "occluded" in train_dir.lower() or "occ" in train_dir.lower():
            print(f"  [OK]   {session_key}: Using occluded data ({num_images} images)")
        else:
            print(f"  [FAIL] {session_key}: S2 should use occluded but using: {train_dir}")
    else:
        if "raw" in train_dir.lower():
            print(f"  [OK]   {session_key}: Using clean data ({num_images} images)")
        else:
            print(f"  [WARN] {session_key}: Using {train_dir}")

print("\\n3. WARNINGS:")
print("-"*50)
if debug_data.get("warnings"):
    for w in debug_data["warnings"]:
        print(f"  - [{w['time']}] {w['message']}")
else:
    print("  No warnings")

print("\\n4. ERRORS:")
print("-"*50)
if debug_data.get("errors"):
    for e in debug_data["errors"]:
        print(f"  - [{e['time']}] {e['message']}")
else:
    print("  No errors")

print("="*70)
'''


if __name__ == "__main__":
    print("="*70)
    print("E3 DEBUG CELLS")
    print("="*70)
    print()
    print("Copy these cells into your notebook to enable comprehensive debugging.")
    print()
    print("Cells to add:")
    print("  1. DEBUG_CELL_IMPORTS - After your imports")
    print("  2. VERIFY_DATA_CELL - Before training loop")
    print("  3. TRAIN_SESSION_DEBUG - Replace train_session function")
    print("  4. TRAINING_LOOP_DEBUG - Replace training loop")
    print("  5. POST_TRAINING_ANALYSIS - After training completes")
    print()
    print("This will log:")
    print("  - Data paths and sample images")
    print("  - Masking hook activations (are they firing?)")
    print("  - Training progress")
    print("  - Warnings for suspicious conditions")
    print()
    print("Logs are saved to Google Drive for crash safety.")
