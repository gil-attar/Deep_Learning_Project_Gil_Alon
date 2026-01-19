"""
Test script to debug why masking hooks aren't firing with Ultralytics.

Run this locally to diagnose the issue.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.Experiment_3.channel_masking import MaskingManager


def test_basic_hook():
    """Test that hooks work on a simple model."""
    print("\n" + "="*60)
    print("TEST 1: Basic hook on simple model")
    print("="*60)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return x

    model = SimpleModel()
    model.train()

    manager = MaskingManager(model, p_apply=1.0, p_channels=0.3, verbose=True)
    num_hooks = manager.add_masking_to_layers(["conv1", "conv2"])
    print(f"Added {num_hooks} hooks")

    # Run forward pass
    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    manager.print_debug_summary()
    manager.remove_all_hooks()

    return manager.get_mask_count() > 0


def test_yolo_direct_forward():
    """Test hooks on YOLO with direct forward pass (not .train())."""
    print("\n" + "="*60)
    print("TEST 2: YOLO direct forward pass")
    print("="*60)

    # Load YOLO
    model = YOLO("yolov8n.pt")

    # Get the inner model
    inner_model = model.model
    print(f"Inner model type: {type(inner_model)}")
    print(f"Inner model training mode: {inner_model.training}")

    # Set to training mode
    inner_model.train()
    print(f"After .train(): {inner_model.training}")

    # Add hooks to backbone layers
    manager = MaskingManager(inner_model, p_apply=1.0, p_channels=0.2, verbose=True)
    num_hooks = manager.add_masking_to_layers(["model.0", "model.1", "model.2"])
    print(f"Added {num_hooks} hooks")

    # Run direct forward pass on inner model
    x = torch.randn(1, 3, 640, 640)
    print("Running forward pass...")
    with torch.no_grad():
        y = inner_model(x)

    manager.print_debug_summary()

    result = manager.get_mask_count() > 0
    manager.remove_all_hooks()
    return result


def test_yolo_predict():
    """Test hooks with YOLO predict (inference)."""
    print("\n" + "="*60)
    print("TEST 3: YOLO predict (inference mode)")
    print("="*60)

    model = YOLO("yolov8n.pt")
    inner_model = model.model

    # Note: predict uses eval mode, so hooks should skip due to not training
    manager = MaskingManager(inner_model, p_apply=1.0, p_channels=0.2, verbose=True)
    num_hooks = manager.add_masking_to_layers(["model.0", "model.1"])
    print(f"Added {num_hooks} hooks")

    # Create dummy image
    import numpy as np
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run predict
    print("Running predict...")
    results = model.predict(dummy_img, verbose=False)

    manager.print_debug_summary()

    # Should have calls but skipped due to not training
    stats = manager.get_detailed_stats()
    print(f"\nExpected: call_count > 0, skip_not_training > 0")
    print(f"Actual: call_count={stats['aggregate']['total_calls']}, skip_not_training={stats['aggregate']['skip_not_training']}")

    manager.remove_all_hooks()
    return stats['aggregate']['total_calls'] > 0


def test_yolo_train_callback():
    """Test adding hooks via Ultralytics callback system."""
    print("\n" + "="*60)
    print("TEST 4: YOLO train with callback hooks")
    print("="*60)

    model = YOLO("yolov8n.pt")

    # We need to add hooks AFTER the trainer sets up the model
    # Use on_train_start callback

    manager_holder = {"manager": None}

    def on_train_start(trainer):
        """Add hooks when training starts."""
        print("\n[CALLBACK] on_train_start called!")
        print(f"[CALLBACK] trainer.model type: {type(trainer.model)}")
        print(f"[CALLBACK] trainer.model.training: {trainer.model.training}")

        # The trainer.model IS the model that will be used for forward passes
        manager = MaskingManager(trainer.model, p_apply=1.0, p_channels=0.2, verbose=True)
        num_hooks = manager.add_masking_to_layers(["model.0", "model.1", "model.2"])
        print(f"[CALLBACK] Added {num_hooks} hooks to trainer.model")

        manager_holder["manager"] = manager

    def on_train_epoch_end(trainer):
        """Check hook stats at end of epoch."""
        print("\n[CALLBACK] on_train_epoch_end called!")
        if manager_holder["manager"]:
            manager_holder["manager"].print_debug_summary()

    # Add callbacks
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Check if data exists
    data_yaml = Path("data/processed/data_clean.yaml")
    if not data_yaml.exists():
        print(f"Skipping train test - {data_yaml} not found")
        print("Run this in Colab or create test data first")
        return None

    # Train for 1 epoch
    print("\nStarting training for 1 epoch...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=320,  # Small for speed
            batch=4,
            device='cpu',  # CPU for testing
            verbose=True
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return False

    if manager_holder["manager"]:
        manager_holder["manager"].print_debug_summary()
        result = manager_holder["manager"].get_mask_count() > 0
        manager_holder["manager"].remove_all_hooks()
        return result

    return False


def investigate_model_structure():
    """Print model structure to understand layer naming."""
    print("\n" + "="*60)
    print("INVESTIGATION: YOLO model structure")
    print("="*60)

    model = YOLO("yolov8n.pt")
    inner = model.model

    print("\nTop-level attributes of YOLO object:")
    for attr in dir(model):
        if not attr.startswith('_'):
            val = getattr(model, attr, None)
            if isinstance(val, nn.Module):
                print(f"  {attr}: {type(val)}")

    print("\nInner model (model.model) named_modules (first 20):")
    for i, (name, module) in enumerate(inner.named_modules()):
        if i < 20:
            print(f"  {name}: {type(module).__name__}")

    print("\nLooking for 'model' prefix in layer names:")
    model_layers = [(n, type(m).__name__) for n, m in inner.named_modules() if n.startswith("model.")]
    print(f"  Found {len(model_layers)} layers with 'model.' prefix")
    for name, mtype in model_layers[:10]:
        print(f"    {name}: {mtype}")


if __name__ == "__main__":
    print("="*60)
    print("MASKING HOOK DEBUGGING TESTS")
    print("="*60)

    # Run tests
    results = {}

    results["test_basic_hook"] = test_basic_hook()
    results["test_yolo_direct_forward"] = test_yolo_direct_forward()
    results["test_yolo_predict"] = test_yolo_predict()

    investigate_model_structure()

    # This test requires training data
    # results["test_yolo_train_callback"] = test_yolo_train_callback()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        status = "PASS" if result else ("SKIP" if result is None else "FAIL")
        print(f"  {test_name}: {status}")
