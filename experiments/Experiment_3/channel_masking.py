"""
Channel Masking Implementation for Experiment 3

This module provides forward hooks that apply channel masking to specific
layers during training. Masking is ONLY applied during training, not inference.

Channel masking zeros out entire feature channels (not individual values),
simulating information loss similar to what occlusions cause.

IMPORTANT: For Ultralytics models, use MaskingCallbacks instead of directly
adding hooks to model.model. The trainer creates a different model object
internally, so hooks must be added via callbacks.
"""

import torch
import torch.nn as nn
from typing import List, Callable, Optional, Dict, Any
import random


class ChannelMaskingHook:
    """
    Forward hook that applies channel masking during training.

    When attached to a layer, it randomly zeros out a fraction of channels
    in the output tensor. This simulates occlusion-like information loss
    at the feature level.

    Masking is ONLY applied:
    - During training (model.training == True)
    - With probability p_apply per forward pass
    """

    def __init__(
        self,
        p_apply: float = 0.5,
        p_channels: float = 0.2,
        name: str = "unnamed",
        verbose: bool = False
    ):
        """
        Args:
            p_apply: Probability of applying masking on each forward pass (0-1)
            p_channels: Fraction of channels to zero when masking is applied (0-1)
            name: Layer name for debugging/logging
            verbose: If True, print when masking is applied (for debugging)
        """
        self.p_apply = p_apply
        self.p_channels = p_channels
        self.name = name
        self.verbose = verbose

        # Detailed tracking for debugging
        self.mask_count = 0           # Times masking was actually applied
        self.call_count = 0           # Total forward pass calls
        self.skip_not_training = 0    # Skipped because model.training=False
        self.skip_probability = 0     # Skipped due to probability
        self.skip_wrong_type = 0      # Skipped due to non-tensor output
        self.skip_wrong_dims = 0      # Skipped due to wrong dimensions
        self.channels_masked_total = 0  # Total channels masked across all calls

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward hook function.

        Args:
            module: The layer this hook is attached to
            input: Layer input (unused)
            output: Layer output tensor to potentially mask

        Returns:
            Possibly masked output tensor
        """
        self.call_count += 1

        # Only mask during training
        if not module.training:
            self.skip_not_training += 1
            return output

        # Only mask with probability p_apply
        if random.random() > self.p_apply:
            self.skip_probability += 1
            return output

        # Handle different output types
        if not isinstance(output, torch.Tensor):
            # Some layers return tuples or other types - skip masking
            self.skip_wrong_type += 1
            return output

        # Need at least 4 dims for channel masking: [B, C, H, W]
        if output.dim() < 4:
            self.skip_wrong_dims += 1
            return output

        # Apply channel masking
        num_channels = output.shape[1]
        num_to_mask = max(1, int(num_channels * self.p_channels))

        # Create channel mask (1 = keep, 0 = mask)
        mask = torch.ones(num_channels, device=output.device, dtype=output.dtype)
        mask_indices = torch.randperm(num_channels)[:num_to_mask]
        mask[mask_indices] = 0

        # Reshape for broadcasting: [1, C, 1, 1]
        mask = mask.view(1, -1, 1, 1)

        # Apply mask (no rescaling - we want information loss)
        self.mask_count += 1
        self.channels_masked_total += num_to_mask

        # Log first few applications for debugging
        if self.verbose and self.mask_count <= 3:
            print(f"[MASK] {self.name}: masked {num_to_mask}/{num_channels} channels "
                  f"(output shape: {output.shape})")

        return output * mask

    def get_stats(self) -> dict:
        """Get detailed statistics for debugging."""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "mask_count": self.mask_count,
            "mask_rate": self.mask_count / max(1, self.call_count),
            "skip_not_training": self.skip_not_training,
            "skip_probability": self.skip_probability,
            "skip_wrong_type": self.skip_wrong_type,
            "skip_wrong_dims": self.skip_wrong_dims,
            "channels_masked_total": self.channels_masked_total
        }


class MaskingManager:
    """
    Manages channel masking hooks for a model.

    Provides methods to add/remove masking hooks to specific layers
    identified by their name prefixes.
    """

    def __init__(
        self,
        model: nn.Module,
        p_apply: float = 0.5,
        p_channels: float = 0.2,
        verbose: bool = False
    ):
        """
        Args:
            model: The model to add masking to
            p_apply: Probability of applying masking per batch
            p_channels: Fraction of channels to zero when masking
            verbose: If True, print debugging info when masking fires
        """
        self.model = model
        self.p_apply = p_apply
        self.p_channels = p_channels
        self.verbose = verbose
        self.hooks = []  # List of (handle, hook_object) tuples
        self.enabled = False

    def add_masking_to_layers(self, layer_prefixes: List[str]) -> int:
        """
        Add masking hooks to layers matching the given prefixes.

        Args:
            layer_prefixes: List of layer name prefixes (e.g., ["model.5", "model.6"])

        Returns:
            Number of hooks added
        """
        prefixes = tuple(layer_prefixes)
        hooks_added = 0

        for name, module in self.model.named_modules():
            # Check if this module's name starts with any of our prefixes
            if name.startswith(prefixes):
                # Only add to modules that produce tensor outputs (Conv, etc.)
                # Skip container modules like Sequential
                if self._is_hookable_module(module):
                    hook = ChannelMaskingHook(
                        p_apply=self.p_apply,
                        p_channels=self.p_channels,
                        name=name,
                        verbose=self.verbose
                    )
                    handle = module.register_forward_hook(hook)
                    self.hooks.append((handle, hook))
                    hooks_added += 1
                    if self.verbose:
                        print(f"  [HOOK] Added masking hook to: {name} ({type(module).__name__})")

        self.enabled = hooks_added > 0
        return hooks_added

    def _is_hookable_module(self, module: nn.Module) -> bool:
        """Check if a module is suitable for hooking (produces tensor output)."""
        hookable_types = (
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.SiLU,
            nn.ReLU,
            nn.LeakyReLU,
            nn.GELU,
            # Add more types as needed
        )
        return isinstance(module, hookable_types)

    def remove_all_hooks(self):
        """Remove all masking hooks from the model."""
        for handle, _ in self.hooks:
            handle.remove()
        self.hooks = []
        self.enabled = False

    def get_mask_count(self) -> int:
        """Get total number of times masking was applied across all hooks."""
        return sum(hook.mask_count for _, hook in self.hooks)

    def get_summary(self) -> dict:
        """Get summary of masking configuration."""
        return {
            "enabled": self.enabled,
            "num_hooks": len(self.hooks),
            "p_apply": self.p_apply,
            "p_channels": self.p_channels,
            "total_mask_applications": self.get_mask_count(),
            "hooked_layers": [hook.name for _, hook in self.hooks]
        }

    def get_detailed_stats(self) -> dict:
        """Get detailed per-hook statistics for debugging."""
        hook_stats = [hook.get_stats() for _, hook in self.hooks]

        # Aggregate stats
        total_calls = sum(s["call_count"] for s in hook_stats)
        total_masks = sum(s["mask_count"] for s in hook_stats)
        total_skip_not_training = sum(s["skip_not_training"] for s in hook_stats)
        total_skip_probability = sum(s["skip_probability"] for s in hook_stats)
        total_skip_wrong_type = sum(s["skip_wrong_type"] for s in hook_stats)
        total_skip_wrong_dims = sum(s["skip_wrong_dims"] for s in hook_stats)

        return {
            "enabled": self.enabled,
            "num_hooks": len(self.hooks),
            "config": {
                "p_apply": self.p_apply,
                "p_channels": self.p_channels
            },
            "aggregate": {
                "total_calls": total_calls,
                "total_mask_applications": total_masks,
                "effective_mask_rate": total_masks / max(1, total_calls),
                "skip_not_training": total_skip_not_training,
                "skip_probability": total_skip_probability,
                "skip_wrong_type": total_skip_wrong_type,
                "skip_wrong_dims": total_skip_wrong_dims
            },
            "per_hook": hook_stats
        }

    def print_debug_summary(self):
        """Print a human-readable debug summary."""
        stats = self.get_detailed_stats()

        print("\n" + "="*60)
        print("MASKING DEBUG SUMMARY")
        print("="*60)
        print(f"Enabled: {stats['enabled']}")
        print(f"Hooks: {stats['num_hooks']}")
        print(f"Config: p_apply={stats['config']['p_apply']}, p_channels={stats['config']['p_channels']}")
        print()
        print("Aggregate Stats:")
        agg = stats['aggregate']
        print(f"  Total forward calls: {agg['total_calls']}")
        print(f"  Masking applied: {agg['total_mask_applications']} ({agg['effective_mask_rate']:.2%})")
        print(f"  Skipped (not training): {agg['skip_not_training']}")
        print(f"  Skipped (probability): {agg['skip_probability']}")
        print(f"  Skipped (wrong type): {agg['skip_wrong_type']}")
        print(f"  Skipped (wrong dims): {agg['skip_wrong_dims']}")

        if agg['total_mask_applications'] == 0 and stats['enabled']:
            print()
            print("*** WARNING: MASKING NEVER ACTIVATED! ***")
            if agg['skip_not_training'] > 0:
                print("    -> Model was in eval mode during forward passes")
            if agg['skip_wrong_dims'] > 0:
                print("    -> Outputs had wrong dimensions (need 4D: [B,C,H,W])")

        print("="*60)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove all hooks."""
        self.remove_all_hooks()
        return False


def apply_channel_masking(
    model: nn.Module,
    layer_prefixes: List[str],
    p_apply: float = 0.5,
    p_channels: float = 0.2
) -> MaskingManager:
    """
    Convenience function to apply channel masking to a model.

    Args:
        model: The model to add masking to
        layer_prefixes: List of layer name prefixes to mask
        p_apply: Probability of applying masking per batch
        p_channels: Fraction of channels to zero

    Returns:
        MaskingManager instance (use .remove_all_hooks() when done)

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')
        >>> manager = apply_channel_masking(
        ...     model.model,
        ...     layer_prefixes=["model.5", "model.6", "model.7", "model.8", "model.9"],
        ...     p_apply=0.5,
        ...     p_channels=0.2
        ... )
        >>> # Train model...
        >>> manager.remove_all_hooks()  # Clean up when done
    """
    manager = MaskingManager(model, p_apply, p_channels)
    num_hooks = manager.add_masking_to_layers(layer_prefixes)
    print(f"Added {num_hooks} masking hooks to layers: {layer_prefixes}")
    return manager


class MaskingCallbacks:
    """
    Ultralytics callback-based masking that hooks into trainer.model.

    IMPORTANT: This is the correct way to add masking to Ultralytics models!
    The trainer creates its own model object internally, so we must add hooks
    via callbacks AFTER the trainer is initialized.

    Usage:
        from ultralytics import YOLO
        from experiments.Experiment_3.channel_masking import MaskingCallbacks

        model = YOLO("yolov8m.pt")
        callbacks = MaskingCallbacks(
            layer_prefixes=["model.5", "model.6"],
            p_apply=0.5,
            p_channels=0.2
        )
        callbacks.register(model)

        # Train - masking will be applied automatically
        model.train(data="data.yaml", epochs=50)

        # Get stats after training
        callbacks.print_summary()
    """

    def __init__(
        self,
        layer_prefixes: List[str],
        p_apply: float = 0.5,
        p_channels: float = 0.2,
        verbose: bool = False
    ):
        """
        Args:
            layer_prefixes: List of layer name prefixes to mask (e.g., ["model.5", "model.6"])
            p_apply: Probability of applying masking per forward pass
            p_channels: Fraction of channels to zero when masking
            verbose: If True, print when masking fires
        """
        self.layer_prefixes = layer_prefixes
        self.p_apply = p_apply
        self.p_channels = p_channels
        self.verbose = verbose
        self.manager: Optional[MaskingManager] = None
        self._trainer = None

    def _on_pretrain_routine_start(self, trainer):
        """
        Called before training starts but after trainer.model is set up.
        This is where we add our hooks to the actual training model.
        """
        self._trainer = trainer

        # trainer.model is the actual model used for forward passes
        actual_model = trainer.model

        if self.verbose:
            print(f"\n[MaskingCallbacks] on_pretrain_routine_start")
            print(f"[MaskingCallbacks] trainer.model type: {type(actual_model)}")
            print(f"[MaskingCallbacks] trainer.model.training: {actual_model.training}")

        # Add hooks to trainer.model
        self.manager = MaskingManager(
            actual_model,
            p_apply=self.p_apply,
            p_channels=self.p_channels,
            verbose=self.verbose
        )
        num_hooks = self.manager.add_masking_to_layers(self.layer_prefixes)

        print(f"[MaskingCallbacks] Added {num_hooks} masking hooks to trainer.model")
        print(f"[MaskingCallbacks] Layer prefixes: {self.layer_prefixes}")
        print(f"[MaskingCallbacks] Config: p_apply={self.p_apply}, p_channels={self.p_channels}")

        if num_hooks == 0:
            print(f"[MaskingCallbacks] WARNING: No hooks added! Check layer_prefixes.")

    def _on_train_epoch_end(self, trainer):
        """Called at end of each training epoch - log masking stats."""
        if self.manager and self.verbose:
            stats = self.manager.get_detailed_stats()
            agg = stats['aggregate']
            print(f"[MaskingCallbacks] Epoch end - "
                  f"mask_applications={agg['total_mask_applications']}, "
                  f"forward_calls={agg['total_calls']}")

    def _on_train_end(self, trainer):
        """Called when training ends - print final summary."""
        if self.manager:
            self.manager.print_debug_summary()

    def register(self, model) -> "MaskingCallbacks":
        """
        Register callbacks with an Ultralytics model.

        Args:
            model: YOLO or RTDETR model instance

        Returns:
            self (for chaining)
        """
        model.add_callback("on_pretrain_routine_start", self._on_pretrain_routine_start)
        model.add_callback("on_train_epoch_end", self._on_train_epoch_end)
        model.add_callback("on_train_end", self._on_train_end)
        return self

    def get_manager(self) -> Optional[MaskingManager]:
        """Get the MaskingManager (only available after training starts)."""
        return self.manager

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get masking statistics (only available after training starts)."""
        if self.manager:
            return self.manager.get_detailed_stats()
        return None

    def print_summary(self):
        """Print masking summary."""
        if self.manager:
            self.manager.print_debug_summary()
        else:
            print("[MaskingCallbacks] No stats available - training hasn't started yet")

    def remove_hooks(self):
        """Remove all hooks (called automatically at end of training)."""
        if self.manager:
            self.manager.remove_all_hooks()


if __name__ == "__main__":
    # Test with a simple model
    print("Testing channel masking...")

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            return x

    model = SimpleModel()
    model.train()

    # Add masking to conv2 and conv3
    manager = MaskingManager(model, p_apply=1.0, p_channels=0.3)
    num_hooks = manager.add_masking_to_layers(["conv2", "conv3"])
    print(f"Added {num_hooks} hooks")

    # Run forward pass
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Masking summary: {manager.get_summary()}")

    # Clean up
    manager.remove_all_hooks()
    print("Hooks removed")
