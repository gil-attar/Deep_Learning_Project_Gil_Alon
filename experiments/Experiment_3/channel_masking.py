"""
Channel Masking Implementation for Experiment 3

This module provides forward hooks that apply channel masking to specific
layers during training. Masking is ONLY applied during training, not inference.

Channel masking zeros out entire feature channels (not individual values),
simulating information loss similar to what occlusions cause.
"""

import torch
import torch.nn as nn
from typing import List, Callable, Optional
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
        name: str = "unnamed"
    ):
        """
        Args:
            p_apply: Probability of applying masking on each forward pass (0-1)
            p_channels: Fraction of channels to zero when masking is applied (0-1)
            name: Layer name for debugging/logging
        """
        self.p_apply = p_apply
        self.p_channels = p_channels
        self.name = name
        self.mask_count = 0  # Track how many times masking was applied

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
        # Only mask during training
        if not module.training:
            return output

        # Only mask with probability p_apply
        if random.random() > self.p_apply:
            return output

        # Handle different output types
        if not isinstance(output, torch.Tensor):
            # Some layers return tuples or other types - skip masking
            return output

        # Need at least 4 dims for channel masking: [B, C, H, W]
        if output.dim() < 4:
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

        return output * mask


class MaskingManager:
    """
    Manages channel masking hooks for a model.

    Provides methods to add/remove masking hooks to specific layers
    identified by their name prefixes.
    """

    def __init__(self, model: nn.Module, p_apply: float = 0.5, p_channels: float = 0.2):
        """
        Args:
            model: The model to add masking to
            p_apply: Probability of applying masking per batch
            p_channels: Fraction of channels to zero when masking
        """
        self.model = model
        self.p_apply = p_apply
        self.p_channels = p_channels
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
                        name=name
                    )
                    handle = module.register_forward_hook(hook)
                    self.hooks.append((handle, hook))
                    hooks_added += 1

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
