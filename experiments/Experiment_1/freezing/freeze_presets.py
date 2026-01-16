from __future__ import annotations
from typing import Iterable, List, Dict
import re
import torch


def freeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_by_prefixes(model: torch.nn.Module, trainable_prefixes: Iterable[str]) -> None:
    """
    Freeze everything, then unfreeze any module whose name starts with one of the prefixes.
    Use recurse=True to ensure container modules enable parameters in their children.
    """
    prefixes = tuple(trainable_prefixes)
    freeze_all(model)

    for name, m in model.named_modules():
        if name.startswith(prefixes):
            for p in m.parameters(recurse=True):
                p.requires_grad = True


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def prefixes_for_range(start: int, end: int) -> List[str]:
    """Inclusive range: model.start ... model.end"""
    return [f"model.{i}" for i in range(start, end + 1)]


def add_topk_backbone(backbone_end: int, k: int) -> List[str]:
    """Returns prefixes for top-K backbone blocks: model.backbone_end, model.backbone_end-1, ..."""
    return [f"model.{i}" for i in range(backbone_end - k + 1, backbone_end + 1)]


# -------------------------
# Presets (K fixed globally)
# -------------------------
TOPK = 2  # recommended for your Experiment 1 ladder


# YOLOv8m: backbone 0-9, neck 10-21, head 22
YOLOV8M_PRESETS = {
    "F0": ["model.22"],  # head only
    "F1": prefixes_for_range(10, 22),  # neck + head
    "F2": prefixes_for_range(10, 22) + add_topk_backbone(backbone_end=9, k=TOPK),  # + top-K backbone blocks
    "F3": prefixes_for_range(0, 22),  # full model
}


# RT-DETR-L: backbone 0-11, neck 12-27, head 28
RTDETR_L_PRESETS = {
    "F0": ["model.28"],  # decoder/head only
    "F1": prefixes_for_range(12, 28),  # neck + head
    "F2": prefixes_for_range(12, 28) + add_topk_backbone(backbone_end=11, k=TOPK),
    "F3": prefixes_for_range(0, 28),  # full model
}
