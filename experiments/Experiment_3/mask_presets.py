"""
Mask Presets for Experiment 3 - Internal Masking vs Occlusion Training

Layer boundaries are based on the architecture analysis from Experiment 1.
See: experiments/Experiment_1/freezing/notes/README.md

YOLOv8m architecture:
  - Backbone: model.0-9
  - Neck: model.10-21
  - Head: model.22

RT-DETR-L architecture:
  - Backbone: model.0-11
  - Neck: model.12-27
  - Head: model.28
"""

from typing import List, Dict


def prefixes_for_range(start: int, end: int) -> List[str]:
    """Generate layer prefixes for inclusive range: model.start ... model.end"""
    return [f"model.{i}" for i in range(start, end + 1)]


# =============================================================================
# YOLOv8 Masking Presets
# =============================================================================
# Backbone: model.0-9 (split into early 0-4, late 5-9)
# Neck: model.10-21
# Head: model.22

YOLOV8_MASK_PRESETS: Dict[str, List[str]] = {
    "backbone_early": prefixes_for_range(0, 4),    # First half of backbone
    "backbone_late":  prefixes_for_range(5, 9),    # Second half of backbone
    "neck":           prefixes_for_range(10, 21),  # Feature pyramid network
    "head":           ["model.22"],                # Detection head
}


# =============================================================================
# RT-DETR Masking Presets
# =============================================================================
# Backbone: model.0-11 (split into early 0-5, late 6-11)
# Neck: model.12-27
# Head: model.28

RTDETR_MASK_PRESETS: Dict[str, List[str]] = {
    "backbone_early": prefixes_for_range(0, 5),    # First half of backbone
    "backbone_late":  prefixes_for_range(6, 11),   # Second half of backbone
    "neck":           prefixes_for_range(12, 27),  # Transformer encoder/fusion
    "head":           ["model.28"],                # Decoder head
}


# =============================================================================
# Session Definitions
# =============================================================================
# Maps session names to their configuration

SESSIONS = {
    "S1_clean_train": {
        "description": "Baseline: train on clean, no masking",
        "train_data": "clean",
        "mask_location": None,
    },
    "S2_occ_train": {
        "description": "Standard: train on occluded images",
        "train_data": "occluded",
        "mask_location": None,
    },
    "S3_mask_backbone_early": {
        "description": "Clean train + mask backbone early layers",
        "train_data": "clean",
        "mask_location": "backbone_early",
    },
    "S4_mask_backbone_late": {
        "description": "Clean train + mask backbone late layers",
        "train_data": "clean",
        "mask_location": "backbone_late",
    },
    "S5_mask_neck": {
        "description": "Clean train + mask neck layers",
        "train_data": "clean",
        "mask_location": "neck",
    },
    "S6_mask_head": {
        "description": "Clean train + mask head layers",
        "train_data": "clean",
        "mask_location": "head",
    },
}


def get_mask_prefixes(model_type: str, mask_location: str) -> List[str]:
    """
    Get layer prefixes for masking based on model type and location.

    Args:
        model_type: "yolo" or "rtdetr"
        mask_location: "backbone_early", "backbone_late", "neck", or "head"

    Returns:
        List of layer name prefixes to mask
    """
    if model_type.lower() in ["yolo", "yolov8", "yolov8n", "yolov8m", "yolov8l"]:
        presets = YOLOV8_MASK_PRESETS
    elif model_type.lower() in ["rtdetr", "rt-detr", "rtdetr-l"]:
        presets = RTDETR_MASK_PRESETS
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'yolo' or 'rtdetr'")

    if mask_location not in presets:
        raise ValueError(f"Unknown mask location: {mask_location}. "
                        f"Options: {list(presets.keys())}")

    return presets[mask_location]


def get_session_config(session_name: str) -> Dict:
    """Get configuration for a session by name."""
    if session_name not in SESSIONS:
        raise ValueError(f"Unknown session: {session_name}. "
                        f"Options: {list(SESSIONS.keys())}")
    return SESSIONS[session_name]


def list_sessions() -> List[str]:
    """List all available session names."""
    return list(SESSIONS.keys())


if __name__ == "__main__":
    # Print summary for verification
    print("=" * 60)
    print("EXPERIMENT 3 - MASKING PRESETS")
    print("=" * 60)

    print("\nYOLOv8 Layer Boundaries:")
    for location, prefixes in YOLOV8_MASK_PRESETS.items():
        print(f"  {location}: {prefixes}")

    print("\nRT-DETR Layer Boundaries:")
    for location, prefixes in RTDETR_MASK_PRESETS.items():
        print(f"  {location}: {prefixes}")

    print("\nSessions:")
    for name, config in SESSIONS.items():
        print(f"  {name}: {config['description']}")
