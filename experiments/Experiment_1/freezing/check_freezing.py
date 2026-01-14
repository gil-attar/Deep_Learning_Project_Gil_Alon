from ultralytics import YOLO, RTDETR
from freeze_presets import (
    YOLOV8M_PRESETS, RTDETR_L_PRESETS,
    unfreeze_by_prefixes, count_params
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "artifacts" / "weights"

print("\nYOLOv8m")
y = YOLO(str(WEIGHTS / "yolov8m.pt")).model
for k, pref in YOLOV8M_PRESETS.items():
    unfreeze_by_prefixes(y, pref)
    print(k, count_params(y))

print("\nRT-DETR-L")
r = RTDETR(str(WEIGHTS / "rtdetr-l.pt")).model
for k, pref in RTDETR_L_PRESETS.items():
    unfreeze_by_prefixes(r, pref)
    print(k, count_params(r))
