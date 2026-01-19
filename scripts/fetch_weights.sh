#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/weights

# YOLOv8m
if [ ! -f artifacts/weights/yolov8m.pt ]; then
  curl -L -o artifacts/weights/yolov8m.pt \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt
fi

# RT-DETR L
if [ ! -f artifacts/weights/rtdetr-l.pt ]; then
  curl -L -o artifacts/weights/rtdetr-l.pt \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt
fi

echo "OK: weights are present under artifacts/weights/"
ls -lh artifacts/weights
