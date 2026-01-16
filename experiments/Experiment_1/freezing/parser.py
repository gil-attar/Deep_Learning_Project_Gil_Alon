import json
import re
from collections import defaultdict
from pathlib import Path

''' 
This script is for parsing both models blocks into top-level 
blocks to identify model backbone, neck and head.
'''

HERE = Path(__file__).resolve().parent
MAPS = HERE / "maps"

YOLO_MAP = MAPS / "yolov8m_modules.json"
RTDETR_MAP = MAPS / "rtdetr-l_modules.json"


def load(path: Path):
    with path.open("r") as f:
        return json.load(f)


def top_index(name: str):
    """
    Extract the top-level block index from a module name like:
      'model.10.m.0.cv1' -> 10
      'model.22'         -> 22
    """
    m = re.match(r"model\.(\d+)", name)
    return int(m.group(1)) if m else None


def summarize(mods):
    """
    Group all submodules by top-level 'model.<N>' index and print a small summary
    of the layer types that appear under each block.
    """
    top = defaultdict(set)
    for m in mods["modules"]:
        idx = top_index(m["name"])
        if idx is None:
            continue
        top[idx].add(m.get("type", "<?>"))

    for idx in sorted(top):
        types = ", ".join(sorted(list(top[idx]))[:8])
        print(f"model.{idx}: {types}")


def main():
    yolo = load(YOLO_MAP)
    rtd = load(RTDETR_MAP)

    print("\n=== YOLOv8m top-level blocks ===")
    summarize(yolo)

    print("\n=== RT-DETR-L top-level blocks ===")
    summarize(rtd)

    print("\nSuggested splits (starting point):")
    print("YOLOv8m: backbone=model.0-9, neck=model.10-21, head=model.22")
    print("RT-DETR-L: backbone=model.0-11, neck=model.12-27, head=model.28")


if __name__ == "__main__":
    main()
