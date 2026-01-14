import json
from pathlib import Path

import torch
from ultralytics import YOLO, RTDETR

ROOT = Path(__file__).resolve().parents[3]  # repo root
WEIGHTS = ROOT / "artifacts" / "weights"
OUTDIR = Path(__file__).resolve().parent / "maps"
OUTDIR.mkdir(parents=True, exist_ok=True)

def module_param_counts(m: torch.nn.Module):
    total = sum(p.numel() for p in m.parameters(recurse=False))
    trainable = sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
    return total, trainable


def dump_model_modules(model, out_json_path: Path):
    rows = []
    for name, m in model.named_modules():
        # skip empty name (root) if you want; I keep it for completeness
        total, trainable = module_param_counts(m)
        rows.append({
            "name": name,
            "type": m.__class__.__name__,
            "params_total_local": int(total),
            "params_trainable_local": int(trainable),
        })

    # global counts
    global_total = sum(p.numel() for p in model.parameters())
    global_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    payload = {
        "global_params_total": int(global_total),
        "global_params_trainable": int(global_trainable),
        "modules": rows,
    }
    out_json_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] wrote: {out_json_path}")


def main():
    # YOLOv8m
    yolo = YOLO(str(WEIGHTS / "yolov8m.pt"))
    dump_model_modules(yolo.model, OUTDIR / "yolov8m_modules.json")

    # RT-DETR-L
    rtdetr = RTDETR(str(WEIGHTS / "rtdetr-l.pt"))
    dump_model_modules(rtdetr.model, OUTDIR / "rtdetr-l_modules.json")


if __name__ == "__main__":
    main()
