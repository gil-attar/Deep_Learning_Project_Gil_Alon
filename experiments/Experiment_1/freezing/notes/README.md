# Experiment 1 – Freezing Ladder Definition (Backbone / Neck / Head)

## Purpose
Experiment 1 measures how the number of trainable parameters during fine-tuning affects performance when adapting large COCO-pretrained detectors (YOLOv8m, RT-DETR-L) to our ingredient dataset (~2,000 images).

We implement a freezing ladder (F0–F3) that monotonically increases the number of trainable parameters.

## How architecture boundaries were identified
We generated per-model module maps using:
- `inspect_modules.py` → writes JSON inventories under `freezing/maps/`
- `parser.py` → groups modules by top-level indices (`model.<N>`) and prints a type summary

Raw parser output is stored in:
- `notes/parser_run.txt`

### YOLOv8m boundary evidence
From `notes/parser_run.txt`:
- `model.9` contains `SPPF` → typical end-of-backbone block
- `model.10` is `Upsample` and `model.11` is `Concat` → start of feature pyramid fusion (neck)
- `model.22` contains `Detect`/`DFL` → detection head

Final split:
- Backbone: `model.0–9`
- Neck: `model.10–21`
- Head: `model.22`

### RT-DETR-L boundary evidence
From `notes/parser_run.txt`:
- `model.0–10` are HGNet stem/blocks (feature extractor)
- `model.11` contains `AIFI` / `MultiheadAttention` → backbone enhancement stage
- `model.13,15,18,20,...` show `Upsample/Concat` → neck fusion
- `model.28` contains `DeformableTransformerDecoder*` → detection head/decoder

Final split:
- Backbone: `model.0–11`
- Neck: `model.12–27`
- Head: `model.28`

## “top-K backbone blocks”
K is the number of backbone blocks closest to the neck that are unfrozen in F2. we set k=2 to try
to limit trainable parameter increase while retaining a noticable change in outputs.

## Freezing ladder (presets)
### YOLOv8m
- F0: head block only (`model.22`)
- F1: neck + head (`model.10–22`)
- F2: neck + head + top-K backbone (`model.10–22` + `model.8–9`)
- F3: full fine-tune (`model.0–22`)

### RT-DETR-L
- F0: head block only (`model.28`)
- F1: neck + head (`model.12–28`)
- F2: neck + head + top-K backbone (`model.12–28` + `model.10–11`)
- F3: full fine-tune (`model.0–28`)

## Trainable parameter counts
Trainable parameter counts per configuration are recorded in:
- `notes/trainable_params.txt`

These counts were produced by:
- `check_freezing.py`
