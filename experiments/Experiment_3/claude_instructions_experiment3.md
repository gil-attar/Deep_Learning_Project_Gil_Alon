# Instructions for Claude Code — Experiment 3 (Internal Masking vs Occlusion Training)

Gil wants you to implement **Experiment 3** for the “Meal Planner from Ingredients Image” project.

This document is the spec you should implement. You can and should reuse/adjust our existing evaluation scripts + existing JSON outputs.

---

## 1) Experiment goal
We are testing robustness to occlusions.

For **each model separately** (e.g., YOLOv8m and RT-DETRl), we will run 6 sessions:

### Session set (6 runs per model)
1. **clean_train → test_occluded**  (baseline robustness)
2. **occluded_train → test_occluded** (standard practice)
3. **clean_train + internal masking @ backbone_early → test_occluded**
4. **clean_train + internal masking @ backbone_late → test_occluded**
5. **clean_train + internal masking @ neck → test_occluded**
6. **clean_train + internal masking @ head → test_occluded**

We will also evaluate **every trained checkpoint** on **test_clean** (separate graphs):
- clean_train → test_clean
- occluded_train → test_clean
- each internal-masking run → test_clean

### Key rule
- **Internal masking is training-only.**
- During test/evaluation we **do not** apply internal masking. Testing uses the full model capacity.
- Occlusion testing is done only by occluding the **input images** (pixel-level occlusion suite).

---

## 2) Dataset + occlusion suite requirements
We already have a working occlusion script from earlier experiments.

### Must-have properties
- We use the **same exact image IDs** for train/val/test across all sessions for a given dataset.
- We use a **fixed occlusion test suite**:
  - Same base test images
  - Same occlusion masks and random seeds
  - The occluded images should be saved to disk (recommended) so every run is identical.

### Naming convention (recommended)
- `test_clean/` : original test images
- `test_occ_fixed/` : occluded versions of exactly the same test images

If the repo already has a different naming scheme, keep it but ensure the concept above is preserved.

---

## 3) Training hyperparameters policy (important)
We want to avoid per-session hyperparameter tuning.

- Use **Ultralytics defaults** for each model.
- Only override:
  - `epochs` (fixed constant for all 6 sessions of that model)
  - `batch` (only if forced by GPU RAM)
  - `imgsz` (only if needed)
  - `project/name` output dirs for clean experiment tracking

No early stopping required. If you keep validation enabled for checkpoint selection, do not tune hyperparameters based on it.

---

## 4) Internal masking implementation spec 
We need internal masking during training for 4 different injection points:
- backbone_early
- backbone_late
- neck
- head

### Masking type - talk about it with Gil and reccomand how to make the correct masking. here is what gpt reccomanded, but its better for you to also have your opinion on it:
Use activation masking (drop activations) rather than freezing weights:
- For a chosen activation tensor `X` shaped `[B, C, H, W]`, create a random mask and zero-out parts of `X`.

Preferred (simple + stable): **channel masking**
- Randomly select a fraction `p_channels` of channels per batch and set them to 0.

Optional (if easy): **spatial masking**
- Apply 1–N random rectangles on feature map spatial dims (like CutOut but on features).

### Mask schedule
- Apply masking with probability `p_apply` each batch.
- Start with safe defaults (can be constants in config):
  - `p_apply = 0.5`
  - `p_channels = 0.1` or `0.2` (keep it modest so training doesn’t collapse)

### Where to hook (Ultralytics)
Implementation should be robust to minor model changes:
- Identify the modules for the target location by name or index.
- Add forward hooks to modify activations during training.
- Ensure masking is enabled only when `model.train()` is True.

We need four config options that map to four hook selections.

---

## 5) Evaluation requirements (reuse our existing scripts)
We already have evaluation scripts and JSON formats from previous experiments.
Please reuse them and only adjust if necessary.

### Required evaluation outputs per run
For each trained run, compute:
1. **Detection metrics** (if available in our pipeline): mAP50 and/or mAP50-95.
2. **Ingredient-list quality** (core metric): Precision / Recall / F1 over ingredient set.
   - Use our existing matching rules (we previously discussed Hungarian matching/IoU matching for detections if needed).

### JSON + CSV outputs
For each session, produce:
- `metrics.json` (machine-readable summary)
- `predictions.json` if we already store predictions like this
- `metrics.csv` aggregated across all sessions

### Plots (pretty graphs). try use the evaluation scripts is possible/modify them
Per model:
- Plot F1 on **test_occluded** for the 6 sessions.
- Plot F1 on **test_clean** for the same 6 sessions (separate graph).
- If available, also plot mAP similarly.

Across both models:
- Final comparison plot: one plot with **12 bars** (6 sessions × 2 models) for F1 on test_occluded.
  - Bars grouped by session, with two adjacent bars per session (Model A vs Model B).
- Also create the same 12-bar plot for test_clean (optional but recommended).

---

## 6) Notebooks to generate (Colab)
Create two notebooks (or generate them programmatically).

### Notebook A — Full Experiment (overnight)
- Runs all 12 trainings (6 sessions × 2 models) with fixed epochs.
- Runs evaluation on both `test_clean` and `test_occ_fixed`.
- Produces:
  - aggregated CSV/JSON
  - all plots

### Notebook B — Smoke Test (super fast)
- Runs the exact same pipeline but with `epochs = 1` for all 12 sessions.
- Goal: catch pathing/syntax/dataset issues before an overnight run.
- Must still:
  - train
  - evaluate
  - write metrics files

Both notebooks should have a single config cell at the top to set:
- dataset paths
- output root dir
- model list
- epochs (override)
- whether to use GPU

---

## 7) Crash-safe overnight runner (important)
We previously used `run_experiment1.sh` to handle notebook crashes / resume.

We want something similar for Experiment 3:
- A bash script (or python runner) that:
  - runs each session sequentially
  - logs stdout/stderr to per-session log files
  - skips completed sessions (checks for `metrics.json` or a `DONE` file)
  - can be resumed after interruption

### Minimum behaviors
- If a session fails, it should:
  - write a `FAILED` marker
  - continue to next session (or stop, but make it configurable)
- At the end, it should run aggregation + plotting.

### “Check it works” requirement
Include a quick sanity run mode:
- uses `epochs=1`
- runs 1 model × 2 sessions
- verifies the resume logic by re-running and skipping completed outputs

---

## 8) Repo integration notes
- Do not rewrite everything from scratch. Prefer integrating with existing:
  - dataset YAMLs
  - evaluation scripts
  - existing JSON schema(s)
- If our JSON formats are inconsistent, propose a minimal harmonization, but keep backward compatibility if possible.

---

## 9) Deliverables checklist
After you implement, Gil should have:
- Two Colab notebooks:
  - `experiment3_full.ipynb`
  - `experiment3_smoketest_1epoch.ipynb`
- A crash-safe runner:
  - `run_experiment3.sh` (or python equivalent)
- A results folder structure like:
  - `runs/exp3/<model>/<session_name>/{weights, metrics.json, plots, logs}`
- Aggregated outputs:
  - `runs/exp3/summary_metrics.csv`
  - `runs/exp3/summary_metrics.json`
  - `runs/exp3/plots/*.png`

---

## 10) Session naming (please follow exactly)
For each model name `<MODEL>`:
- `<MODEL>__S1_clean_train`
- `<MODEL>__S2_occ_train`
- `<MODEL>__S3_mask_backbone_early`
- `<MODEL>__S4_mask_backbone_late`
- `<MODEL>__S5_mask_neck`
- `<MODEL>__S6_mask_head`

This will simplify aggregation.

---

## 11) Clarifying assumptions you should resolve by inspecting the repo (do not ask Gil unless necessary)
- Where the occlusion generator lives and how it currently saves/loads occluded images.
- The exact dataset YAML path(s).
- Existing evaluation script entrypoints and expected JSON formats.
- Which two models are currently used (likely YOLOv8 + RT-DETR) and how they are trained in our repo.

Implement based on repo reality and keep the above experiment logic intact.
