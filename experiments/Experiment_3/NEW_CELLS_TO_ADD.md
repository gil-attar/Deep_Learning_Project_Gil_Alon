# NEW CELLS TO ADD TO E3_run_evaluate.ipynb

## IMPORTANT: Before running a NEW training run

In the configuration cell (cell 2), you need to **COMMENT OUT** the hardcoded RUN_ID line:

```python
# RUN_ID = "E3_20260120_162601"  # <-- COMMENT THIS OUT for a new run!
```

This will let the notebook generate a new unique RUN_ID for your fresh training.

---

## Add these cells AFTER cell-29 (the summary table cell)

### Cell A: Markdown header
```markdown
## 9.1 Evaluate Train/Val Splits (Overfitting Analysis)

This section evaluates all 12 trained models on the **train** and **val** splits to check for overfitting.
- If F1_train >> F1_val >> F1_test → overfitting
- If F1_train ≈ F1_val ≈ F1_test → good generalization
```

### Cell B: Train/Val evaluation code
```python
# Evaluate all models on train and val splits (clean images only)
# This helps detect overfitting: if F1_train >> F1_val >> F1_test, the model overfits

print("="*70)
print("EVALUATING TRAIN/VAL SPLITS FOR OVERFITTING ANALYSIS")
print("="*70)
print(f"This will run inference on {len(MODELS) * len(SESSIONS_TO_RUN)} models x 2 splits = {len(MODELS) * len(SESSIONS_TO_RUN) * 2} evaluations")
print("Estimated time: ~10-15 minutes\n")

# Load indices
with open("data/processed/evaluation/train_index.json") as f:
    train_index = json.load(f)
with open("data/processed/evaluation/val_index.json") as f:
    val_index = json.load(f)

# Load ground truth and class names
train_gts = load_ground_truth("data/processed/evaluation/train_index.json")
val_gts = load_ground_truth("data/processed/evaluation/val_index.json")
class_names = load_class_names("data/processed/evaluation/test_index.json")

# Store results
train_val_metrics = []

for model_name in MODELS:
    for session_name in SESSIONS_TO_RUN:
        # Check if training completed
        if not is_session_complete(output_dir, model_name, session_name):
            print(f"SKIP (no training): {model_name}__{session_name}")
            continue

        # Find weights
        run_dir = output_dir / f"{model_name}__{session_name}"
        weights_path = run_dir / "weights" / "best.pt"

        if not weights_path.exists():
            print(f"WARNING: Weights not found for {model_name}__{session_name}")
            continue

        print(f"\n[{model_name}__{session_name}]")

        # Evaluate on TRAIN split (clean images)
        print(f"  Evaluating on train split ({len(train_index['images'])} images)...")
        train_preds = generate_predictions(str(weights_path), "data/raw/train/images", train_index)
        train_sweep = eval_detection_prf_at_iou(train_preds, train_gts, iou_threshold=0.5)
        train_best_thr = max(train_sweep.keys(), key=lambda k: train_sweep[k]['f1'])
        train_metrics = train_sweep[train_best_thr]
        print(f"    Train F1: {train_metrics['f1']:.4f} @ conf={train_best_thr}")

        # Evaluate on VAL split (clean images)
        print(f"  Evaluating on val split ({len(val_index['images'])} images)...")
        val_preds = generate_predictions(str(weights_path), "data/raw/valid/images", val_index)
        val_sweep = eval_detection_prf_at_iou(val_preds, val_gts, iou_threshold=0.5)
        val_best_thr = max(val_sweep.keys(), key=lambda k: val_sweep[k]['f1'])
        val_metrics = val_sweep[val_best_thr]
        print(f"    Val F1: {val_metrics['f1']:.4f} @ conf={val_best_thr}")

        # Store results
        train_val_metrics.append({
            "model": model_name,
            "session": session_name,
            "F1_train": train_metrics['f1'],
            "P_train": train_metrics['precision'],
            "R_train": train_metrics['recall'],
            "F1_val": val_metrics['f1'],
            "P_val": val_metrics['precision'],
            "R_val": val_metrics['recall'],
        })

# Save train/val metrics
with open(output_dir / "train_val_metrics.json", 'w') as f:
    json.dump(train_val_metrics, f, indent=2)

print("\n" + "="*70)
print("Train/Val evaluation complete!")
print(f"Saved to: {output_dir / 'train_val_metrics.json'}")
print("="*70)
```

### Cell C: Markdown header for comprehensive table
```markdown
## 9.2 Comprehensive Results Table (Train/Val/Test)

This table combines train, val, and test metrics to show the full picture:
- **F1_train, P_train, R_train**: Performance on training data (clean)
- **F1_val, P_val, R_val**: Performance on validation data (clean)
- **F1_test_clean, P_test_clean, R_test_clean**: Performance on test data (clean)
- **F1_test_occ, P_test_occ, R_test_occ**: Performance on test data (40% occluded)

**Interpretation:**
- If train >> val >> test_clean: Overfitting
- If train ≈ val ≈ test_clean: Good generalization
- test_clean vs test_occ gap: Occlusion robustness
```

### Cell D: Comprehensive table code
```python
# Create comprehensive table combining train/val/test metrics
# Merge train_val_metrics with existing test metrics

# Convert train_val_metrics to DataFrame
df_train_val = pd.DataFrame(train_val_metrics)

# Get test metrics from all_metrics (already loaded)
df_test = pd.DataFrame(all_metrics)

# Pivot test metrics to get clean and occluded columns
df_test_clean = df_test[df_test['test_type'] == 'clean'][['model', 'session', 'f1', 'precision', 'recall']].copy()
df_test_clean = df_test_clean.rename(columns={
    'f1': 'F1_test_clean',
    'precision': 'P_test_clean',
    'recall': 'R_test_clean'
})

df_test_occ = df_test[df_test['test_type'] == 'occluded'][['model', 'session', 'f1', 'precision', 'recall']].copy()
df_test_occ = df_test_occ.rename(columns={
    'f1': 'F1_test_occ',
    'precision': 'P_test_occ',
    'recall': 'R_test_occ'
})

# Merge all together
comprehensive = df_train_val.merge(df_test_clean, on=['model', 'session'])
comprehensive = comprehensive.merge(df_test_occ, on=['model', 'session'])

# Reorder columns for clarity
column_order = [
    'model', 'session',
    'F1_train', 'F1_val', 'F1_test_clean', 'F1_test_occ',
    'P_train', 'P_val', 'P_test_clean', 'P_test_occ',
    'R_train', 'R_val', 'R_test_clean', 'R_test_occ'
]
comprehensive = comprehensive[column_order]

# Display the table
print("\n" + "="*120)
print("COMPREHENSIVE RESULTS: Train / Val / Test (Clean) / Test (Occluded)")
print("="*120)
print(comprehensive.to_string(index=False))

# Save to CSV
comprehensive.to_csv(output_dir / "comprehensive_metrics.csv", index=False)
print(f"\nSaved to: {output_dir / 'comprehensive_metrics.csv'}")

# Also display with better formatting using pandas styling (if in notebook)
print("\n" + "="*120)
print("OVERFITTING ANALYSIS")
print("="*120)
for _, row in comprehensive.iterrows():
    model_session = f"{row['model']}__{row['session']}"
    train_f1 = row['F1_train']
    val_f1 = row['F1_val']
    test_f1 = row['F1_test_clean']

    # Calculate gaps
    train_val_gap = train_f1 - val_f1
    val_test_gap = val_f1 - test_f1

    # Determine overfitting status
    if train_val_gap > 0.15:
        status = "OVERFITTING (train >> val)"
    elif train_val_gap > 0.08:
        status = "MODERATE overfit"
    elif train_f1 < 0.3:
        status = "UNDERFITTING (poor learning)"
    else:
        status = "OK (good generalization)"

    print(f"{model_session:45s} | Train: {train_f1:.3f} | Val: {val_f1:.3f} | Test: {test_f1:.3f} | Gap: {train_val_gap:+.3f} | {status}")

print("="*120)
```

---

## Summary of changes needed:

1. **COMMENT OUT line 111** (the hardcoded RUN_ID) to get a fresh run
2. **Add 4 new cells** after the summary table (cell-29):
   - Cell A: Markdown header for train/val evaluation
   - Cell B: Code to evaluate train/val splits
   - Cell C: Markdown header for comprehensive table
   - Cell D: Code to create and display the comprehensive table
