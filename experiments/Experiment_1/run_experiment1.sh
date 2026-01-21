#!/usr/bin/env bash
# Experiment 1: run all 8 runs (2 models x F0-F3), with per-run retry (max 2 attempts).
# RESUME ENABLED (hardened):
#   - Skip a run only if run_summary.json exists, matches (EPOCHS, IMGSZ, SEED),
#     AND all contract-required artifacts exist (RUN_CONTRACT.md).
#   - Optionally archive incomplete run dirs before re-running.

set -u  # error on unset vars

# --------- CONFIG ---------
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
SEED="${SEED:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

RUNNER="experiments/Experiment_1/runOneTest.py"

ARCHIVE_INCOMPLETE="${ARCHIVE_INCOMPLETE:-1}"

# --------- KEEPALIVE ---------
keepalive() {
  while true; do
    echo "[KEEPALIVE] $(date -u '+%Y-%m-%d %H:%M:%S UTC') still running..."
    sleep 60
  done
}
keepalive &
KEEPALIVE_PID=$!

cleanup() {
  kill "$KEEPALIVE_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --------- HELPERS ---------

# Contract-required artifacts (RUN_CONTRACT.md)
required_artifacts_exist() {
  local model="$1"
  local freeze="$2"
  local run_dir="experiments/Experiment_1/runs/${model}/${freeze}"

  # Required files
  local req=(
    "run_manifest.json"
    "run_summary.json"
    "train_summary.json"
    "predictions/val_predictions.json"
    "predictions/test_predictions.json"
    "eval/val/metrics.json"
    "eval/val/summary.csv"
    "eval/val/plots/threshold_sweep.png"
    "eval/val/plots/per_class_f1.png"
    "eval/val/plots/confusion_matrix.png"
    "eval/val/plots/count_mae_comparison.png"
    "eval/test/metrics.json"
    "eval/test/summary.csv"
    "eval/test/plots/threshold_sweep.png"
    "eval/test/plots/per_class_f1.png"
    "eval/test/plots/confusion_matrix.png"
    "eval/test/plots/count_mae_comparison.png"
    "weights/best.pt"
    "weights/last.pt"
  )

  for rel in "${req[@]}"; do
    if [[ ! -f "${run_dir}/${rel}" ]]; then
      return 1
    fi
  done

  # args.yaml can be in one of several acceptable locations
  if [[ -f "${run_dir}/ultralytics/args.yaml" ]] || [[ -f "${run_dir}/args.yaml" ]]; then
    return 0
  fi

  return 1
}

# Return 0 (true) if run_summary.json exists AND matches this sweep's EPOCHS/IMGSZ/SEED
# AND all contract-required artifacts exist.
is_completed_and_matching() {
  local model="$1"
  local freeze="$2"
  local summary="experiments/Experiment_1/runs/${model}/${freeze}/run_summary.json"

  [[ -f "$summary" ]] || return 1

  python3 - <<PY
import json, sys
p = r"$summary"
want_epochs = int(r"$EPOCHS")
want_imgsz  = int(r"$IMGSZ")
want_seed   = int(r"$SEED")
with open(p, "r") as f:
    j = json.load(f)
m = j.get("manifest", {})
ok = (
    int(m.get("epochs", -1)) == want_epochs and
    int(m.get("imgsz",  -1)) == want_imgsz and
    int(m.get("seed",   -1)) == want_seed
)
sys.exit(0 if ok else 2)
PY

  local rc=$?
  [[ $rc -eq 0 ]] || return 1

  required_artifacts_exist "$model" "$freeze" || return 1
  return 0
}

archive_if_incomplete() {
  local model="$1"
  local freeze="$2"
  local run_dir="experiments/Experiment_1/runs/${model}/${freeze}"

  [[ -d "$run_dir" ]] || return 0

  # If it is fully complete per contract, do not archive.
  if required_artifacts_exist "$model" "$freeze"; then
    return 0
  fi

  if [[ "${ARCHIVE_INCOMPLETE}" == "1" ]]; then
    local archive_dir="experiments/Experiment_1/runs/_incomplete_archive"
    mkdir -p "$archive_dir"
    local ts
    ts="$(date -u '+%Y%m%d_%H%M%S')"
    local dest="${archive_dir}/${model}_${freeze}_${ts}"
    echo "[RESUME] Incomplete/contract-missing run detected at ${run_dir}. Archiving to ${dest}"
    mv "$run_dir" "$dest"
  else
    echo "[RESUME] Incomplete/contract-missing run detected at ${run_dir}. Leaving in place (may overwrite/merge artifacts)."
  fi
}

run_one() {
  local model="$1"
  local freeze="$2"

  echo "============================================================"
  echo "[RUN] model=${model} freeze=${freeze} epochs=${EPOCHS} imgsz=${IMGSZ} seed=${SEED}"
  echo "============================================================"

  if is_completed_and_matching "$model" "$freeze"; then
    echo "[SKIP] model=${model} freeze=${freeze} already completed (matching epochs/imgsz/seed + contract artifacts)."
    return 0
  fi

  archive_if_incomplete "$model" "$freeze"

  python3 "$RUNNER" --model "$model" --freeze "$freeze" --epochs "$EPOCHS" --imgsz "$IMGSZ" --seed "$SEED"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[OK] model=${model} freeze=${freeze} (attempt 1)"
    return 0
  fi

  echo "[WARN] model=${model} freeze=${freeze} failed (attempt 1, rc=${rc}). Retrying once..."
  sleep 5

  python3 "$RUNNER" --model "$model" --freeze "$freeze" --epochs "$EPOCHS" --imgsz "$IMGSZ" --seed "$SEED"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[OK] model=${model} freeze=${freeze} (attempt 2)"
    return 0
  fi

  echo "[FAIL] model=${model} freeze=${freeze} failed twice (rc=${rc}). Skipping to next run."
  return $rc
}

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi available; GPU status:"
  nvidia-smi || true
else
  echo "[INFO] nvidia-smi not found (CPU runtime or drivers not exposed)."
fi

MODELS=("yolov8m" "rtdetr-l")
FREEZES=("F0" "F1" "F2" "F3")

FAILED=0
for m in "${MODELS[@]}"; do
  for f in "${FREEZES[@]}"; do
    run_one "$m" "$f" || FAILED=$((FAILED+1))
  done
done

echo "============================================================"
echo "[DONE] Completed all runs. Failed runs: ${FAILED} (each had 2 attempts)."
echo "============================================================"
exit 0
