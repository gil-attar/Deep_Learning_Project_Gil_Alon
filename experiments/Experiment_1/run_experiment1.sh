#!/usr/bin/env bash
# Experiment 1: run all 8 runs (2 models × F0–F3), with per-run retry (max 2 attempts).
# Level-1 RESUME ENABLED:
#   - Skip a run if run_summary.json exists AND matches (EPOCHS, IMGSZ, SEED).
#   - Optionally archive incomplete run dirs (missing run_summary.json) before re-running.

set -u  # error on unset vars

# --------- CONFIG ---------
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
SEED="${SEED:-0}"

# If Colab has a GPU, Ultralytics will typically auto-use it.
# This line makes it explicit that we want GPU 0 if present.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Optional: reduce noisy parallelism on Colab
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Path to our runner
RUNNER="experiments/Experiment_1/runOneTest.py"

# Resume behavior knobs:
#  - ARCHIVE_INCOMPLETE=1 moves partial run dirs aside before re-running (safer).
#  - ARCHIVE_INCOMPLETE=0 leaves partial dirs in place (may mix old/new artifacts).
ARCHIVE_INCOMPLETE="${ARCHIVE_INCOMPLETE:-1}"

# --------- KEEPALIVE (prevents idle disconnect due to no output) ---------
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

# Return 0 (true) if run_summary.json exists AND matches this sweep's EPOCHS/IMGSZ/SEED.
# Return nonzero otherwise.
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
}

archive_if_incomplete() {
  local model="$1"
  local freeze="$2"
  local run_dir="experiments/Experiment_1/runs/${model}/${freeze}"
  local summary="${run_dir}/run_summary.json"

  [[ -d "$run_dir" ]] || return 0
  [[ -f "$summary" ]] && return 0

  if [[ "${ARCHIVE_INCOMPLETE}" == "1" ]]; then
    local archive_dir="experiments/Experiment_1/runs/_incomplete_archive"
    mkdir -p "$archive_dir"
    local ts
    ts="$(date -u '+%Y%m%d_%H%M%S')"
    local dest="${archive_dir}/${model}_${freeze}_${ts}"
    echo "[RESUME] Incomplete run detected at ${run_dir}. Archiving to ${dest}"
    mv "$run_dir" "$dest"
  else
    echo "[RESUME] Incomplete run detected at ${run_dir}. Leaving in place (may overwrite/merge artifacts)."
  fi
}

run_one() {
  local model="$1"
  local freeze="$2"

  echo "============================================================"
  echo "[RUN] model=${model} freeze=${freeze} epochs=${EPOCHS} imgsz=${IMGSZ} seed=${SEED}"
  echo "============================================================"

  # Level-1 resume: skip if already complete and matches current settings
  if is_completed_and_matching "$model" "$freeze"; then
    echo "[SKIP] model=${model} freeze=${freeze} already completed (matching epochs/imgsz/seed)."
    return 0
  fi

  # If partial outputs exist, optionally archive before re-running
  archive_if_incomplete "$model" "$freeze"

  # Attempt 1
  python3 "$RUNNER" --model "$model" --freeze "$freeze" --epochs "$EPOCHS" --imgsz "$IMGSZ" --seed "$SEED"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[OK] model=${model} freeze=${freeze} (attempt 1)"
    return 0
  fi

  echo "[WARN] model=${model} freeze=${freeze} failed (attempt 1, rc=${rc}). Retrying once..."
  sleep 5

  # Attempt 2
  python3 "$RUNNER" --model "$model" --freeze "$freeze" --epochs "$EPOCHS" --imgsz "$IMGSZ" --seed "$SEED"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[OK] model=${model} freeze=${freeze} (attempt 2)"
    return 0
  fi

  echo "[FAIL] model=${model} freeze=${freeze} failed twice (rc=${rc}). Skipping to next run."
  return $rc
}

# --------- (Optional) quick GPU visibility print ---------
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi available; GPU status:"
  nvidia-smi || true
else
  echo "[INFO] nvidia-smi not found (CPU runtime or drivers not exposed)."
fi

# --------- RUN MATRIX ---------
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
