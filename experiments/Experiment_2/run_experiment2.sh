#!/usr/bin/env bash
# Runs: 2 models × 5 epoch budgets = 10 runs
# Epoch sweep (default): 5 10 20 40 80
# Level-1 RESUME ENABLED:
#   - Skip a run if run_summary.json exists AND matches (EPOCHS, IMGSZ, SEED).
#   - Optionally archive incomplete run dirs (missing run_summary.json) before re-running.

set -u  # error on unset vars

# --------- CONFIG (constants for this experiment) ---------
IMGSZ="${IMGSZ:-640}"
SEED="${SEED:-0}"

# Choose ONE freeze configuration from Experiment 1 and keep it fixed for all E2 runs.
# Override from environment if desired:  FREEZE=F2 bash experiments/Experiment_2/run_experiment2.sh
FREEZE="${FREEZE:-F2}"

# Epoch sweep list (space-separated). Override from environment if desired:
#   EPOCHS_LIST="5 10 20 40 80" bash experiments/Experiment_2/run_experiment2.sh
if [[ -n "${EPOCHS_LIST:-}" ]]; then
  read -r -a EPOCHS_ARR <<< "${EPOCHS_LIST}"
else
  EPOCHS_ARR=(5 10 20 40 80)
fi

# If Colab has a GPU, Ultralytics will typically auto-use it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Optional: reduce noisy parallelism on Colab
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Path to our E2 runner
RUNNER="experiments/Experiment_2/runOneTest.py"

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

# Return 0 if run_summary.json exists AND matches (epochs/imgsz/seed) for this specific run.
is_completed_and_matching() {
  local model="$1"
  local freeze="$2"
  local epochs="$3"
  local summary="experiments/Experiment_2/runs/${model}/${freeze}/E${epochs}/run_summary.json"

  [[ -f "$summary" ]] || return 1

  python3 - <<PY
import json, sys
p = r"$summary"
want_epochs = int(r"$epochs")
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
  local epochs="$3"
  local run_dir="experiments/Experiment_2/runs/${model}/${freeze}/E${epochs}"
  local summary="${run_dir}/run_summary.json"

  [[ -d "$run_dir" ]] || return 0
  [[ -f "$summary" ]] && return 0

  if [[ "${ARCHIVE_INCOMPLETE}" == "1" ]]; then
    local archive_dir="experiments/Experiment_2/runs/_incomplete_archive"
    mkdir -p "$archive_dir"
    local ts
    ts="$(date -u '+%Y%m%d_%H%M%S')"
    local dest="${archive_dir}/${model}_${freeze}_E${epochs}_${ts}"
    echo "[RESUME] Incomplete run detected at ${run_dir}. Archiving to ${dest}"
    mv "$run_dir" "$dest"
  else
    echo "[RESUME] Incomplete run detected at ${run_dir}. Leaving in place (may overwrite/merge artifacts)."
  fi
}

run_one() {
  local model="$1"
  local freeze="$2"
  local epochs="$3"

  echo "============================================================"
  echo "[RUN] model=${model} freeze=${freeze} epochs=${epochs} imgsz=${IMGSZ} seed=${SEED}"
  echo "============================================================"

  # Level-1 resume: skip if already complete and matches current settings
  if is_completed_and_matching "$model" "$freeze" "$epochs"; then
    echo "[SKIP] model=${model} freeze=${freeze} epochs=${epochs} already completed (matching epochs/imgsz/seed)."
    return 0
  fi

  # If partial outputs exist, optionally archive before re-running
  archive_if_incomplete "$model" "$freeze" "$epochs"

  # Attempt 1
  python3 "$RUNNER" --model "$model" --freeze "$freeze" --epochs "$epochs" --imgsz "$IMGSZ" --seed "$SEED"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[OK] model=${model} freeze=${freeze} epochs=${epochs} (attempt 1)"
    return 0
  fi

  echo "[WARN] model=${model} freeze=${freeze} epochs=${epochs} failed (attempt 1, rc=${rc}). Retrying once..."
  sleep 5

  # Attempt 2
  python3 "$RUNNER" --model "$model" --freeze "$freeze" --epochs "$epochs" --imgsz "$IMGSZ" --seed "$SEED"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[OK] model=${model} freeze=${freeze} epochs=${epochs} (attempt 2)"
    return 0
  fi

  echo "[FAIL] model=${model} freeze=${freeze} epochs=${epochs} failed twice (rc=${rc}). Skipping to next run."
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

FAILED=0

for m in "${MODELS[@]}"; do
  for e in "${EPOCHS_ARR[@]}"; do
    run_one "$m" "$FREEZE" "$e" || FAILED=$((FAILED+1))
  done
done

echo "============================================================"
echo "[DONE] Completed all runs. Failed runs: ${FAILED} (each had 2 attempts)."
echo "       Fixed freeze: ${FREEZE}"
echo "       Epoch sweep:  ${EPOCHS_ARR[*]}"
echo "============================================================"
exit 0
