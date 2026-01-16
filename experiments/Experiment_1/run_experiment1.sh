#!/usr/bin/env bash
# Experiment 1: run all 8 runs (2 models × F0–F3), with per-run retry (max 2 attempts).
# Colab-compatible: paste into a Colab cell (prefix with !bash run_all_e1.sh) or run directly.

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
RUNNER="./runOneTest.py"

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
run_one() {
  local model="$1"
  local freeze="$2"

  echo "============================================================"
  echo "[RUN] model=${model} freeze=${freeze} epochs=${EPOCHS} imgsz=${IMGSZ} seed=${SEED}"
  echo "============================================================"

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
