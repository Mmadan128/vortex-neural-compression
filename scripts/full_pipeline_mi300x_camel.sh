#!/usr/bin/env bash
# Full CAMEL pipeline for AMD MI300X (ROCm): setup -> download -> train -> eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/camel_mi300x.yaml}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_MB="${SAMPLE_MB:-200}"
VORTEX_MB="${VORTEX_MB:-$SAMPLE_MB}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FULL_VORTEX="${FULL_VORTEX:-0}"

LOG_DIR="${LOG_DIR:-logs/mi300x_camel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" results

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

if [[ "$SKIP_SETUP" != "1" ]]; then
  if [[ ! -x ".venv/bin/python" ]]; then
    echo "[setup] Creating virtual environment (.venv)"
    python3 -m venv .venv
    PYTHON=".venv/bin/python"
  fi

  echo "[setup] Installing dependencies"
  "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install -r requirements.txt
fi

echo "[info] Python: $PYTHON"
"$PYTHON" -c "import torch; print('[info] torch:', torch.__version__, 'cuda_available=', torch.cuda.is_available(), 'hip=', torch.version.hip)"

if command -v rocm-smi >/dev/null 2>&1; then
  echo "[info] rocm-smi snapshot"
  rocm-smi > "$LOG_DIR/rocm_smi_start.txt" 2>&1 || true
fi

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  echo "[step] CAMEL data download + extraction + split"
  "$PYTHON" experiments/camel_experiment/download.py --profile server --all-steps \
    2>&1 | tee "$LOG_DIR/download.log"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "[step] Training with $CONFIG"
  "$PYTHON" scripts/train.py --config "$CONFIG" --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/train.log"
fi

MODEL_PATH="experiments/camel_experiment/checkpoints/best.pt"
TEST_DATA="experiments/camel_experiment/data/camel_test.bin"
OUT_JSON="results/camel_mi300x_results.json"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[error] Missing checkpoint: $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$TEST_DATA" ]]; then
  echo "[error] Missing test data: $TEST_DATA"
  exit 1
fi

echo "[step] Evaluation / benchmark"
EVAL_ARGS=(
  scripts/evaluate.py
  --model "$MODEL_PATH"
  --data "$TEST_DATA"
  --config "$CONFIG"
  --device "$DEVICE"
  --sample-mb "$SAMPLE_MB"
  --vortex-mb "$VORTEX_MB"
  --batch-size "$EVAL_BATCH_SIZE"
  --out-json "$OUT_JSON"
)

if [[ "$FULL_VORTEX" == "1" ]]; then
  EVAL_ARGS+=(--full-vortex)
fi

"$PYTHON" "${EVAL_ARGS[@]}" 2>&1 | tee "$LOG_DIR/eval.log"

echo "[done] Full MI300X CAMEL pipeline completed"
echo "[done] Results JSON: $OUT_JSON"
echo "[done] Logs: $LOG_DIR"
