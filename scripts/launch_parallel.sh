#!/usr/bin/env bash
# =============================================================================
# launch_parallel.sh — Run atlas + camel concurrently on a single MI300X
# =============================================================================
# Jobs are launched with nohup so they SURVIVE SSH disconnects.
# Both jobs share the GPU through ROCm's HIP runtime.
# Total VRAM: ~22.5 GB of 192 GB  (atlas 20 GB + camel 2.5 GB)
#
# Job layout:
#   atlas  : 201M params, ~20.0 GB VRAM, batch 256, ~98 steps/s
#   camel  :  25M params,  ~2.5 GB VRAM, batch 128, ~49 steps/s
#
# Checkpointing:
#   Each job auto-resumes from its latest.pt if it exists.
#   best.pt  — lowest val BPD ever seen
#   latest.pt — most recent checkpoint (always current)
#   step_XXXXXXX.pt — periodic snapshots
#
# Usage:
#   ./scripts/launch_parallel.sh            # launch atlas + camel (default)
#   ./scripts/launch_parallel.sh atlas      # atlas only
#   ./scripts/launch_parallel.sh camel      # camel only
#
# Logs:
#   logs/parallel_<YYYYMMDD_HHMMSS>/atlas.log
#   logs/parallel_<YYYYMMDD_HHMMSS>/camel.log
#   logs/parallel_<YYYYMMDD_HHMMSS>/atlas.pid
#   logs/parallel_<YYYYMMDD_HHMMSS>/camel.pid
#
# Prerequisites:
#   - Run both download pipelines first
#   - Python env must be activated:
#       source vortex-codec/.venv/bin/activate
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ── Timestamp log directory ───────────────────────────────────────────────────
LOG_DIR="logs/parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[launch] Log directory: $LOG_DIR"

# ── Job definitions ───────────────────────────────────────────────────────────
declare -A JOB_CONFIG=(
  [atlas]="configs/amd_mi300x.yaml"
  [camel]="configs/camel_mi300x.yaml"
)

declare -A JOB_DATA=(
  [atlas]="experiments/atlas_experiment/data/atlas_train.bin"
  [camel]="experiments/camel_experiment/data/camel_train.bin"
)

declare -A JOB_LOG=(
  [atlas]="$LOG_DIR/atlas.log"
  [camel]="$LOG_DIR/camel.log"
)

# ── Which jobs to run ─────────────────────────────────────────────────────────
# Default: atlas + camel. Override by passing names as args.
if [[ $# -gt 0 ]]; then
  JOBS=("$@")
else
  JOBS=(atlas camel)
fi

# ── Guard: abort if any requested job is already running ─────────────────────
for JOB in "${JOBS[@]}"; do
  for PIDFILE in logs/parallel_*/"${JOB}.pid"; do
    [[ -f "$PIDFILE" ]] || continue
    RUNNING_PID=$(cat "$PIDFILE" 2>/dev/null) || continue
    if kill -0 "$RUNNING_PID" 2>/dev/null; then
      echo ""
      echo "[ERROR] Job '$JOB' is already running (PID $RUNNING_PID, pidfile $PIDFILE)."
      echo "        To stop it:  kill $RUNNING_PID"
      echo "        To stop all: kill \$(cat logs/parallel_*/${JOB}.pid) 2>/dev/null"
      echo "        Refusing to launch a second instance to avoid GPU contention."
      echo ""
      exit 1
    fi
  done
done

# ── Launch (nohup — survives SSH disconnect) ──────────────────────────────────
declare -A PIDS

for JOB in "${JOBS[@]}"; do
  CONFIG="${JOB_CONFIG[$JOB]:-}"
  DATA="${JOB_DATA[$JOB]:-}"
  LOG="${JOB_LOG[$JOB]:-}"

  if [[ -z "$CONFIG" ]]; then
    echo "[warn] Unknown job '$JOB', skipping."
    continue
  fi

  if [[ ! -f "$DATA" ]]; then
    echo "[warn] Data file not found for '$JOB': $DATA"
    echo "       Run the download.py for this experiment first; skipping."
    continue
  fi

  echo "[launch] Starting job '$JOB'  config=$CONFIG"
  echo "         log -> $LOG"
  nohup python -u scripts/train.py --config "$CONFIG" > "$LOG" 2>&1 &
  PIDS[$JOB]=$!
  echo "${PIDS[$JOB]}" > "${LOG%.log}.pid"
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "[error] No jobs launched (data files missing?). Exiting."
  exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Vortex-Codec parallel training — MI300X         ║"
echo "╠══════════════════════════════════════════════════╣"
printf "║  %-8s  PID %-8s  %-24s ║\n" "Job" "" "Log"
echo "╠══════════════════════════════════════════════════╣"
for JOB in "${!PIDS[@]}"; do
  printf "║  %-8s  %-12s  %-24s ║\n" \
    "$JOB" "${PIDS[$JOB]}" "$(basename "${JOB_LOG[$JOB]}")"
done
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Jobs are running detached. SSH disconnect will NOT kill them."
echo ""
echo "── Monitor ────────────────────────────────────────────"
echo "  tail -f $LOG_DIR/*.log"
echo "  watch -n2 rocm-smi"
echo ""
echo "── Check if still running ─────────────────────────────"
for JOB in "${!PIDS[@]}"; do
  echo "  kill -0 ${PIDS[$JOB]} && echo '$JOB alive' || echo '$JOB done'"
done
echo ""
echo "── Stop all jobs ──────────────────────────────────────"
printf "  kill"
for JOB in "${!PIDS[@]}"; do
  printf " %s" "${PIDS[$JOB]}"
done
echo ""
echo ""
echo "── Checkpoints ────────────────────────────────────────"
echo "  experiments/atlas_experiment/checkpoints/latest.pt"
echo "  experiments/camel_experiment/checkpoints/latest.pt"
echo "  (auto-resumed next time you run this script)"
echo "───────────────────────────────────────────────────────"
