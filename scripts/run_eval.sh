#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — Evaluate both trained models on their test sets
# =============================================================================
# Runs after training is complete. Produces:
#   - Live comparison tables (Vortex vs gzip/zlib/lzma/zstd)
#   - results/atlas_results.json    ← paste into paper tables
#   - results/camel_results.json
#
# Usage:
#   ./scripts/run_eval.sh                     # eval both (default)
#   ./scripts/run_eval.sh atlas               # atlas only
#   ./scripts/run_eval.sh camel               # camel only
#   ./scripts/run_eval.sh atlas --full        # evaluate on FULL test set (slow)
#
# The --full flag evaluates Vortex on the entire test file.
# Without it, Vortex and baselines both run on a 50 MB sample (fast, ~5 min).
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

FULL=""
JOBS=()
for arg in "$@"; do
  case "$arg" in
    --full) FULL="--full-vortex" ;;
    atlas|camel) JOBS+=("$arg") ;;
    *) echo "[warn] Unknown arg '$arg', ignoring." ;;
  esac
done
[[ ${#JOBS[@]} -eq 0 ]] && JOBS=(atlas camel)

mkdir -p results

run_eval() {
  local JOB="$1"
  local MODEL CONFIG TEST_BIN OUT_JSON

  case "$JOB" in
    atlas)
      MODEL="experiments/atlas_experiment/checkpoints/best.pt"
      CONFIG="configs/amd_mi300x.yaml"
      TEST_BIN="experiments/atlas_experiment/data/atlas_test.bin"
      OUT_JSON="results/atlas_results.json"
      ;;
    camel)
      MODEL="experiments/camel_experiment/checkpoints/best.pt"
      CONFIG="configs/camel_mi300x.yaml"
      TEST_BIN="experiments/camel_experiment/data/camel_test.bin"
      OUT_JSON="results/camel_results.json"
      ;;
  esac

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Evaluating: $JOB"
  echo "══════════════════════════════════════════════════════════════"

  if [[ ! -f "$MODEL" ]]; then
    echo "[error] Checkpoint not found: $MODEL"
    echo "        Training may still be running — check with:"
    echo "        tail -f logs/parallel_*/${JOB}.log"
    return 1
  fi

  if [[ ! -f "$TEST_BIN" ]]; then
    echo "[error] Test data not found: $TEST_BIN"
    echo "        Run the download pipeline first."
    return 1
  fi

  echo "  Model  : $MODEL"
  echo "  Data   : $TEST_BIN  ($(du -sh "$TEST_BIN" | cut -f1))"
  echo "  Output : $OUT_JSON"
  echo ""

  python scripts/evaluate.py \
    --model    "$MODEL"   \
    --data     "$TEST_BIN" \
    --config   "$CONFIG"  \
    --out-json "$OUT_JSON" \
    --sample-mb 200 \
    $FULL
}

FAILED=0
for JOB in "${JOBS[@]}"; do
  run_eval "$JOB" || FAILED=$((FAILED + 1))
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Results saved:"
for JOB in "${JOBS[@]}"; do
  case "$JOB" in
    atlas) echo "    results/atlas_results.json" ;;
    camel) echo "    results/camel_results.json" ;;
  esac
done
echo ""
echo "  Copy to local machine:"
echo "    scp root@<mi300x-ip>:~/vortex-codec/results/*.json ."
echo "══════════════════════════════════════════════════════════════"

[[ $FAILED -eq 0 ]] || exit 1
