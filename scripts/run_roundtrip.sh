#!/usr/bin/env bash
# =============================================================================
# run_roundtrip.sh — End-to-end lossless verification
# =============================================================================
# Compresses a 10 MB chunk of the test set, decompresses it, and verifies
# byte-for-byte identity with the original. This proves the arithmetic
# coding implementation is lossless — essential for the paper's claims.
#
# Usage:
#   ./scripts/run_roundtrip.sh              # both experiments
#   ./scripts/run_roundtrip.sh atlas
#   ./scripts/run_roundtrip.sh camel
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

JOBS=("$@")
[[ ${#JOBS[@]} -eq 0 ]] && JOBS=(atlas camel)

CHUNK_MB=10   # small enough for a quick test

run_roundtrip() {
  local JOB="$1"
  local MODEL CONFIG TEST_BIN

  case "$JOB" in
    atlas)
      MODEL="experiments/atlas_experiment/checkpoints/best.pt"
      CONFIG="configs/amd_mi300x.yaml"
      TEST_BIN="experiments/atlas_experiment/data/atlas_test.bin"
      ;;
    camel)
      MODEL="experiments/camel_experiment/checkpoints/best.pt"
      CONFIG="configs/camel_mi300x.yaml"
      TEST_BIN="experiments/camel_experiment/data/camel_test.bin"
      ;;
  esac

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Round-trip test: $JOB  (${CHUNK_MB} MB sample)"
  echo "══════════════════════════════════════════════════════════════"

  if [[ ! -f "$MODEL" ]]; then
    echo "[skip] Checkpoint not found: $MODEL"; return 0
  fi
  if [[ ! -f "$TEST_BIN" ]]; then
    echo "[skip] Test data not found: $TEST_BIN"; return 0
  fi

  TMPDIR="$(mktemp -d)"
  trap "rm -rf $TMPDIR" EXIT

  CHUNK_BYTES=$(( CHUNK_MB * 1024 * 1024 ))
  SAMPLE="$TMPDIR/sample.bin"
  COMPRESSED="$TMPDIR/sample.vxc"
  DECOMPRESSED="$TMPDIR/sample_out.bin"

  # Extract a small sample (record-aligned for atlas=102B, camel=44B)
  echo "  Extracting ${CHUNK_MB} MB chunk from $TEST_BIN ..."
  python3 - <<PYEOF
import sys
path = "$TEST_BIN"
out  = "$SAMPLE"
record_bytes = 102 if "$JOB" == "atlas" else 44
target = $CHUNK_BYTES
with open(path, "rb") as f:
    raw = f.read(target)
# Trim to record boundary
n = (len(raw) // record_bytes) * record_bytes
with open(out, "wb") as f:
    f.write(raw[:n])
print(f"  Sample : {n:,} bytes  ({n//record_bytes:,} records)")
PYEOF

  echo "  Compressing ..."
  python scripts/compress.py \
    --model  "$MODEL"      \
    --input  "$SAMPLE"     \
    --output "$COMPRESSED" \
    --config "$CONFIG"

  CMP_SIZE=$(wc -c < "$COMPRESSED")
  ORI_SIZE=$(wc -c < "$SAMPLE")
  echo "  Compressed: $(( ORI_SIZE / 1024 )) KB -> $(( CMP_SIZE / 1024 )) KB"

  echo "  Decompressing ..."
  python scripts/decompress.py \
    --model  "$MODEL"       \
    --input  "$COMPRESSED"  \
    --output "$DECOMPRESSED" \
    --config "$CONFIG"

  echo "  Verifying byte-for-byte identity ..."
  if cmp -s "$SAMPLE" "$DECOMPRESSED"; then
    echo "  ✓ PASS — round-trip is lossless for $JOB"
  else
    echo "  ✗ FAIL — output differs from input!"
    echo "    original   : $ORI_SIZE bytes"
    echo "    decompressed: $(wc -c < "$DECOMPRESSED") bytes"
    exit 1
  fi

  rm -rf "$TMPDIR"
  trap - EXIT
}

for JOB in "${JOBS[@]}"; do
  run_roundtrip "$JOB"
done

echo ""
echo "  All round-trip tests passed."
