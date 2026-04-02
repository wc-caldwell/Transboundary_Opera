#!/usr/bin/env bash
# Iterates over every aquifer/frame directory and invokes process_frame.py
# once per frame as a **separate process**, so each frame gets a clean
# memory space and a kernel crash only loses that one frame.
#
# Usage:
#   bash run_all_frames.sh /path/to/data 20200101 20231231
#
# Optional env vars:
#   SKIP_FRAMES   -- space-separated frame IDs to skip (default: "3067")
#   LOG_DIR       -- where to write per-frame logs (default: ./logs)

set -euo pipefail

DATA_DIR="${1:?Usage: $0 <data-dir> <start-date> <end-date>}"
START_DATE="${2:?Usage: $0 <data-dir> <start-date> <end-date>}"
END_DATE="${3:?Usage: $0 <data-dir> <start-date> <end-date>}"

SKIP_FRAMES="${SKIP_FRAMES:-3067}"
LOG_DIR="${LOG_DIR:-./logs}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$LOG_DIR"

echo "============================================"
echo " Data dir:    $DATA_DIR"
echo " Date range:  $START_DATE -> $END_DATE"
echo " Skip frames: $SKIP_FRAMES"
echo " Logs:        $LOG_DIR"
echo "============================================"
echo ""

# Build the list of (aquifer, frame) pairs
PAIRS=()
for aquifer_dir in "$DATA_DIR"/*/; do
    [ -d "$aquifer_dir" ] || continue
    aquifer="$(basename "$aquifer_dir")"
    for frame_dir in "$aquifer_dir"/*/; do
        [ -d "$frame_dir" ] || continue
        frame="$(basename "$frame_dir")"
        PAIRS+=("$aquifer/$frame")
    done
done

echo "Found ${#PAIRS[@]} aquifer/frame pairs."
echo ""

SUCCESS=0
FAIL=0

for pair in "${PAIRS[@]}"; do
    aquifer="${pair%%/*}"
    frame="${pair##*/}"
    logfile="$LOG_DIR/${aquifer}__${frame}.log"

    echo ">>> [$((SUCCESS + FAIL + 1))/${#PAIRS[@]}] $aquifer / $frame"

    if python3 "$SCRIPT_DIR/process_frame.py" \
        --data-dir "$DATA_DIR" \
        --aquifer "$aquifer" \
        --frame "$frame" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --skip-frames $SKIP_FRAMES \
        2>&1 | tee "$logfile"; then
        ((SUCCESS++)) || true
    else
        echo "!!! FAILED: $aquifer / $frame (see $logfile)"
        ((FAIL++)) || true
    fi
done

echo ""
echo "============================================"
echo " Finished.  Success: $SUCCESS  Failed: $FAIL"
echo "============================================"
