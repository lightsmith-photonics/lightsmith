#!/usr/bin/env bash
# Run the full-paper suite in the background, resuming after any crash.
#   bash benchmarks/run_full_paper.sh [out_dir]
# Each restart keeps finished problems (run.py --resume); a problem that kills the runner twice is skipped.
# Run it again with the same out_dir after a reboot. Only one driver may use a directory at a time.
set -u
cd "$(dirname "$0")/.."
OUT=${1:-benchmarks/results/$(date +%Y%m%d-%H%M%S)-full-paper}
mkdir -p "$OUT"
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  echo "[driver] $OUT/.lock exists: another driver is running (remove it if not)" >&2
  exit 2
fi
trap 'rmdir "$OUT/.lock"' EXIT
for attempt in $(seq 1 20); do
  echo "[driver] attempt $attempt $(date -Iseconds)" | tee -a "$OUT/driver.log"
  uv run --no-sync python benchmarks/run.py --suite full-paper --devices cpu,gpu --max-run-s 90 \
    --out "$OUT" --resume >>"$OUT/run.log" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "[driver] finished $(date -Iseconds)" | tee -a "$OUT/driver.log"
    exit 0
  fi
  echo "[driver] runner exited with $status; restarting in 30 s" | tee -a "$OUT/driver.log"
  sleep 30
done
exit 1
