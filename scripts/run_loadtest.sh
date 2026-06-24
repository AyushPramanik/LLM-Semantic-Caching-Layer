#!/usr/bin/env bash
# Run a headless Locust benchmark (~2000+ requests) and save reports.
set -euo pipefail

HOST="${1:-http://localhost:8000}"
USERS="${USERS:-50}"
SPAWN_RATE="${SPAWN_RATE:-10}"
RUNTIME="${RUNTIME:-1m}"

REPORT_DIR="$(dirname "$0")/../loadtests/reports"
mkdir -p "$REPORT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "Load testing ${HOST} (users=${USERS}, spawn=${SPAWN_RATE}, time=${RUNTIME})"
locust -f loadtests/locustfile.py \
  --host "$HOST" \
  --headless -u "$USERS" -r "$SPAWN_RATE" -t "$RUNTIME" \
  --csv "$REPORT_DIR/bench-$STAMP" \
  --html "$REPORT_DIR/bench-$STAMP.html" \
  --only-summary

echo "Reports written to $REPORT_DIR/bench-$STAMP.*"
