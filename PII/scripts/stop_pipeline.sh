#!/usr/bin/env bash
# =============================================================================
# Stop every Stage 9 pipeline process started by start_pipeline.sh
# =============================================================================
set -uo pipefail

WORKER_PORT="${PII_WORKER_PORT:-8003}"
GATEWAY_PORT="${PII_GATEWAY_PORT:-8080}"

stop_port() {
  local port="$1" label="$2"
  local pids
  pids="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    echo "  $label (port $port): not running"
    return
  fi
  echo "  $label (port $port): stopping pid(s) $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 2
  pids="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  # shellcheck disable=SC2086
  [ -n "$pids" ] && { echo "    still alive, sending SIGKILL"; kill -9 $pids 2>/dev/null || true; }
}

echo "Stopping Stage 9 pipeline..."
stop_port "$GATEWAY_PORT" "gateway"
stop_port "$WORKER_PORT"  "cpu worker"

if pgrep -f "ngrok http" >/dev/null 2>&1; then
  echo "  ngrok: stopping"
  pkill -f "ngrok http" 2>/dev/null || true
else
  echo "  ngrok: not running"
fi

echo "Done. The terminal windows stay open so you can read the logs; close them yourself."
