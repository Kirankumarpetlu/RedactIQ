#!/usr/bin/env bash
# =============================================================================
# Stage 9 PII pipeline — one-command startup (macOS / Linux)
# =============================================================================
# Opens three terminal windows:
#   Terminal 1  CPU worker  : port 8003  (loads spaCy + GLiNER2 into memory)
#   Terminal 2  Gateway     : port 8080  (Swagger UI at /docs)
#   Terminal 3  ngrok tunnel: public HTTPS URL -> gateway
#
# Then waits for health, opens Swagger in Chrome, and prints the public link.
#
# Usage:
#   ./scripts/start_pipeline.sh                 # full stack + ngrok + browser
#   ./scripts/start_pipeline.sh --no-ngrok      # local only, no public tunnel
#   ./scripts/start_pipeline.sh --no-browser    # do not open Chrome
#   ./scripts/start_pipeline.sh --background    # no terminal windows; log to logs/
#
# Environment overrides:
#   PII_VENV=venv_pii        virtualenv directory to use
#   PII_WORKER_PORT=8003     CPU worker port
#   PII_GATEWAY_PORT=8080    gateway / Swagger port
#   PII_CONFIG_PATH=config/pipeline_config.yaml
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WORKER_PORT="${PII_WORKER_PORT:-8003}"
GATEWAY_PORT="${PII_GATEWAY_PORT:-8080}"
CONFIG_PATH="${PII_CONFIG_PATH:-config/pipeline_config.yaml}"
MODEL_DIR="models/gliner2-primary"
LOG_DIR="$PROJECT_ROOT/logs"

USE_NGROK=1
USE_BROWSER=1
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --no-ngrok)   USE_NGROK=0 ;;
    --no-browser) USE_BROWSER=0 ;;
    --background) BACKGROUND=1 ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  OK\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. Resolve the virtualenv
# -----------------------------------------------------------------------------
say "Resolving Python environment"
PY=""
for candidate in "${PII_VENV:-}" .venv_improved .venv venv_pii venv39 venv; do
  [ -z "$candidate" ] && continue
  if [ -x "$PROJECT_ROOT/$candidate/bin/python" ]; then
    if "$PROJECT_ROOT/$candidate/bin/python" -c "import fastapi, uvicorn, gliner2, presidio_analyzer, spacy" 2>/dev/null; then
      PY="$PROJECT_ROOT/$candidate/bin/python"
      ok "using $candidate ($("$PY" --version 2>&1))"
      break
    fi
  fi
done
[ -n "$PY" ] || die "No virtualenv with the full dependency set (fastapi, gliner2, presidio, spacy).
      Follow the setup steps in README.md, or set PII_VENV=<dir>."

# -----------------------------------------------------------------------------
# 2. Preflight: model weights, spaCy model, config, ports
# -----------------------------------------------------------------------------
say "Preflight checks"
"$PY" -m scripts.download_data_assets --group runtime --check-only \
  || die "Runtime data missing. Run: $PY -m scripts.download_data_assets --group runtime"
[ -f "$MODEL_DIR/model.safetensors" ] || die "GLiNER checkpoint missing at $MODEL_DIR.
      Provision it first:  $PY scripts/provision_models.py --gliner-revision <APPROVED_SHA>"
ok "GLiNER checkpoint present"

"$PY" -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null \
  || die "spaCy model en_core_web_sm not installed. Run: $PY -m spacy download en_core_web_sm"
ok "spaCy en_core_web_sm present"

[ -f "$CONFIG_PATH" ] || die "Config not found: $CONFIG_PATH"
ok "config $CONFIG_PATH"

for port in "$WORKER_PORT" "$GATEWAY_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    die "Port $port is already in use. Stop the process, or run ./scripts/stop_pipeline.sh"
  fi
done
ok "ports $WORKER_PORT and $GATEWAY_PORT free"

# -----------------------------------------------------------------------------
# 3. Terminal launcher (macOS Terminal.app, or common Linux emulators)
# -----------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

open_terminal() {
  # open_terminal <window-title> <shell-command> <logfile-basename>
  local title="$1" cmd="$2" logname="$3"

  if [ "$BACKGROUND" -eq 1 ]; then
    # shellcheck disable=SC2086
    nohup bash -lc "cd '$PROJECT_ROOT' && $cmd" >"$LOG_DIR/$logname.log" 2>&1 &
    ok "$title -> background (logs/$logname.log, pid $!)"
    return
  fi

  case "$(uname -s)" in
    Darwin)
      osascript >/dev/null <<OSA
tell application "Terminal"
  activate
  do script "cd '$PROJECT_ROOT' && clear && echo '=== $title ===' && $cmd"
end tell
OSA
      ok "$title -> new Terminal window"
      ;;
    Linux)
      local launcher=""
      for t in gnome-terminal konsole xfce4-terminal xterm; do
        command -v "$t" >/dev/null 2>&1 && launcher="$t" && break
      done
      if [ -z "$launcher" ]; then
        nohup bash -lc "cd '$PROJECT_ROOT' && $cmd" >"$LOG_DIR/$logname.log" 2>&1 &
        ok "$title -> background (no terminal emulator found; logs/$logname.log)"
      else
        case "$launcher" in
          gnome-terminal) gnome-terminal --title="$title" -- bash -lc "cd '$PROJECT_ROOT' && $cmd; exec bash" & ;;
          konsole)        konsole -p tabtitle="$title" -e bash -lc "cd '$PROJECT_ROOT' && $cmd; exec bash" & ;;
          xfce4-terminal) xfce4-terminal --title="$title" -e "bash -lc \"cd '$PROJECT_ROOT' && $cmd; exec bash\"" & ;;
          xterm)          xterm -T "$title" -e bash -lc "cd '$PROJECT_ROOT' && $cmd; exec bash" & ;;
        esac
        ok "$title -> new $launcher window"
      fi
      ;;
    *) die "Unsupported OS $(uname -s). Use start_pipeline.bat on Windows." ;;
  esac
}

wait_for_health() {
  # wait_for_health <url> <label> <timeout-seconds>
  local url="$1" label="$2" timeout="${3:-300}" elapsed=0
  printf '    waiting for %s ' "$label"
  while ! curl -sf "$url" >/dev/null 2>&1; do
    sleep 2; elapsed=$((elapsed + 2))
    printf '.'
    if [ "$elapsed" -ge "$timeout" ]; then
      printf '\n'; die "$label did not come up within ${timeout}s. Check its terminal window."
    fi
  done
  printf ' ready (%ss)\n' "$elapsed"
}

# -----------------------------------------------------------------------------
# 4. Terminal 1 — CPU worker (loads the models; slowest step)
# -----------------------------------------------------------------------------
say "Terminal 1: CPU worker on port $WORKER_PORT"
open_terminal "TERMINAL 1: CPU WORKER (port $WORKER_PORT)" \
  "PII_SERVICE_ROLE=cpu-worker PII_API_PORT=$WORKER_PORT PII_CONFIG_PATH=$CONFIG_PATH '$PY' -m pii_pipeline" \
  "worker"
wait_for_health "http://localhost:$WORKER_PORT/health" "CPU worker (loading GLiNER ~1.2GB)" 600

# -----------------------------------------------------------------------------
# 5. Terminal 2 — gateway / Swagger
# -----------------------------------------------------------------------------
say "Terminal 2: gateway on port $GATEWAY_PORT"
open_terminal "TERMINAL 2: GATEWAY + SWAGGER (port $GATEWAY_PORT)" \
  "PII_SERVICE_ROLE=gateway PII_API_PORT=$GATEWAY_PORT PII_ENABLED_WORKERS=cpu_worker PII_CONFIG_PATH=$CONFIG_PATH '$PY' -m pii_pipeline" \
  "gateway"
wait_for_health "http://localhost:$GATEWAY_PORT/health" "gateway" 120

# -----------------------------------------------------------------------------
# 6. Warm the lazily-built stages (IndicNER + medical recognizers)
# -----------------------------------------------------------------------------
say "Warming IndicNER and the medical recognizers (first request builds them)"
curl -s -X POST "http://localhost:$GATEWAY_PORT/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text":"Patient John Doe, DOB 1985-04-12, NPI 1234567893.","language":"en","gliner_threshold":0.95}' \
  >/dev/null || die "Warmup request failed. Check the worker terminal."
ok "pipeline warm — all stages resident"

# -----------------------------------------------------------------------------
# 7. Terminal 3 — ngrok
# -----------------------------------------------------------------------------
PUBLIC_URL=""
if [ "$USE_NGROK" -eq 1 ]; then
  if ! command -v ngrok >/dev/null 2>&1; then
    printf '\033[1;33m  WARN\033[0m ngrok not installed; skipping the public tunnel.\n'
  else
    say "Terminal 3: ngrok tunnel -> localhost:$GATEWAY_PORT"
    open_terminal "TERMINAL 3: NGROK TUNNEL (-> $GATEWAY_PORT)" \
      "ngrok http $GATEWAY_PORT --log=stdout" "ngrok"
    printf '    waiting for tunnel '
    for _ in $(seq 1 30); do
      PUBLIC_URL="$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null \
        | "$PY" -c 'import sys,orjson;d=orjson.loads(sys.stdin.buffer.read());print(d["tunnels"][0]["public_url"] if d.get("tunnels") else "")' 2>/dev/null || true)"
      [ -n "$PUBLIC_URL" ] && break
      sleep 1; printf '.'
    done
    [ -n "$PUBLIC_URL" ] && printf ' up\n' || printf '\n    (tunnel not ready; check Terminal 3)\n'
  fi
fi

# -----------------------------------------------------------------------------
# 8. Open Swagger in Chrome
# -----------------------------------------------------------------------------
if [ "$USE_BROWSER" -eq 1 ]; then
  say "Opening Swagger UI in Chrome"
  if [ "$(uname -s)" = "Darwin" ]; then
    open -a "Google Chrome" "http://localhost:$GATEWAY_PORT/docs" 2>/dev/null \
      || open "http://localhost:$GATEWAY_PORT/docs"
    [ -n "$PUBLIC_URL" ] && open -a "Google Chrome" "$PUBLIC_URL/docs" 2>/dev/null || true
  else
    (google-chrome "http://localhost:$GATEWAY_PORT/docs" >/dev/null 2>&1 &) \
      || (xdg-open "http://localhost:$GATEWAY_PORT/docs" >/dev/null 2>&1 &) || true
    [ -n "$PUBLIC_URL" ] && (google-chrome "$PUBLIC_URL/docs" >/dev/null 2>&1 &) || true
  fi
fi

# -----------------------------------------------------------------------------
# 9. Summary
# -----------------------------------------------------------------------------
cat <<SUMMARY

------------------------------------------------------------------
 Stage 9 PII pipeline is up
------------------------------------------------------------------
  Swagger UI (local) : http://localhost:$GATEWAY_PORT/docs
  Health             : http://localhost:$GATEWAY_PORT/health
  Live metrics       : http://localhost:$GATEWAY_PORT/pipeline/metrics
  CPU worker         : http://localhost:$WORKER_PORT/health
SUMMARY
if [ -n "$PUBLIC_URL" ]; then
cat <<SUMMARY
  Public Swagger     : $PUBLIC_URL/docs      <-- share this
  ngrok inspector    : http://127.0.0.1:4040
SUMMARY
fi
cat <<SUMMARY

  Stop everything    : ./scripts/stop_pipeline.sh
------------------------------------------------------------------
SUMMARY
