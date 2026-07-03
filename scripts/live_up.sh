#!/bin/zsh
# Start the live bridge: 2 Wine executors + 2 Mac brains (D-032).
# Terminals must ALREADY be running and logged in (attach-only policy).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$HOME/Library/Application Support/net.metaquotes.wine.metatrader5"
WINE="/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64"
DC="$PREFIX/drive_c"
PY="$ROOT/.venv/bin/python"

for t in "copy1" "copy2"; do
  if ! pgrep -f "MetaTrader 5 $t.terminal64.exe" > /dev/null; then
    echo "REFUSED: 'MetaTrader 5 $t' terminal is not running — open it first"
    exit 1
  fi
done

mkdir -p "$DC/daytrader/accounts" "$DC/daytrader/62130224" "$DC/daytrader/62130225"
cp "$ROOT/bridge/accounts/"*.json "$DC/daytrader/accounts/"
cp "$ROOT/bridge/mt5_executor_daytrader.py" "$DC/daytrader/"

for pol in v2 v3; do
  mkdir -p "$ROOT/runs/live/$pol"
  if [ -f "$ROOT/runs/live/$pol/executor.pid" ] && \
     kill -0 "$(cat "$ROOT/runs/live/$pol/executor.pid")" 2>/dev/null; then
    echo "$pol executor already running"
  else
    WINEPREFIX="$PREFIX" "$WINE" "C:\\Python312\\python.exe" \
      "C:\\daytrader\\mt5_executor_daytrader.py" \
      --config "C:\\daytrader\\accounts\\$pol.json" \
      >> "$ROOT/runs/live/$pol/executor.log" 2>&1 &
    echo $! > "$ROOT/runs/live/$pol/executor.pid"
    echo "$pol executor started (pid $!)"
  fi
done

sleep 6
for pol in v2 v3; do
  if [ -f "$ROOT/runs/live/$pol/loop.pid" ] && \
     kill -0 "$(cat "$ROOT/runs/live/$pol/loop.pid")" 2>/dev/null; then
    echo "$pol brain already running"
  else
    nohup "$PY" -m daytrader.cli live-loop --policy $pol $LOOP_ARGS \
      >> "$ROOT/runs/live/$pol/loop.out" 2>&1 &
    echo $! > "$ROOT/runs/live/$pol/loop.pid"
    echo "$pol brain started (pid $!) $LOOP_ARGS"
  fi
done
echo "live bridge up. status: scripts/live_status.sh"
