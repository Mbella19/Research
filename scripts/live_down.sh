#!/bin/zsh
# Stop the live bridge processes. NEVER touches wineserver or the terminals
# (the tv-mt5-copier shares them). Open positions are left untouched — use
# the HALT file (runs/live/<policy>/HALT) if you want the brain to flatten
# before stopping.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for pol in v2 v3; do
  for kind in loop executor; do
    f="$ROOT/runs/live/$pol/$kind.pid"
    if [ -f "$f" ]; then
      pid="$(cat "$f")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "stopped $pol $kind (pid $pid)"
      fi
      rm -f "$f"
    fi
  done
done
