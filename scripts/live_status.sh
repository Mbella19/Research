#!/bin/zsh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DC="$HOME/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c"
for pol in v2 v3; do
  login=$([ "$pol" = "v2" ] && echo 62130224 || echo 62130225)
  echo "── $pol (login $login) ─────────────────────────────"
  for kind in executor loop; do
    f="$ROOT/runs/live/$pol/$kind.pid"
    if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
      echo "  $kind: RUNNING (pid $(cat "$f"))"
    else
      echo "  $kind: DOWN"
    fi
  done
  "$ROOT/.venv/bin/python" - << PYEOF
import json, time, pathlib
p = pathlib.Path("$DC/daytrader/$login/status.json")
if p.exists():
    st = json.loads(p.read_text())
    age = time.time() - p.stat().st_mtime
    print(f"  status: srv {st.get('server_time')} (file age {age:.0f}s) "
          f"equity {st.get('equity'):,} trade_allowed={st.get('trade_allowed')}")
    for pos in st.get("positions", []):
        print(f"  position: magic {pos['magic']} {'LONG' if pos['type']==1 else 'SHORT'} "
              f"{pos['volume']} lots @ {pos['price_open']} sl={pos['sl']} tp={pos['tp']}")
else:
    print("  status: no status.json yet")
PYEOF
  tail -2 "$ROOT/runs/live/$pol/loop.out" 2>/dev/null | sed 's/^/  loop: /'
done
