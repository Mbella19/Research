# Live Paper-Trading Guide — v2 + v3 on MT5 (D-032)

**Architecture:** all trading intelligence runs in this repo's Python (the
exact frozen pipeline the backtests validated). Each MT5 terminal has a thin
Wine-side executor that only exports bars/status/deals and executes order
files. Parity gate: `pytest -m slow` (replay = engine, D-032b).

| policy | terminal | account | magics (S1/S2) |
|---|---|---|---|
| v2 | MetaTrader 5 copy1 | Pepperstone demo **62130224** | 622001 / 622002 |
| v3 | MetaTrader 5 copy2 | Pepperstone demo **62130225** | 623001 / 623002 |

## Daily operation

```
scripts/live_up.sh       # start 2 executors + 2 brains (terminals must be open & logged in)
scripts/live_status.sh   # health, equity, open positions, last loop lines
scripts/live_down.sh     # stop bridge processes (never touches terminals/copier)
```

**Requirements while trading:**
- Both MT5 terminals OPEN and logged into their accounts (attach-only: the
  bridge refuses to launch terminals or store passwords; if a terminal logs
  out, log in manually in the MT5 window).
- **AutoTrading button ON in BOTH terminals** (MT5 rejects API orders with
  retcode 10027 otherwise; the brain treats it as MISSED entries + loud log,
  never a halt).
- Mac awake and plugged in (each brain spawns `caffeinate -i`; don't close
  the lid).
- **Never trade manually on these two accounts** — broker equity feeds the
  sizing; foreign positions trigger the orphan halt.

## What the brain does (per policy, engine-parity)

- S1: decision on every completed 5m bar (full-history feature rebuild,
  ~12–18s — bit-identical to training); market entry with SL/TP anchored to
  the actual fill (±1.5/3.0 ATR); exits: SL/TP on broker, 240-bars horizon,
  23:30 force-flat; guards exactly as engine.py:147-166.
- S2: 23:00 entry when yesterday's close > SMA50 (full-history gate), exit
  next trading day 16:30; v3 adds the 5×dailyATR catastrophe stop + vol
  de-risk sizing; v2 runs stopless (its backtest parity).
- MISSED rule: a decision older than 90s is skipped, never chased.
  Stale feed ⇒ no entries; exits still fire; EARLY_FLAT after 21:00.
- Restart-safe: state.json + broker reconciliation (adopt / finalize from
  deals / orphan-halt). Restarting the brain mid-position is safe.

## Files (per policy, runs/live/<policy>/)

`decisions.csv` every 5m evaluation · `trades.csv` closed trades with real
swap/commission · `equity.csv` · `loop.out`/`executor.log` · `state.json` ·
`HALT` — create this file to make the brain flatten everything and stop
entering (delete it + restart to resume).

## Weekly referee (weekends)

```
.venv/bin/python -m daytrader.cli live-referee --policy v2
.venv/bin/python -m daytrader.cli live-referee --policy v3
```
Recomputes every decision from full history (side-equality vs decisions.csv)
+ trade summary. Any side mismatch = investigate before continuing.
The FWD-01 pick rule stands: ≥2–3 months, higher forward Sharpe wins unless
halt-flagged; ties → v3.

## Halt rules (armed, D-032)

Auto: 3 consecutive REAL order rejections · orphan position · (manual
check) S1 expectancy LB95 < 0 with n ≥ 20 · S2 bootstrap LB < −5e-4 ·
equity −8% from start · 30-min session bar gap. HALT file any time.

## Known live-vs-backtest differences (all pre-registered)

Real spread/slippage (the point of the test) · broker lot step 0.1 (engine
assumed 0.01; R-neutral) · 0.1-price rounding on SL/TP · real S2 swap
(alert if > 2.5bp/night equivalent) · samebar-settle class (~4.7% of S1
entries land ≤1 bar late after an intrabar stop cascade; D-032b) · single
shared account equity per policy (both sleeves compound together).

## Troubleshooting

**status.json stale but executor pid alive** → half-dead MT5 session after
a server drop (D-032d). Build 2026-07-03b auto-reattaches after ~30s of
silent-None API replies; if it ever persists > 5 min, `scripts/live_down.sh
&& scripts/live_up.sh` (safe any time — brains are restart-proof).
One-shot check of the AutoTrading button / connection without touching the
bridge: `WINEPREFIX="$HOME/Library/Application Support/net.metaquotes.wine.metatrader5" \
"/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64" \
"C:\\Python312\\python.exe" "C:\\daytrader\\probe_trade_allowed.py" \
"C:\\Program Files\\MetaTrader 5 copy1\\terminal64.exe"` (or copy2).

## iCloud warning

If “Desktop & Documents” sync with *Optimize Mac Storage* is enabled, macOS
can evict `data/live/*.parquet` / `runs/live/*`. Disable optimize-storage
for this Mac, or move those dirs off Desktop (symlink) if evictions appear.
