# NAS100 Day-Trading AI

An instrument-agnostic intraday ML trading research stack. Current instrument:
NAS100 on MT5 1-minute exports. Decisions on 5m bars, exits resolved on the 1m
path, all costs stressed above realistic levels.

## The discipline (read first)

- **TRAINING** (real, 2020→2025-05) — all fitting and every hyperparameter /
  feature / geometry decision, via purged walk-forward CV with embargo.
- **VALIDATION** (real, 2025-06→12) — decision gate only. Every look is
  pre-registered and auto-appended to `notes/decisions.md` (VLOOK entries).
- **LOCKED OOS** (real, 2026-01→06) — `daytrader oos --confirm-single-shot`
  runs it **exactly once, ever** (guarded by `runs/OOS_EXECUTED.flag`).
- **SYNTHETIC** (5 universes, 2030–2035) — stress screen + optional
  regularizer. Can veto a recipe, can never select one.

## Pipeline

```
make setup                       # venv + deps (+ macOS hidden-flag guard)
daytrader ingest                 # MT5 TSV → parquet + integrity report
daytrader profile                # session profile + synth realism report
daytrader audit-synth            # synth↔real leakage audit
daytrader features --source all  # feature matrices (lookahead-tested)
daytrader labels --source all    # triple-barrier labels (1m-path checked)
daytrader benchmark              # rule-strategy baselines + B&H
daytrader train-lgbm --stage search|barriers|synth|final
daytrader train-tcn  --stage cv|final
daytrader champion               # stacking blend + champion selection
daytrader validate --look champion|threshold     # ledgered validation looks
daytrader robustness             # stress battery (one ledgered look)
daytrader freeze                 # immutable FINAL artifact record
daytrader oos --confirm-single-shot              # THE single shot
daytrader forward --csv <new export>             # paper-trade harness
make test                        # incl. the anti-lookahead gate
```

`scripts/overnight_chain.sh` chains the heavy stages.

## Retraining on another instrument

1. Copy the whole project directory; delete `data/`, `models/`, `runs/`,
   and `notes/decisions.md` VLOOK entries (fresh ledger).
2. Drop the new MT5 exports in place and edit **`config/instrument.yaml`**
   only: symbol, point size/value, session times, file paths, cost baseline.
3. Re-run the pipeline from `ingest`. Feature/label caches key themselves to
   the config, and every stage re-derives from the new data.

## Where to look when something is off

- `runs/<stage>_<ts>/run.log` — DEBUG log of every run.
- `runs/<stage>_<ts>/plots/` — training curves, calibration, importance,
  equity/DD, stress curves.
- `notes/findings.md` — the running lab notebook.
- `notes/decisions.md` — the append-only decision + validation-look ledger.

## v2 — Two-sleeve portfolio (current deployable)

S1 (frozen v1 intraday GBT) + S2 (gated overnight drift: long from 23:00 to
next 16:30 while yesterday's close > SMA50, lagged). Frozen in
`models/FINAL_FROZEN_V2.json`; full story in `notes/FINAL_REPORT.md`.

```bash
daytrader portfolio --stage grid          # S2 pre-registered 24-combo grid (training)
daytrader portfolio --stage oof           # honest portfolio OOF (training)
daytrader portfolio --stage validate --which s2|portfolio   # ledgered looks
daytrader portfolio --stage robustness    # cost/slip/swap/gate/synth battery
daytrader portfolio --stage freeze-v2     # immutable v2 artifact
daytrader portfolio --stage finaltest --confirm-single-shot # single shot (flag-gated)
daytrader signal  --csv <mt5 export>      # BOTH sleeves' live orders
daytrader forward --csv <mt5 export>      # per-sleeve divergence + halt flags
```

New instrument: swap the CSVs + edit `config/instrument.yaml`, then rerun the
whole pipeline INCLUDING the S2 grid — the overnight anomaly must re-earn its
place per instrument.

## v3 (current deployable)

S1 rebuilt: 142 features (adds daily-context / calendar / tape-character
blocks), champion via pre-registered arena (beat v1, geometry-B, EV-reg,
meta-label, TCN, blend). S2 + catastrophe stop (5×ATR) + one-sided vol
de-risk. Frozen: `models/FINAL_FROZEN_V3.json`. Report: `notes/FINAL_REPORT.md`.
`daytrader train-lgbm --stage search|final|evreg|arena` · `portfolio --stage
oof|validate|robustness|freeze-v3|fullhistory`.

## Live paper trading (v3.1, D-032)

Both frozen policies trade two Pepperstone 50k demo accounts automatically:
the exact validated pipeline decides on the Mac; thin Wine-side executors
(same mechanism as the resident tv-mt5-copier) export bars and execute
orders per terminal. Parity to the backtest engine is test-enforced
(`pytest -m slow`: replay==engine, D-032b). Operate with
`scripts/live_up.sh` / `live_status.sh` / `live_down.sh`; weekly
`daytrader live-referee --policy v2|v3`. Full guide: `notes/LIVE_GUIDE.md`.
