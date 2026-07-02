#!/bin/zsh
# Overnight pipeline chain — each stage logs to runs/<stage>_*/run.log.
# Aborts on first failure so a broken stage can't poison downstream artifacts.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== $(date) synth gate ==="
$PY -m daytrader.cli train-lgbm --stage synth
echo "=== $(date) lgbm final ==="
$PY -m daytrader.cli train-lgbm --stage final
echo "=== $(date) tcn cv ==="
$PY -m daytrader.cli train-tcn --stage cv
echo "=== $(date) tcn final ==="
$PY -m daytrader.cli train-tcn --stage final
echo "=== $(date) champion ==="
$PY -m daytrader.cli champion
echo "=== $(date) CHAIN DONE ==="
