"""Backtest engine known-answer tests: fills, costs, ambiguity, session flat."""
import numpy as np
import pandas as pd

from daytrader.backtest.engine import CostCfg, RiskCfg, run_backtest

COST = CostCfg(spread_mult=1.0, spread_floor_pts=10.0, slippage_pts=5.0,
               commission_per_lot_side=0.5, point_size=0.1)
RISK = RiskCfg(equity0=10_000.0, risk_per_trade=0.005, leverage_cap=50.0,
               max_trades_per_day=6, daily_stop_R=-3.0, value_per_unit=1.0,
               min_lot=0.01, lot_step=0.01, max_lot=1000.0)


def _mk_day(price: float = 100.0):
    ts = pd.date_range("2025-03-03 01:00", "2025-03-03 23:58", freq="1min")
    n = len(ts)
    df = pd.DataFrame({
        "ts": ts,
        "open": np.full(n, price), "high": np.full(n, price),
        "low": np.full(n, price), "close": np.full(n, price),
        "tickvol": np.full(n, 50, dtype=np.int32),
        "spread_pts": np.full(n, 10.0, dtype=np.float32),
    })
    return df


def _sig_at(df1m, hhmm: str, side: int, atr: float = 2.0):
    t = pd.Timestamp(f"2025-03-03 {hhmm}")
    return pd.DataFrame({
        "ts": [t], "avail_ts": [t + pd.Timedelta(minutes=5)],
        "day": [t.normalize()], "side": [side], "atr_abs": [atr],
    })


def test_long_tp_exact_fill_and_costs():
    df = _mk_day()
    i = df.index[df["ts"] == "2025-03-03 10:30"][0]
    df.loc[i, "high"] = 105.6  # TP touch
    sig = _sig_at(df, "10:00", +1)
    res = run_backtest(df, sig, COST, RISK, tp_atr=2.0, sl_atr=1.2)
    t = res["trades"]
    assert len(t) == 1
    # entry = open 100 + spread 1.0 + slip 0.5 = 101.5 (buy at ask); TP = 105.5
    assert np.isclose(t["entry"].iloc[0], 101.5)
    assert t["reason"].iloc[0] == "tp"
    assert np.isclose(t["exit"].iloc[0], 105.5)
    # risk $50 / (2.4 price × $1) = 20.83 lots; pnl = 4×20.83 − 2×0.5×20.83
    assert np.isclose(t["lots"].iloc[0], 20.83)
    assert np.isclose(t["pnl"].iloc[0], 4 * 20.83 - 1.0 * 20.83, atol=1e-6)


def test_same_bar_ambiguity_resolves_to_sl():
    df = _mk_day()
    i = df.index[df["ts"] == "2025-03-03 10:20"][0]
    df.loc[i, "high"] = 106.0   # both barriers inside one 1m bar
    df.loc[i, "low"] = 98.0
    sig = _sig_at(df, "10:00", +1)
    res = run_backtest(df, sig, COST, RISK, tp_atr=2.0, sl_atr=1.2)
    t = res["trades"]
    assert t["reason"].iloc[0] == "sl"
    # SL = 101.5 − 2.4 = 99.1, exit with slip → 98.6
    assert np.isclose(t["exit"].iloc[0], 99.1 - 0.5)
    assert t["pnl"].iloc[0] < 0


def test_no_entry_after_cutoff_and_forced_flat():
    df = _mk_day()
    late = _sig_at(df, "22:50", +1)      # after no_entry_after 22:45 → blocked
    res = run_backtest(df, late, COST, RISK)
    assert len(res["trades"]) == 0

    sig = _sig_at(df, "22:00", +1)       # no barrier ever hit → flat at 23:30
    res = run_backtest(df, sig, COST, RISK)
    t = res["trades"]
    assert len(t) == 1
    assert t["reason"].iloc[0] == "flat"
    assert t["exit_ts"].iloc[0] == pd.Timestamp("2025-03-03 23:30")
    # flat exit sells at bid − slip = 99.5
    assert np.isclose(t["exit"].iloc[0], 99.5)


def test_short_pays_spread_on_cover():
    df = _mk_day()
    i = df.index[df["ts"] == "2025-03-03 11:00"][0]
    df.loc[i, "low"] = 94.0
    sig = _sig_at(df, "10:00", -1)
    res = run_backtest(df, sig, COST, RISK, tp_atr=2.0, sl_atr=1.2)
    t = res["trades"]
    # entry = 100 − slip 0.5 = 99.5 (sell at bid); TP = 99.5 − 4 = 95.5,
    # cover at ask: low+spread = 95.0 ≤ 95.5 → filled at 95.5
    assert t["reason"].iloc[0] == "tp"
    assert np.isclose(t["entry"].iloc[0], 99.5)
    assert np.isclose(t["exit"].iloc[0], 95.5)
