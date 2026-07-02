"""Sleeve-2 overnight executor known-answer tests: fills/costs, stop with
gap-through, weekend holds & swap nights, and the gate availability lag."""
import numpy as np
import pandas as pd

from daytrader.backtest.engine import CostCfg
from daytrader.portfolio.overnight import S2Params, daily_frame, gate_series, sleeve2_run

COST = CostCfg(spread_mult=1.0, spread_floor_pts=10.0, slippage_pts=0.0,
               commission_per_lot_side=0.0, point_size=0.1)  # spread = 1.0 px unit


def _mk_days(days, price_fn):
    """Sparse 1m frame: bars at 01:00, 12:00, 16:30, 23:00, 23:59 per day.
    price_fn(day_idx, minute) -> (open, high, low, close)."""
    rows = []
    for di, d in enumerate(days):
        for hhmm in ("01:00", "12:00", "16:30", "23:00", "23:59"):
            o, h, l, c = price_fn(di, hhmm)
            rows.append((pd.Timestamp(f"{d} {hhmm}"), o, h, l, c, 50, 10.0))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                       "tickvol", "spread_pts"])


def _flat(px):
    return lambda di, hhmm: (px, px, px, px)


def test_usc_window_entry_exit_prices():
    days = ["2024-01-01", "2024-01-02", "2024-01-03"]
    def px(di, hhmm):
        if di == 0:
            return (100.0, 100.0, 100.0, 100.0)
        if di == 1 and hhmm == "16:30":
            return (110.0, 110.0, 110.0, 110.0)
        return (105.0, 105.0, 105.0, 105.0)
    df = _mk_days(days, px)
    gate = pd.Series([True, False, False], index=pd.DatetimeIndex(days))
    out = sleeve2_run(df, S2Params(window="usC"), COST, gate_override=gate)
    t = out["trades"]
    assert len(t) == 1
    # entry: day1 23:00 open 100 + spread 1.0 = 101; exit: day2 16:30 open 110
    assert np.isclose(t["entry"].iloc[0], 101.0)
    assert np.isclose(t["exit"].iloc[0], 110.0)
    assert np.isclose(t["ret_net"].iloc[0], (110.0 - 101.0) / 101.0)
    assert t["reason"].iloc[0] == "window"
    assert t["nights"].iloc[0] == 1
    # settled on the EXIT day
    assert np.isclose(out["daily_ret"].loc["2024-01-02"], t["ret_net"].iloc[0])
    assert out["daily_ret"].loc["2024-01-01"] == 0.0


def test_postgap_window_same_day():
    days = ["2024-01-01", "2024-01-02"]
    def px(di, hhmm):
        if di == 1 and hhmm == "01:00":
            return (200.0, 200.0, 200.0, 200.0)
        if di == 1 and hhmm == "16:30":
            return (204.0, 204.0, 204.0, 204.0)
        return (200.0, 200.0, 200.0, 200.0)
    df = _mk_days(days, px)
    gate = pd.Series([False, True], index=pd.DatetimeIndex(days))
    out = sleeve2_run(df, S2Params(window="postgap"), COST, gate_override=gate)
    t = out["trades"]
    assert len(t) == 1
    assert str(t["entry_ts"].iloc[0]) == "2024-01-02 01:00:00"
    assert np.isclose(t["entry"].iloc[0], 201.0)     # 200 + spread 1.0
    assert np.isclose(t["exit"].iloc[0], 204.0)
    assert t["nights"].iloc[0] == 0


def test_stop_hit_and_gap_through():
    # 16 flat warm-up days give ATR14 = TR mean; TR=2 via daily range 99..101
    days = [str(d.date()) for d in pd.bdate_range("2024-02-01", periods=18)]
    def px(di, hhmm):
        if di < 16:
            if hhmm == "12:00":
                return (100.0, 101.0, 99.0, 100.0)   # daily TR = 2
            return (100.0, 100.0, 100.0, 100.0)
        if di == 16:                                  # entry evening
            return (100.0, 100.0, 100.0, 100.0)
        # di == 17: morning collapse — 01:00 bar opens BELOW the stop (gap-through)
        if hhmm == "01:00":
            return (93.0, 93.5, 92.0, 93.0)
        return (95.0, 95.5, 94.5, 95.0)
    df = _mk_days(days, px)
    gate = pd.Series([d == days[16] for d in days], index=pd.DatetimeIndex(days))
    out = sleeve2_run(df, S2Params(window="usC", stop_atr=2.5), COST, gate_override=gate)
    t = out["trades"]
    assert len(t) == 1
    # entry 101 (100+1); ATR14 = 2 ⇒ stop = 101 − 5 = 96; day-17 01:00 opens at 93 < 96
    assert t["reason"].iloc[0] == "stop"
    assert np.isclose(t["exit"].iloc[0], 93.0)       # gap-through fills at the open
    # same setup but low touches the stop without gapping: fills AT the level
    def px2(di, hhmm):
        if di < 16:
            if hhmm == "12:00":
                return (100.0, 101.0, 99.0, 100.0)
            return (100.0, 100.0, 100.0, 100.0)
        if di == 16:
            return (100.0, 100.0, 100.0, 100.0)
        if hhmm == "01:00":
            return (97.0, 97.5, 95.0, 97.0)          # low 95 ≤ 96, open 97 > 96
        return (98.0, 98.5, 97.5, 98.0)
    out2 = sleeve2_run(_mk_days(days, px2), S2Params(window="usC", stop_atr=2.5),
                       COST, gate_override=gate)
    assert out2["trades"]["reason"].iloc[0] == "stop"
    assert np.isclose(out2["trades"]["exit"].iloc[0], 96.0)


def test_weekend_hold_and_swap_nights():
    days = ["2024-03-01", "2024-03-04"]              # Friday → Monday
    df = _mk_days(days, _flat(100.0))
    gate = pd.Series([True, False], index=pd.DatetimeIndex(days))
    out = sleeve2_run(df, S2Params(window="usC"), COST, swap_bp_night=2.0,
                      gate_override=gate)
    t = out["trades"]
    assert len(t) == 1 and t["nights"].iloc[0] == 3   # Fri→Mon = 3 calendar nights
    base = (100.0 - 101.0) / 101.0                    # spread-only loss
    assert np.isclose(t["ret_net"].iloc[0], base - 3 * 2.0e-4)


def test_gate_availability_lag():
    """Gate on day D must use closes ≤ D−1: a crash ON day D still permits
    day-D entries and blocks day D+1."""
    days = pd.bdate_range("2023-01-02", periods=120)
    close = np.linspace(100.0, 220.0, 120)
    close[100:] = 50.0                                # crash at day index 100
    d = pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                     index=days)
    g = gate_series(d, S2Params(gate="sma50"))
    assert bool(g.iloc[100])                          # crash day: still on (lag)
    assert not bool(g.iloc[101])                      # blocked from next day
    assert not g.iloc[:50].any()                      # SMA warm-up: never on


def test_catastrophe_stop_fires_on_crash():
    """v3 risk package: the 5×ATR stop must fire on a synthetic overnight crash
    (and only then — P3 measured 0 fires on real training data)."""
    days = [str(d.date()) for d in pd.bdate_range("2024-02-01", periods=18)]
    def px(di, hhmm):
        if di < 16:
            if hhmm == "12:00":
                return (100.0, 101.0, 99.0, 100.0)   # ATR14 = 2 ⇒ stop = 101 − 10 = 91
            return (100.0, 100.0, 100.0, 100.0)
        if di == 16:
            return (100.0, 100.0, 100.0, 100.0)
        if hhmm == "01:00":                           # −12% overnight collapse
            return (88.0, 88.5, 86.0, 88.0)
        return (89.0, 89.5, 88.5, 89.0)
    df = _mk_days(days, px)
    gate = pd.Series([d == days[16] for d in days], index=pd.DatetimeIndex(days))
    out = sleeve2_run(df, S2Params(window="usC", stop_atr=5.0), COST, gate_override=gate)
    t = out["trades"]
    assert t["reason"].iloc[0] == "stop"
    assert np.isclose(t["exit"].iloc[0], 88.0)        # gap-through: open below 91


def test_derisk_exposure_one_sided_and_lagged():
    from daytrader.portfolio.overnight import derisk_exposure

    days = pd.bdate_range("2022-01-03", periods=320)
    rng = np.random.default_rng(7)
    ret = np.full(320, 0.001) + rng.normal(0, 0.004, 320)
    ret[290:] = rng.normal(0, 0.05, 30)               # vol explosion at the end
    close = 100 * np.exp(np.cumsum(ret))
    d = pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                     index=days)
    e = derisk_exposure(d)
    assert float(e.max()) <= 1.0 + 1e-12              # NEVER levers up
    assert float(e.iloc[-1]) < 0.5                    # de-levers into the spike
    assert (e.iloc[:252] == 1.0).all()                # warm-up: neutral
    # availability lag: perturbing day D's close must not change expo at D
    d2 = d.copy(); d2.iloc[-1] = d2.iloc[-1] * 0.5
    e2 = derisk_exposure(d2)
    assert np.isclose(float(e.iloc[-1]), float(e2.iloc[-1]))


def test_daily_frame_shape():
    days = ["2024-01-01", "2024-01-02"]
    df = _mk_days(days, _flat(100.0))
    d = daily_frame(df)
    assert list(d.index) == [pd.Timestamp(x) for x in days]
    assert (d["close"] == 100.0).all()
