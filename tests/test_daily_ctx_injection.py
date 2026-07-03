"""daily_ctx injection (D-032 live hot path): the `daily` feature group
computed from an injected full-history day-aggregates table must reproduce
the full-history build exactly, and ctx rows dated after the window must be
causally inert."""
import numpy as np
import pandas as pd
import pytest

from daytrader.data.resample import resample_bars
from daytrader.features.daily_context import daily, day_aggs
from daytrader.features.registry import build_features_from_1m


def _mk_1m(n_days=260, bars_per_day=60, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    rows, px = [], 1000.0
    for d in days:
        for k in range(bars_per_day):
            ts = d + pd.Timedelta(minutes=60 + k)      # 01:00 onward
            o = px
            c = px + rng.normal(0, 1.0)
            h = max(o, c) + abs(rng.normal(0, 0.3))
            l = min(o, c) - abs(rng.normal(0, 0.3))
            rows.append((ts, o, h, l, c, int(rng.integers(10, 100)), 10.0))
            px = c
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "tickvol", "spread_pts"])
    for col in ("open", "high", "low", "close", "spread_pts"):
        df[col] = df[col].astype(np.float32)
    return df


DF1M = _mk_1m()
DF5 = resample_bars(DF1M, 5)
AGGS = day_aggs(DF5)


def test_default_vs_injected_identical():
    a = daily(DF5)
    b = daily(DF5, ctx=AGGS)
    pd.testing.assert_frame_equal(a, b)


def test_window_injection_matches_full_history():
    full = daily(DF5)
    cut = DF5["day"].unique()[-40]                     # last 40 days window
    win5 = DF5[DF5["day"] >= cut].reset_index(drop=True)
    win = daily(win5, ctx=AGGS)
    ref = full[DF5["day"] >= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(win, ref)


def test_future_ctx_rows_are_inert():
    days = DF5["day"].unique()
    mid = days[200]
    win5 = DF5[(DF5["day"] >= days[160]) & (DF5["day"] <= mid)].reset_index(drop=True)
    ctx_past = AGGS[AGGS.index <= mid]
    ctx_fut = AGGS.copy()
    ctx_fut.loc[ctx_fut.index > mid, :] = 9e9          # poison the future
    a = daily(win5, ctx=ctx_past)
    b = daily(win5, ctx=ctx_fut)
    pd.testing.assert_frame_equal(a, b)


def test_missing_ctx_day_raises():
    ctx = AGGS.drop(AGGS.index[100])
    win5 = DF5[DF5["day"] >= DF5["day"].unique()[90]].reset_index(drop=True)
    with pytest.raises(ValueError, match="missing"):
        daily(win5, ctx=ctx)


def test_registry_wiring_passes_ctx_through():
    a = build_features_from_1m(DF1M, ["daily"])
    b = build_features_from_1m(DF1M, ["daily"], daily_ctx=AGGS)
    pd.testing.assert_frame_equal(a, b)
