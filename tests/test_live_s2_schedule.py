"""S2 live schedule semantics (overnight.py sleeve2_run parity): 23:00 entry
when gate ON, next-trading-day 16:30 exit (weekends held), no-23:00-bar day
skipped, catastrophe stop from the FILL (v3) vs no stop (v2), derisk expo."""
import numpy as np
import pandas as pd
import pytest

from daytrader.config import clear_overrides

from live_helpers import dense_day, drive, rig, set_bars, sparse_warm_days, trades


@pytest.fixture(autouse=True)
def _clean():
    yield
    clear_overrides()


def _warm_plus(test_days: list[pd.DataFrame], **warm_kw) -> pd.DataFrame:
    warm = sparse_warm_days(**warm_kw)
    px = float(warm["close"].iloc[-1])
    parts = [warm]
    for td in test_days:
        for c in ("open", "high", "low", "close"):
            td[c] = (td[c] - np.float32(25000.0) + np.float32(px)).astype(np.float32)
        parts.append(td)
    return pd.concat(parts, ignore_index=True)


def test_entry_exit_and_weekend_hold():
    # warm ends Thu 2025-07-24; test Fri + Mon (weekend hold)
    fri, mon = "2025-07-25", "2025-07-28"
    bars = _warm_plus([dense_day(fri), dense_day(mon)],
                      n=290, start="2024-06-14")
    loop, bus, hist, out = rig(bars, policy="v2", sleeves=("s2",),
                               start_ts=f"{fri} 01:00")
    drive(loop, bus)
    tr = trades(out)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["entry_ts"].startswith(f"{fri} 23:00")
    assert t["exit_ts"].startswith(f"{mon} 16:30")     # held over the weekend
    assert t["reason"] == "window"
    assert t["expo"] == 1.0 and (pd.isna(t["sl"]) or str(t["sl"]) == "")


def test_no_2300_bar_no_entry():
    d1, d2 = "2025-07-25", "2025-07-28"
    early = dense_day(d1, end="20:00")                 # day dies before 23:00
    bars = _warm_plus([early, dense_day(d2)], n=290, start="2024-06-14")
    loop, bus, hist, out = rig(bars, policy="v2", sleeves=("s2",),
                               start_ts=f"{d1} 01:00")
    drive(loop, bus)
    tr = trades(out)
    # no entry on the early-close day; Monday evening entry is the only one
    assert len(tr) == 0 or not str(tr.iloc[0]["entry_ts"]).startswith(d1)


def test_v3_catastrophe_stop_fires_from_fill():
    fri, mon = "2025-07-25", "2025-07-28"
    dmon = dense_day(mon)
    bars = _warm_plus([dense_day(fri), dmon], n=290, start="2024-06-14")
    px = float(bars["close"].iloc[-1])
    # Monday 03:00: crash far below any 5×dailyATR stop
    set_bars(bars, f"{mon} 03:00", f"{mon} 03:05", l=px - 4000)
    loop, bus, hist, out = rig(bars, policy="v3", sleeves=("s2",),
                               start_ts=f"{fri} 01:00")
    drive(loop, bus)
    tr = trades(out)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["reason"] == "sl"                          # stop, not window
    assert str(t["exit_ts"]).startswith(f"{mon} 03:0")
    fill, sl = float(t["entry"]), float(t["sl"])
    assert sl < fill                                    # anchored below FILL
    assert float(t["exit"]) <= sl + 1e-6                # gap-through ≤ level


def test_v2_no_stop_survives_crash():
    fri, mon = "2025-07-25", "2025-07-28"
    dmon = dense_day(mon)
    bars = _warm_plus([dense_day(fri), dmon], n=290, start="2024-06-14")
    px = float(bars["close"].iloc[-1])
    set_bars(bars, f"{mon} 03:00", f"{mon} 03:05", l=px - 4000)
    loop, bus, hist, out = rig(bars, policy="v2", sleeves=("s2",),
                               start_ts=f"{fri} 01:00")
    drive(loop, bus)
    tr = trades(out)
    assert len(tr) == 1
    assert tr.iloc[0]["reason"] == "window"             # rode it to 16:30
    assert str(tr.iloc[0]["exit_ts"]).startswith(f"{mon} 16:30")


def test_gate_off_no_entry():
    fri = "2025-07-25"
    warmish = sparse_warm_days(n=290, start="2024-06-14")
    # collapse the last 10 warm closes far below the SMA50 → gate OFF
    last10 = warmish["ts"] >= warmish["ts"].iloc[-1] - pd.Timedelta(days=14)
    for c in ("open", "high", "low", "close"):
        warmish.loc[last10, c] = np.float32(1000.0)
    bars = pd.concat([warmish, dense_day(fri, px=1000.0)], ignore_index=True)
    loop, bus, hist, out = rig(bars, policy="v2", sleeves=("s2",),
                               start_ts=f"{fri} 01:00")
    drive(loop, bus)
    assert len(trades(out)) == 0


def test_v3_derisk_expo_below_one():
    fri, mon = "2025-07-25", "2025-07-28"
    bars = _warm_plus([dense_day(fri), dense_day(mon)],
                      n=290, start="2024-06-14", vol_spike_last=25)
    # big equity: the de-risk crushes notional; sizing must stay ≥ min_lot
    loop, bus, hist, out = rig(bars, policy="v3", sleeves=("s2",),
                               start_ts=f"{fri} 01:00", equity0=2_000_000.0)
    drive(loop, bus)
    tr = trades(out)
    assert len(tr) == 1
    assert 0.0 < float(tr.iloc[0]["expo"]) < 0.9        # de-levered
