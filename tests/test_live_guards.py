"""Engine-order entry guards in the live loop (engine.py:147-166 parity),
on hand-crafted bars + injected signals."""
import pandas as pd
import pytest

from daytrader.config import clear_overrides

from live_helpers import decisions, dense_day, drive, mk_sig, rig, set_bars, trades

D = "2026-03-02"        # a Monday


@pytest.fixture(autouse=True)
def _clean():
    yield
    clear_overrides()


def test_concurrent_blocked_and_horizon_exit():
    bars = dense_day(D)
    sig = mk_sig([(f"{D} 10:00", 1), (f"{D} 10:30", 1)])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus)
    tr, dec = trades(out), decisions(out)
    assert len(tr) == 1                                   # second blocked
    row = dec[dec["ts"] == f"{D} 10:30:00"]
    assert row["reason"].iloc[0] == "position_open"
    t = tr.iloc[0]
    # flat prices: horizon exit 240 bars after the 10:05 entry bar → 14:05
    assert t["entry_ts"] == f"{D} 10:05:00"
    assert t["exit_ts"] == f"{D} 14:05:00" and t["reason"] == "time"


def test_daily_stop_R():
    bars = dense_day(D)
    # three fast losers: price collapses after each entry (SL −1R each)
    ent = [("10:00", "10:06"), ("11:00", "11:06"), ("12:00", "12:06"),
           ("13:00", None)]
    for hh, crash in ent[:3]:
        set_bars(bars, f"{D} {crash}", f"{D} {crash}", l=24800)
    sig = mk_sig([(f"{D} {hh}:00" if ":" not in hh else f"{D} {hh}", 1)
                  for hh, _ in ent])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus)
    tr, dec = trades(out), decisions(out)
    assert len(tr) == 3 and (tr["reason"] == "sl").all()
    assert tr["R_or_ret"].sum() < -3.0
    row = dec[dec["ts"] == f"{D} 13:00:00"]
    assert row["reason"].iloc[0] == "daily_stop_R"


def test_max_trades_per_day():
    bars = dense_day(D)
    hours = ["03:00", "05:00", "07:00", "09:00", "11:00", "13:00", "15:00"]
    for hh in hours:
        h = int(hh[:2])
        # winner: spike above TP (60 pts) right after each entry
        set_bars(bars, f"{D} {h:02d}:06", f"{D} {h:02d}:07", h=25100)
    sig = mk_sig([(f"{D} {hh}", 1) for hh in hours])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 02:00")
    drive(loop, bus)
    tr, dec = trades(out), decisions(out)
    assert len(tr) == 6 and (tr["reason"] == "tp").all()
    row = dec[dec["ts"] == f"{D} 15:00:00"]
    assert row["reason"].iloc[0] == "max_trades_per_day"


def test_entry_cutoff():
    bars = dense_day(D)
    sig = mk_sig([(f"{D} 22:40", 1), (f"{D} 22:45", 1)])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 22:00")
    drive(loop, bus)
    tr, dec = trades(out), decisions(out)
    assert len(tr) == 1                     # 22:40 allowed (closes 22:45)
    assert tr.iloc[0]["entry_ts"] == f"{D} 22:45:00"
    assert tr.iloc[0]["reason"] == "flat"   # forced flat at 23:30
    assert tr.iloc[0]["exit_ts"] == f"{D} 23:30:00"
    row = dec[dec["ts"] == f"{D} 22:45:00"]
    assert row["reason"].iloc[0] == "past_entry_cutoff"


def test_sizing_floor_not_counted():
    bars = dense_day(D)
    sig = pd.concat([mk_sig([(f"{D} 10:00", 1)], atr=1e9),   # lots < 0.01
                     mk_sig([(f"{D} 11:00", 1)], atr=20.0)])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus)
    tr, dec = trades(out), decisions(out)
    assert dec[dec["ts"] == f"{D} 10:00:00"]["reason"].iloc[0] == "sizing_floor"
    assert len(tr) == 1                     # the 11:00 one traded
    assert tr.iloc[0]["entry_ts"] == f"{D} 11:05:00"


def test_exit_then_enter_same_boundary():
    bars = dense_day(D)
    sig = mk_sig([(f"{D} 10:00", 1), (f"{D} 14:00", 1)])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus)
    tr = trades(out)
    assert len(tr) == 2
    # trade 1 horizon-exits at 14:05 open; trade 2 enters the SAME bar
    # (engine settles exits with exit_idx ≤ e before evaluating the entry)
    assert tr.iloc[0]["exit_ts"] == f"{D} 14:05:00"
    assert tr.iloc[1]["entry_ts"] == f"{D} 14:05:00"
