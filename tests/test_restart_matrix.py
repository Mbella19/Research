"""Restart reconciliation (D-032): every (state × broker) combination must
resolve safely — adopt, finalize-from-deals, orphan-halt — and downtime
decisions are MISSED, never chased."""
import json

import pandas as pd
import pytest

from daytrader.config import clear_overrides
from daytrader.live.loop import LiveLoop

from live_helpers import decisions, dense_day, drive, mk_sig, rig, set_bars, trades

D = "2026-03-02"


@pytest.fixture(autouse=True)
def _clean():
    yield
    clear_overrides()


def _restart(loop, out):
    """New loop instance on the same state dir / bus / hist (crash+restart)."""
    return LiveLoop(loop.p, loop.hist, loop.bus, out_dir=out,
                    sim_signals=(loop._sim_sig.reset_index()
                                 if loop._sim_sig is not None else None),
                    sleeves=loop.sleeves)


def test_open_open_adopts_and_exits_on_schedule():
    bars = dense_day(D)
    sig = mk_sig([(f"{D} 10:00", 1)])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus, n=90)                     # fill + hold
    assert loop.s1 is not None and loop.s1.filled
    loop2 = _restart(loop, out)
    assert loop2.s1 is not None and loop2.s1.ticket == loop.s1.ticket
    drive(loop2, bus)                          # continue to horizon
    tr = trades(out)
    assert len(tr) == 1
    assert tr.iloc[0]["exit_ts"] == f"{D} 14:05:00"    # same schedule kept
    assert tr.iloc[0]["reason"] == "time"


def test_open_flat_finalizes_from_deals():
    bars = dense_day(D)
    set_bars(bars, f"{D} 11:30", f"{D} 11:31", l=24800)   # SL while "down"
    sig = mk_sig([(f"{D} 10:00", 1)])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus, n=90)
    assert loop.s1 is not None
    # crash: bus advances unattended past the SL bar
    while bus.pre_open():
        bus.deliver()
        if str(hist.bars["ts"].iloc[-1]) >= f"{D} 12:00:00":
            break
    loop2 = _restart(loop, out)
    for _ in range(4):                          # debounce = 3 statuses
        loop2.step()
    assert loop2.s1 is None
    tr = trades(out)
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "sl"


def test_flat_orphan_halts_entries():
    bars = dense_day(D)
    loop, bus, hist, out = rig(bars, sig=mk_sig([(f"{D} 12:00", 1)]),
                               start_ts=f"{D} 09:00")
    drive(loop, bus, n=30)
    # a position with OUR magic appears out of nowhere
    bus.positions[9999] = {"magic": loop.p.magic_s1, "side": 1, "lots": 1.0,
                           "entry": 25000.0, "sl": None, "tp": None,
                           "ts_open": hist.bars["ts"].iloc[-1], "j_open": 0}
    drive(loop, bus, n=5)
    assert loop.entries_halted and "ORPHAN" in loop.halt_reason
    dec = decisions(out)
    assert len(trades(out)) == 0               # the 12:00 signal never traded


def test_downtime_decisions_missed_not_chased():
    bars = dense_day(D)
    sig = mk_sig([(f"{D} 12:00", 1)])          # fires while "down"
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus, n=30)                     # ~09:30
    while bus.pre_open():                      # down until 13:00
        bus.deliver()
        if str(hist.bars["ts"].iloc[-1]) >= f"{D} 13:00:00":
            break
    loop2 = _restart(loop, out)
    drive(loop2, bus)
    assert len(trades(out)) == 0               # never chased
    dec = decisions(out)
    missed = dec[dec["ts"] == f"{D} 12:00:00"]
    assert len(missed) == 0 or "MISSED" in str(missed["reason"].iloc[0])


def test_state_file_round_trip():
    bars = dense_day(D)
    sig = mk_sig([(f"{D} 10:00", 1)])
    loop, bus, hist, out = rig(bars, sig=sig, start_ts=f"{D} 09:00")
    drive(loop, bus, n=90)
    st = json.loads((out / "state.json").read_text())
    assert st["policy"] == "v2" and st["s1"]["filled"] is True
    assert st["day"]["trades"] == 1
    loop2 = _restart(loop, out)
    assert loop2.day.trades == 1 and loop2.order_seq == loop.order_seq
