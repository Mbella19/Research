"""Shared builders for the synthetic live-loop tests (not collected)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from daytrader.backtest.engine import make_cost_cfg
from daytrader.live.loop import LiveLoop
from daytrader.live.policy import PolicyRuntime
from daytrader.live.replay import ReplayHistory, SimBus


def dense_day(day: str, px: float = 25000.0, spread: float = 10.0,
              start="01:00", end="23:59") -> pd.DataFrame:
    t0 = pd.Timestamp(f"{day} {start}")
    t1 = pd.Timestamp(f"{day} {end}")
    ts = pd.date_range(t0, t1, freq="1min")
    df = pd.DataFrame({"ts": ts, "open": px, "high": px, "low": px,
                       "close": px, "tickvol": 25, "spread_pts": spread})
    for c in ("open", "high", "low", "close", "spread_pts"):
        df[c] = df[c].astype(np.float32)
    return df


def sparse_warm_days(n=300, start="2024-06-03", px0=20000.0, slope=5.0,
                     vol_spike_last=0, spread=10.0) -> pd.DataFrame:
    """Rising sparse days (gate ON); optional σ spike on the last k days."""
    days = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(5)
    rows = []
    for i, d in enumerate(days):
        c = px0 + slope * i + rng.normal(0, 15.0)   # baseline daily vol
        if vol_spike_last and i >= n - vol_spike_last:
            # alternating shocks, ENDING high so the SMA50 gate stays ON
            c += (350.0 if i % 2 == 1 else -280.0)
        for hhmm in ("01:00", "12:00", "16:30", "23:00", "23:59"):
            rows.append((pd.Timestamp(f"{d.date()} {hhmm}"),
                         c, c + 2, c - 2, c, 25, spread))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "tickvol", "spread_pts"])
    for c in ("open", "high", "low", "close", "spread_pts"):
        df[c] = df[c].astype(np.float32)
    return df


def set_bars(df: pd.DataFrame, ts_from: str, ts_to: str, *, o=None, h=None,
             l=None, c=None) -> None:
    m = (df["ts"] >= pd.Timestamp(ts_from)) & (df["ts"] <= pd.Timestamp(ts_to))
    for col, v in (("open", o), ("high", h), ("low", l), ("close", c)):
        if v is not None:
            df.loc[m, col] = np.float32(v)


def mk_sig(entries: list[tuple], atr: float = 20.0) -> pd.DataFrame:
    """[(bin_ts, side)] → signal frame in from_probabilities' output schema."""
    ts = pd.DatetimeIndex([pd.Timestamp(t) for t, _ in entries])
    return pd.DataFrame({
        "ts": ts, "avail_ts": ts + pd.Timedelta(minutes=5),
        "day": ts.normalize(),
        "atr_abs": np.float32(atr),
        "side": np.array([s for _, s in entries], dtype=np.int8),
        "ev_atr": np.float32(0.5),
    })


def rig(bars: pd.DataFrame, policy="v2", sleeves=("s1",), sig=None,
        start_ts=None, equity0=10_000.0, out=None):
    p = PolicyRuntime.load(policy)
    bars = bars.sort_values("ts").reset_index(drop=True)
    start_i = 0 if start_ts is None else int(
        np.searchsorted(bars["ts"].to_numpy(),
                        np.datetime64(pd.Timestamp(start_ts))))
    hist = ReplayHistory(bars, start_i)
    bus = SimBus(hist, make_cost_cfg(), equity0=equity0, vpu=p.vpu)
    out = Path(out) if out else Path(tempfile.mkdtemp()) / "rig"
    loop = LiveLoop(p, hist, bus, out_dir=out, sim_signals=sig,
                    sleeves=sleeves)
    return loop, bus, hist, out


def drive(loop, bus, n=None):
    steps = 0
    while bus.pre_open():
        loop.step()
        bus.deliver()
        loop.step()
        steps += 1
        if n is not None and steps >= n:
            break
    return steps


def trades(out: Path) -> pd.DataFrame:
    f = out / "trades.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def decisions(out: Path) -> pd.DataFrame:
    f = out / "decisions.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()
