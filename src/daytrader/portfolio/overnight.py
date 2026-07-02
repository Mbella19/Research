"""Sleeve 2 — gated overnight drift (rule-based, long-only, pre-registered menu).

The alpha is structural: NAS100's secular drift concentrates in the
non-US-session hours (measured on TRAINING: usC→usO +6.6%/yr unconditional,
trend-gated +13.4%/yr t=+2.26, at ~60% of B&H's daily vol; D-021). The sleeve
holds a long index position through that window when a lagged daily trend
gate is on; the gate's job is regime quarantine, not prediction.

Causality contract (pinned by tests): every gate / vol-cap / ATR value used
for a position opened on day D is computed from COMPLETED daily bars ≤ D−1.
A 23:00 entry never sees that day's 23:59 close; a 01:00 entry never sees
its own day at all. This is one day STALER than the D-021 study measured
(which had a 59-minute availability peek) — the grid re-measures honestly.

Execution (bid quotes, long only): buy at ask = open + spread + slip; sell
at bid = open − slip; optional protective stop on the 1m low path with
gap-through fills at the (worse) open; commission per lot both ways; optional
CFD swap financing per night held (futures execution ⇒ 0, stress row ⇒ >0).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import CostCfg
from ..config import instrument
from ..utils.log import get_logger

log = get_logger("sleeve2")

WINDOWS = ("usC", "postgap")            # 23:00→next 16:30  |  01:00→same-day 16:30
GATES = ("sma50", "sma100", "sma50_rising")
EXIT_MOD = 16 * 60 + 30                 # US cash open, broker time


@dataclass(frozen=True)
class S2Params:
    window: str = "usC"
    gate: str = "sma50"
    volcap: bool = False
    stop_atr: float | None = None       # v3: 5.0 = catastrophe insurance (P3);
                                        # tight stops (2.5) remain refuted
    ivol: bool = False                  # D-022 symmetric overlay — REFUTED, kept for tests
    derisk: bool = False                # D-026 P4: one-sided de-lever in high vol

    def tag(self) -> str:
        return (f"{self.window}|{self.gate}|{'volcap' if self.volcap else 'novc'}"
                f"|{f'stop{self.stop_atr:.1f}' if self.stop_atr else 'nostop'}"
                f"{'|ivol' if self.ivol else ''}{'|derisk' if self.derisk else ''}")


def daily_frame(df1m: pd.DataFrame) -> pd.DataFrame:
    g = df1m.groupby(df1m["ts"].dt.normalize())
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last()})
    d.index.name = "day"
    return d


def gate_series(d: pd.DataFrame, p: S2Params) -> pd.Series:
    """True at index day D ⇒ a position MAY be opened on day D.
    Only completed closes ≤ D−1 enter the computation (final .shift(1))."""
    c = d["close"]
    if p.gate == "sma50":
        on = c > c.rolling(50).mean()
    elif p.gate == "sma100":
        on = c > c.rolling(100).mean()
    elif p.gate == "sma50_rising":
        s = c.rolling(50).mean()
        on = (c > s) & (s > s.shift(1))
    else:
        raise ValueError(f"unknown gate {p.gate}")
    if p.volcap:
        rv = np.log(c).diff().rolling(20).std()
        thr = rv.rolling(504, min_periods=252).quantile(0.9)
        on = on & ~(rv > thr)           # NaN threshold (warm-up) ⇒ cap inactive
    return on.astype(bool).shift(1, fill_value=False)   # availability lag: closes ≤ D−1


def ivol_exposure(d: pd.DataFrame) -> pd.Series:
    """D-022 overlay: expo_D = clip(expanding-median(σ)≤D−1 / σ_{D−1}, 0.25, 2),
    σ = EWMA(span 20) of daily ln returns; 1.0 until 252 days of history."""
    sig = np.log(d["close"]).diff().ewm(span=20).std()
    med = sig.expanding(min_periods=252).median()
    expo = (med / sig).clip(0.25, 2.0).shift(1)     # info ≤ D−1
    return expo.fillna(1.0)


def derisk_exposure(d: pd.DataFrame) -> pd.Series:
    """D-026 P4 (adopted): ONE-SIDED de-lever — expo = min(1, median σ / σ),
    never levers up (the lever-up half of ivol_exposure was the refuted part).
    Same causality: expanding median, EWMA-20 σ, all ≤ D−1."""
    sig = np.log(d["close"]).diff().ewm(span=20).std()
    med = sig.expanding(min_periods=252).median()
    expo = (med / sig).clip(upper=1.0).shift(1)     # info ≤ D−1
    return expo.fillna(1.0)


def _daily_atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = np.maximum(d["high"] - d["low"],
                    np.maximum((d["high"] - d["close"].shift(1)).abs(),
                               (d["low"] - d["close"].shift(1)).abs()))
    return tr.rolling(n).mean().shift(1)  # completed days ≤ D−1


def sleeve2_run(df1m: pd.DataFrame, p: S2Params, cost: CostCfg,
                swap_bp_night: float = 0.0,
                gate_override: pd.Series | None = None,
                context_daily: pd.DataFrame | None = None) -> dict:
    """Run the sleeve on 1m bars. Returns per-unit-notional economics:
    trades (ret_net = net fractional P&L of the position's notional) and
    daily_ret (settled on exit day, index = all trading days in the data).

    context_daily: optional daily OHLC with HISTORY BEFORE the evaluation
    window (live always has it; without it the SMA gate sits dark through
    its warm-up). Causally airtight — the gate value for day D depends only
    on closes ≤ D−1, so a context extending past the window is inert."""
    if p.window not in WINDOWS:
        raise ValueError(f"unknown window {p.window}")
    d = daily_frame(df1m)
    ctx = context_daily if context_daily is not None else d
    gate = (gate_series(ctx, p).reindex(d.index, fill_value=False)
            if gate_override is None else gate_override.astype(bool))
    atr_d = _daily_atr(ctx).reindex(d.index)
    if p.derisk:
        expo = derisk_exposure(ctx).reindex(d.index, fill_value=1.0)
    elif p.ivol:
        expo = ivol_exposure(ctx).reindex(d.index, fill_value=1.0)
    else:
        expo = pd.Series(1.0, index=d.index)
    vpu = float(instrument()["value_per_price_unit_per_lot"])

    ts = df1m["ts"].to_numpy()
    day1 = df1m["ts"].dt.normalize().to_numpy()
    mod = (df1m["ts"].dt.hour * 60 + df1m["ts"].dt.minute).to_numpy()
    o = df1m["open"].to_numpy(np.float64)
    lo = df1m["low"].to_numpy(np.float64)
    sp = (np.maximum(df1m["spread_pts"].to_numpy(np.float64), cost.spread_floor_pts)
          * cost.spread_mult * cost.point_size)
    slip = cost.slippage_pts * cost.point_size

    days = d.index.to_numpy()
    start = np.flatnonzero(np.concatenate([[True], day1[1:] != day1[:-1]]))
    end = np.append(start[1:], len(ts))
    entry_mod = 23 * 60 if p.window == "usC" else 0

    def bar_at(di: int, minute: int) -> int:
        s_, e_ = start[di], end[di]
        j = s_ + int(np.searchsorted(mod[s_:e_], minute, "left"))
        return j if j < e_ else -1

    rows = []
    for di in range(len(days)):
        if not bool(gate.iloc[di]):
            continue
        ei = bar_at(di, entry_mod)
        if ei < 0:
            continue
        xdi = di + 1 if p.window == "usC" else di
        if xdi >= len(days):
            continue
        xi = bar_at(xdi, EXIT_MOD)
        if xi < 0:
            xi = end[xdi] - 1           # no 16:30 bar that day: exit at day end
        if xi <= ei:
            continue
        entry = o[ei] + sp[ei] + slip
        exit_i, reason = xi, "window"
        a = float(atr_d.iloc[di]) if np.isfinite(atr_d.iloc[di]) else np.nan
        if p.stop_atr and np.isfinite(a):
            stop_lvl = entry - p.stop_atr * a
            hit = np.flatnonzero(lo[ei:xi] <= stop_lvl)
            if hit.size:
                j = ei + int(hit[0])
                lvl = min(stop_lvl, o[j]) if o[j] < stop_lvl else stop_lvl
                exit_i, reason = j, "stop"
        exit_px = ((lvl if reason == "stop" else o[exit_i]) - slip)
        comm_frac = 2.0 * cost.commission_per_lot_side / (entry * vpu)
        nights = int((day1[exit_i] - day1[ei]) / np.timedelta64(1, "D"))
        e_d = float(expo.iloc[di])
        ret = e_d * ((exit_px - entry) / entry - comm_frac
                     - swap_bp_night * 1e-4 * max(nights, 0))
        rows.append((ts[ei], ts[exit_i], float(entry), float(exit_px),
                     float(ret), reason, int(nights), e_d))

    trades = pd.DataFrame(rows, columns=["entry_ts", "exit_ts", "entry", "exit",
                                         "ret_net", "reason", "nights", "expo"])
    daily_ret = pd.Series(0.0, index=pd.DatetimeIndex(days), name="s2_ret")
    if len(trades):
        settled = trades.groupby(trades["exit_ts"].dt.normalize())["ret_net"].sum()
        daily_ret.loc[settled.index] = settled.to_numpy()
    return {"trades": trades, "daily_ret": daily_ret,
            "exposure": float(gate.mean()), "params": p.tag()}
