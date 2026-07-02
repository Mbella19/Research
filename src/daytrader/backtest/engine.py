"""Event-driven backtester with stressed costs and day-trading discipline.

Execution model (MT5 bars are BID quotes):
  LONG : buy at bid + spread (+slippage); SL/TP/flat exits on bid.
  SHORT: sell at bid (−slippage); exits on ask = bid + spread.
  Entry at the open of the first 1m bar at/after the decision-bar close.
  SL and time/flat exits pay slippage; TP is a limit and fills at touch.
  Same-1m-bar SL+TP ambiguity resolves to SL (pessimistic).
  Per-1m-bar recorded spread × stress multiplier, floored.
  Positions: one at a time, max N entries/day, daily R stop, forced flat
  at the session cutoff. Sizing: fixed-fractional risk on current equity,
  capped by leverage; lots rounded down to lot_step.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import experiment, instrument
from ..utils.log import get_logger

log = get_logger("backtest")


@dataclass
class CostCfg:
    spread_mult: float
    spread_floor_pts: float
    slippage_pts: float
    commission_per_lot_side: float
    point_size: float


@dataclass
class RiskCfg:
    equity0: float
    risk_per_trade: float
    leverage_cap: float
    max_trades_per_day: int
    daily_stop_R: float
    value_per_unit: float
    min_lot: float
    lot_step: float
    max_lot: float


def _minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def make_cost_cfg(stress_mult: float | None = None, slip_mult: float = 1.0,
                  commission_mult: float = 1.0) -> CostCfg:
    """Costs from the active profile; stress_mult multiplies ON TOP of the
    profile's own (already stressed) spread multiplier."""
    from ..config import costs

    ins = instrument()
    c = costs()
    return CostCfg(
        spread_mult=float(c["spread_mult"]) * (stress_mult if stress_mult is not None else 1.0),
        spread_floor_pts=float(c["spread_floor_points"]),
        slippage_pts=float(c["slippage_points"]) * slip_mult,
        commission_per_lot_side=float(c["commission_per_lot_per_side"]) * commission_mult,
        point_size=float(ins["point_size"]),
    )


def make_risk_cfg(risk_per_trade: float | None = None) -> RiskCfg:
    ins = instrument()
    bt = experiment()["backtest"]
    return RiskCfg(
        equity0=float(bt["equity0"]),
        risk_per_trade=float(risk_per_trade if risk_per_trade is not None
                             else bt["risk_per_trade"]),
        leverage_cap=float(bt["leverage_cap"]),
        max_trades_per_day=int(bt["max_trades_per_day"]),
        daily_stop_R=float(bt["daily_stop_R"]),
        value_per_unit=float(ins["value_per_price_unit_per_lot"]),
        min_lot=float(ins["min_lot"]),
        lot_step=float(ins["lot_step"]),
        max_lot=float(ins["max_lot"]),
    )


def run_backtest(df1m: pd.DataFrame, sig: pd.DataFrame,
                 cost: CostCfg, risk: RiskCfg,
                 tp_atr: float | None = None, sl_atr: float | None = None) -> dict:
    """sig: columns ts, avail_ts, day, side (+1/-1/0), atr_abs (decision-TF ATR).
    Returns {trades, daily, metrics-ready arrays}."""
    ex = experiment()
    lab = ex["labels"]
    tp_k = float(tp_atr if tp_atr is not None else lab["tp_atr"])
    sl_k = float(sl_atr if sl_atr is not None else lab["sl_atr"])
    W = int(lab["horizon_bars"]) * int(ex["timeframes"]["decision_minutes"])
    overnight = bool(lab.get("overnight"))
    sess = instrument()["session"]
    no_entry_mod = _minutes(sess["no_entry_after"])
    flat_mod = _minutes(sess["force_flat_at"])

    ts1 = df1m["ts"].to_numpy()
    day1 = df1m["ts"].dt.normalize().to_numpy()
    mod1 = (df1m["ts"].dt.hour * 60 + df1m["ts"].dt.minute).to_numpy()
    o1 = df1m["open"].to_numpy(np.float64)
    h1 = df1m["high"].to_numpy(np.float64)
    l1 = df1m["low"].to_numpy(np.float64)
    c1 = df1m["close"].to_numpy(np.float64)
    sp1 = (np.maximum(df1m["spread_pts"].to_numpy(np.float64), cost.spread_floor_pts)
           * cost.spread_mult * cost.point_size)
    slip = cost.slippage_pts * cost.point_size
    n1 = len(ts1)

    # per-bar day-flat cutoff index (first 1m bar at/after force_flat, same day)
    day_change = np.flatnonzero(np.concatenate([[True], day1[1:] != day1[:-1]]))
    day_end = np.append(day_change[1:], n1)
    flat_of_day = {}
    for s_, e_ in zip(day_change, day_end):
        flat_of_day[day1[s_]] = s_ + int(np.searchsorted(mod1[s_:e_], flat_mod, "left"))
    flat_idx_1m = np.array([flat_of_day[d] for d in day1], dtype=np.int64)

    side = sig["side"].to_numpy()
    atr5 = sig["atr_abs"].to_numpy(np.float64)
    sig_close = sig["avail_ts"].to_numpy()
    sig_day = sig["day"].to_numpy()
    sig_mod = (sig["ts"].dt.hour * 60 + sig["ts"].dt.minute).to_numpy() + int(
        ex["timeframes"]["decision_minutes"])
    entry_idx = np.searchsorted(ts1, sig_close, "left")

    trades = []
    equity = risk.equity0
    open_pos: list[tuple] = []   # (exit_idx, pnl, R) of live positions
    max_conc = int(experiment()["backtest"].get("max_concurrent", 1))
    cur_day = None
    day_trades = 0
    day_R = 0.0

    for k in range(len(sig)):
        s = side[k]
        if s == 0:
            continue
        e = entry_idx[k]
        if e >= n1 or day1[e] != sig_day[k]:
            continue
        if sig_mod[k] > no_entry_mod:
            continue
        # settle positions that exited before this entry
        if open_pos:
            still = []
            for xp, pnl_p, R_p in open_pos:
                if xp <= e:
                    equity += pnl_p
                    day_R += R_p
                else:
                    still.append((xp, pnl_p, R_p))
            open_pos = still
        if len(open_pos) >= max_conc:
            continue
        if sig_day[k] != cur_day:
            cur_day, day_trades, day_R = sig_day[k], 0, 0.0
        if day_trades >= risk.max_trades_per_day or day_R <= risk.daily_stop_R:
            continue
        a = atr5[k]
        if not np.isfinite(a) or a <= 0:
            continue

        end = min(e + W, n1) if overnight else min(e + W, flat_idx_1m[e])
        if end - e < 3:
            continue
        # bid-quote fills: exactly one full spread + 2 slips per round trip
        sp_e = sp1[e]
        if s > 0:
            entry = o1[e] + sp_e + slip          # buy at ask
            sl_lvl = entry - sl_k * a
            tp_lvl = entry + tp_k * a
            sl_hit = l1[e:end] <= sl_lvl         # exits at bid
            tp_hit = h1[e:end] >= tp_lvl
        else:
            entry = o1[e] - slip                 # sell at bid
            sl_lvl = entry + sl_k * a
            tp_lvl = entry - tp_k * a
            sl_hit = h1[e:end] + sp1[e:end] >= sl_lvl   # cover at ask
            tp_hit = l1[e:end] + sp1[e:end] <= tp_lvl

        sl_i = int(np.argmax(sl_hit)) if sl_hit.any() else 1 << 30
        tp_i = int(np.argmax(tp_hit)) if tp_hit.any() else 1 << 30
        if sl_i <= tp_i and sl_i < 1 << 30:
            x = e + sl_i
            # gap-through stops fill at the (worse) open, not the level
            if s > 0:
                lvl = min(sl_lvl, o1[x]) if o1[x] < sl_lvl else sl_lvl
                exit_px = lvl - slip
            else:
                ask_open = o1[x] + sp1[x]
                lvl = max(sl_lvl, ask_open) if ask_open > sl_lvl else sl_lvl
                exit_px = lvl + slip
            reason = "sl"
        elif tp_i < 1 << 30:
            x = e + tp_i
            exit_px = tp_lvl
            reason = "tp"
        else:
            x = min(end, n1 - 1)
            base = o1[x] if (x < n1 and day1[x] == day1[e]) else c1[end - 1]
            if x >= n1 or day1[x] != day1[e]:
                x = end - 1
            exit_px = (base - slip) if s > 0 else (base + sp1[x] + slip)
            reason = "flat" if end < e + W else "time"

        risk_price = sl_k * a
        risk_dollars = equity * risk.risk_per_trade
        lots = risk_dollars / (risk_price * risk.value_per_unit)
        lev_cap = equity * risk.leverage_cap / (entry * risk.value_per_unit)
        lots = min(lots, lev_cap, risk.max_lot)
        lots = np.floor(lots / risk.lot_step) * risk.lot_step
        if lots < risk.min_lot:
            continue

        commission = 2 * cost.commission_per_lot_side * lots
        pnl = (exit_px - entry) * s * lots * risk.value_per_unit - commission
        R = pnl / max(risk_dollars, 1e-9)
        day_trades += 1
        open_pos.append((x, pnl, R))
        trades.append((sig["ts"].iloc[k], ts1[x], int(s), float(entry), float(exit_px),
                       float(lots), float(pnl), float(R), reason, float(a),
                       float(sp_e), float(commission)))

    cols = ["entry_ts", "exit_ts", "side", "entry", "exit", "lots", "pnl", "R",
            "reason", "atr", "spread_cost_px", "commission"]
    tdf = pd.DataFrame(trades, columns=cols)
    if len(tdf):  # equity settles in exit order (only closed P&L compounds)
        tdf = tdf.sort_values("exit_ts", kind="stable").reset_index(drop=True)
        tdf["equity_after"] = risk.equity0 + tdf["pnl"].cumsum()
        if tdf["equity_after"].min() <= risk.equity0 * 0.1:
            log.warning("equity dipped below 10% of initial — inspect before trusting")
    else:
        tdf["equity_after"] = pd.Series(dtype=float)

    # daily equity marks (flat overnight ⇒ step function at day ends)
    all_days = pd.DatetimeIndex(day1[day_change])
    if len(tdf):
        last_eq_per_day = tdf.groupby(tdf["exit_ts"].dt.normalize())["equity_after"].last()
        eq = last_eq_per_day.reindex(all_days).ffill().fillna(risk.equity0)
    else:
        eq = pd.Series(risk.equity0, index=all_days, dtype=float)
    daily = eq.to_frame("equity")
    daily["ret"] = daily["equity"].pct_change().fillna(0.0)
    return {"trades": tdf, "daily": daily}
