"""Signal generators: rule-based benchmarks and (later) model-driven signals.

Every generator returns the decision-bar frame with a `side` column
(+1 long / −1 short / 0 flat) decided ONLY from information available at the
bar close — the engine handles entry on the next 1m bar.
"""
import numpy as np
import pandas as pd

from ..config import instrument


def _base(feat: pd.DataFrame) -> pd.DataFrame:
    sig = feat[["ts", "avail_ts", "day"]].copy()
    point = float(instrument()["point_size"])
    sig["atr_abs"] = feat["_atr_points"] * point if "_atr_points" in feat else np.nan
    return sig


def ms_rule(feat: pd.DataFrame) -> pd.DataFrame:
    """The MS.txt indicator's native entries (4H trend + 30m zone break on 5m)."""
    sig = _base(feat)
    sig["side"] = np.where(feat["ms_buy_break"] > 0, 1,
                           np.where(feat["ms_sell_break"] > 0, -1, 0)).astype(np.int8)
    return sig


def momentum(feat: pd.DataFrame, thr: float = 1.5) -> pd.DataFrame:
    """Naive ATR-normalized momentum breakout, aligned with the 4h MS trend."""
    sig = _base(feat)
    long_ = (feat["ret_12"] > thr) & (feat["ms_trend_4h"] >= 0)
    short_ = (feat["ret_12"] < -thr) & (feat["ms_trend_4h"] <= 0)
    sig["side"] = np.where(long_, 1, np.where(short_, -1, 0)).astype(np.int8)
    return sig


def from_probabilities(feat: pd.DataFrame, p_long: np.ndarray, p_short: np.ndarray,
                       tp_atr: float, sl_atr: float, cost_atr: np.ndarray,
                       min_ev_atr: float, prob_floor: float,
                       allowed_sides: str = "both",
                       drift_atr: np.ndarray | float = 0.0) -> pd.DataFrame:
    """Cost-aware EV gate: trade the side whose calibrated probability clears
    EV = p·TP − (1−p)·SL − cost ± drift > min_ev (all in ATR units).
    The drift term prices the instrument's secular per-horizon drift into ONE
    symmetric formula — no side is banned; direction asymmetry, where it
    appears, is measured physics, not policy."""
    sig = _base(feat)
    ev_long = p_long * tp_atr - (1 - p_long) * sl_atr - cost_atr + drift_atr
    ev_short = p_short * tp_atr - (1 - p_short) * sl_atr - cost_atr - drift_atr
    if allowed_sides == "long":
        ev_short = np.full_like(ev_short, -np.inf)
    elif allowed_sides == "short":
        ev_long = np.full_like(ev_long, -np.inf)
    best = np.where(ev_long >= ev_short, 1, -1)
    best_ev = np.maximum(ev_long, ev_short)
    best_p = np.where(best > 0, p_long, p_short)
    side = np.where((best_ev > min_ev_atr) & (best_p > prob_floor), best, 0)
    sig["side"] = side.astype(np.int8)
    sig["ev_atr"] = best_ev.astype(np.float32)
    return sig
