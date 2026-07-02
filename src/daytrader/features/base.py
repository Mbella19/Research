"""`base` feature group — price/vol/volume/spread geometry on decision bars.

All features are stationary and scale-free (ATR-normalized, ratios, logs, or
bounded). No raw price levels. Everything uses only the current and past bars
of the decision timeframe (plus per-day anchored running stats).
"""
import numpy as np
import pandas as pd

from ..config import instrument

MAX_LOOKBACK = 450  # bars of warmup before features are trustworthy (ema200/atr96)

RET_HORIZONS = [1, 3, 6, 12, 24, 48, 144]


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = np.maximum.reduce(
        [
            (df["high"] - df["low"]).to_numpy(),
            (df["high"] - prev_close).abs().to_numpy(),
            (df["low"] - prev_close).abs().to_numpy(),
        ]
    )
    return pd.Series(tr, index=df.index).ewm(alpha=1 / period, min_periods=period).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = np.log(df["close"].astype("float64"))
    a = atr(df, 14)
    atr_frac = (a / df["close"]).clip(lower=1e-6)  # ATR in return units

    for k in RET_HORIZONS:
        out[f"ret_{k}"] = ((c - c.shift(k)) / (atr_frac * np.sqrt(k))).clip(-10, 10)

    for span in (20, 50, 200):
        ema = df["close"].ewm(span=span, min_periods=span).mean()
        out[f"ema_dist_{span}"] = ((df["close"] - ema) / a).clip(-15, 15)

    atr_slow = pd.Series(
        np.maximum.reduce(
            [
                (df["high"] - df["low"]).to_numpy(),
                (df["high"] - df["close"].shift(1)).abs().to_numpy(),
                (df["low"] - df["close"].shift(1)).abs().to_numpy(),
            ]
        ),
        index=df.index,
    ).ewm(alpha=1 / 96, min_periods=96).mean()
    out["atr_regime"] = np.log(a / atr_slow).clip(-2, 2)

    r1 = c.diff()
    rv_fast = r1.rolling(12).std()
    rv_slow = r1.rolling(144).std()
    out["rv_ratio"] = np.log(rv_fast / rv_slow.clip(lower=1e-8)).clip(-2, 2)

    for k in (12, 48):
        hi = df["high"].rolling(k).max()
        lo = df["low"].rolling(k).min()
        out[f"range_pos_{k}"] = ((df["close"] - lo) / (hi - lo).clip(lower=1e-9)).clip(0, 1)

    rng = (df["high"] - df["low"]).clip(lower=1e-9)
    out["body_frac"] = ((df["close"] - df["open"]) / rng).clip(-1, 1)
    out["upper_wick"] = ((df["high"] - np.maximum(df["open"], df["close"])) / rng).clip(0, 1)
    out["lower_wick"] = ((np.minimum(df["open"], df["close"]) - df["low"]) / rng).clip(0, 1)
    out["range_atr"] = ((df["high"] - df["low"]) / a).clip(0, 8)
    out["gap_open"] = ((df["open"] - df["close"].shift(1)) / a).clip(-8, 8)

    out["efficiency_12"] = (
        (c - c.shift(12)).abs() / r1.abs().rolling(12).sum().clip(lower=1e-9)
    ).clip(0, 1)

    # ── day-anchored ────────────────────────────────────────────────
    g = df.groupby("day", sort=False)
    day_open = g["open"].transform("first")
    run_hi = g["high"].cummax()
    run_lo = g["low"].cummin()
    out["dist_day_open"] = ((df["close"] - day_open) / a).clip(-15, 15)
    out["dist_day_high"] = ((df["close"] - run_hi) / a).clip(-15, 0)
    out["dist_day_low"] = ((df["close"] - run_lo) / a).clip(0, 15)
    out["day_range_atr"] = ((run_hi - run_lo) / a).clip(0, 30)

    tp = (df["high"] + df["low"] + df["close"]) / 3
    tv = df["tickvol"].astype("float64").clip(lower=1.0)
    vwap = (tp * tv).groupby(df["day"], sort=False).cumsum() / tv.groupby(
        df["day"], sort=False
    ).cumsum()
    out["dist_vwap"] = ((df["close"] - vwap) / a).clip(-15, 15)

    tv_base = tv.ewm(span=288, min_periods=96).mean()
    out["tickvol_rel"] = np.log(tv / tv_base.clip(lower=1.0)).clip(-3, 3)
    out["tickvol_accel"] = np.log(
        tv.ewm(span=6, min_periods=6).mean() / tv.ewm(span=48, min_periods=48).mean()
    ).clip(-3, 3)

    sp = df["spread_pts"].astype("float64")
    out["spread_now"] = sp.clip(0, 100)
    out["spread_rel"] = np.log(sp.clip(lower=0.5) / sp.ewm(span=288, min_periods=96).mean().clip(lower=0.5)).clip(-2, 2)

    # ATR in points — used by labels/backtest, exported as helper (not a model feature)
    out["_atr_points"] = a / float(instrument()["point_size"])
    return out.astype("float32")
