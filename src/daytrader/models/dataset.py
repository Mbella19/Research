"""Training-matrix assembly + purged k-fold splits.

Scoring discipline: CV test blocks are REAL-training rows only. Synthetic rows
(2030+ timestamps) may only ever sit on the train side, weighted by w_synth.
Purging removes any train row whose label window overlaps a test block
(± embargo); this kills label-overlap leakage across fold boundaries.
"""
import gc

import numpy as np
import pandas as pd

from ..config import experiment, instrument, synth_sources
from ..features.registry import build_features, feature_cols
from ..labels.triple_barrier import build_labels
from ..utils.log import get_logger

log = get_logger("models.dataset")


LABEL_COLS = {"eligible", "y_long", "y_short", "x_long", "x_short", "r_end_atr",
              "entry_open", "atr_abs", "w_uniq_long", "w_uniq_short",
              "t1_long", "t1_short", "source", "is_real"}


def load_xy(source: str) -> tuple[pd.DataFrame, list[str]]:
    feat = build_features(source)
    lab = build_labels(source)
    fcols = feature_cols(feat)          # from the FEATURES frame only — labels
    assert set(fcols).isdisjoint(LABEL_COLS)  # must never enter the matrix
    df = feat.merge(lab, on="ts", how="inner")
    df = df[df["eligible"]].reset_index(drop=True)
    nan_rows = df[fcols].isna().any(axis=1)
    if nan_rows.any():
        log.info(f"{source}: dropping {int(nan_rows.sum()):,} rows with NaN features")
        df = df[~nan_rows].reset_index(drop=True)
    df["source"] = source
    df["is_real"] = source.startswith("real")
    return df, fcols


def realized_R(df: pd.DataFrame, side: str) -> np.ndarray:
    """Label-implied trade outcome in ATR units (gross of costs)."""
    lab = experiment()["labels"]
    code = df[f"x_{side}"].to_numpy()
    r_end = df["r_end_atr"].to_numpy(np.float32)
    if side == "short":
        r_end = -r_end
    out = np.where(code == 1, lab["tp_atr"],
                   np.where(code == 2, -lab["sl_atr"], r_end)).astype(np.float32)
    return out


def cost_atr(df: pd.DataFrame, spread_mult: float | None = None,
             slip_mult: float = 1.0, profile: str | None = None) -> np.ndarray:
    """All-in round-trip cost per trade in ATR units (named or active cost
    profile; spread_mult multiplies on top of the profile's own multiplier)."""
    from ..config import costs

    ins = instrument()
    c = costs(profile)
    point = float(ins["point_size"])
    sm = float(c["spread_mult"]) * (spread_mult if spread_mult is not None else 1.0)
    spread_px = np.maximum(df["spread_now"].to_numpy(np.float64),
                           c["spread_floor_points"]) * sm * point
    slip_px = 2 * c["slippage_points"] * slip_mult * point
    comm_px = 2 * c["commission_per_lot_per_side"] / ins["value_per_price_unit_per_lot"]
    if "atr_abs" in df.columns:
        atr_px = df["atr_abs"].to_numpy(np.float64)
    else:                       # features frame: ATR carried in points
        atr_px = df["_atr_points"].to_numpy(np.float64) * point
    return ((spread_px + slip_px + comm_px) / np.maximum(atr_px, 1e-9)).astype(np.float32)


_REAL_DAILY_CLOSES: pd.Series | None = None


def _real_daily_closes() -> pd.Series:
    """Daily closes across all real sources (for trailing-μ warm-up across
    window boundaries). Causally airtight: the trailing EWMA is a one-sided
    filter, so the value mapped to day D depends only on days < D — rows
    later in the file can never influence earlier μ values."""
    global _REAL_DAILY_CLOSES
    if _REAL_DAILY_CLOSES is None:
        import os

        from ..utils import paths

        oos_ok = ((paths.RUNS_DIR / "OOS_EXECUTED.flag").exists()
                  or os.environ.get("DAYTRADER_UNLOCK_OOS") == "1")
        sources = ["real_training", "real_validation"] + (
            ["real_locked_oos"] if oos_ok else [])   # same policy as load_bars
        parts = []
        for s in sources:
            p = paths.PARQUET_DIR / f"{s}.parquet"
            if p.exists():
                b = pd.read_parquet(p, columns=["ts", "close"])
                parts.append(b.groupby(b["ts"].dt.normalize())["close"].last())
        _REAL_DAILY_CLOSES = pd.concat(parts).sort_index()
        _REAL_DAILY_CLOSES = _REAL_DAILY_CLOSES[~_REAL_DAILY_CLOSES.index.duplicated()]
    return _REAL_DAILY_CLOSES


def drift_atr(df: pd.DataFrame) -> np.ndarray:
    """Per-row expected drift over the label horizon, in ATR units.

    decision.drift_mode:
      constant — μ fixed in config (v1; measured on the FULL training window,
                 which peeks relative to decisions inside that window)
      trailing — μ_d = EWMA(span 126, min 63) of daily ln returns, lagged one
                 day (D-023 causality fix; no forward information at all)."""
    ex = experiment()
    dec = ex["decision"]
    mode = dec.get("drift_mode", "constant")
    h_frac_day = ex["labels"]["horizon_bars"] * ex["timeframes"]["decision_minutes"] / 1380.0
    point = float(instrument()["point_size"])
    if "atr_abs" in df.columns:
        atr_frac = df["atr_abs"].to_numpy(np.float64) / df["close"].to_numpy(np.float64)
    else:
        atr_frac = (df["_atr_points"].to_numpy(np.float64) * point
                    / df["close"].to_numpy(np.float64))
    if mode == "trailing":
        day = df["ts"].dt.normalize()
        close_d = _real_daily_closes()
        if close_d.index.min() > day.min() or close_d.index.max() < day.max():
            close_d = df.groupby(day)["close"].last()   # synth / foreign source
        mu_d = (np.log(close_d).diff().ewm(span=126, min_periods=63).mean()
                .shift(1).fillna(0.0))
        mu = day.map(mu_d).to_numpy(np.float64)
    else:
        mu = float(dec.get("drift_mu_daily", 0.0))
        if mu == 0.0:
            return np.zeros(len(df), dtype=np.float32)
    out = mu * h_frac_day / np.maximum(atr_frac, 1e-6)
    return np.clip(out, -0.5, 0.5).astype(np.float32)


def assemble(w_synth: float = 0.0, sources_synth: list[str] | None = None) -> dict:
    """Real training rows (+ optionally synth) into one training bundle."""
    first, fcols = load_xy("real_training")
    parts = [first]
    if w_synth > 0:
        for s in (sources_synth or list(synth_sources().keys())):
            parts.append(load_xy(s)[0])
            gc.collect()
    df = pd.concat(parts, ignore_index=True)
    del parts, first
    gc.collect()

    X = df[fcols].to_numpy(np.float32)
    is_real = df["is_real"].to_numpy()
    bundle = {
        "X": X,
        "feature_names": fcols,
        "is_real": is_real,
        "ts": df["ts"].to_numpy(),
        "spread_now": df["spread_now"].to_numpy(np.float32),
        "atr_abs": df["atr_abs"].to_numpy(np.float32),
        "cost_atr": cost_atr(df),
    }
    for side in ("long", "short"):
        w = df[f"w_uniq_{side}"].to_numpy(np.float32).copy()
        w[~is_real] *= w_synth
        bundle[f"y_{side}"] = df[f"y_{side}"].to_numpy(np.int8)
        bundle[f"w_{side}"] = w
        bundle[f"t1_{side}"] = df[f"t1_{side}"].to_numpy()
        bundle[f"R_{side}"] = realized_R(df, side)
    del df
    gc.collect()
    log.info(f"assembled: {len(X):,} rows ({int(is_real.sum()):,} real) × {len(fcols)} features")
    return bundle


def purged_folds(bundle: dict, n_folds: int | None = None,
                 embargo_days: float | None = None) -> list[dict]:
    """Contiguous real-time test blocks; train side purged + embargoed.
    Synth rows are always train-side."""
    cv = experiment()["cv"]
    n_folds = n_folds or cv["n_folds"]
    embargo = pd.Timedelta(days=embargo_days if embargo_days is not None else cv["embargo_days"])

    ts = pd.DatetimeIndex(bundle["ts"])
    is_real = bundle["is_real"]
    real_ts = ts[np.flatnonzero(is_real)]
    edges = np.quantile(real_ts.asi8, np.linspace(0, 1, n_folds + 1))
    # label windows end at the later of the two sides' exits
    t1_end = pd.DatetimeIndex(
        np.maximum(pd.DatetimeIndex(bundle["t1_long"]).asi8,
                   pd.DatetimeIndex(bundle["t1_short"]).asi8).view("datetime64[ns]"))
    folds = []
    for k in range(n_folds):
        t0 = pd.Timestamp(int(edges[k]))
        t1 = pd.Timestamp(int(edges[k + 1]))
        upper = (ts <= t1) if k == n_folds - 1 else (ts < t1)
        test_mask = is_real & (ts >= t0) & upper
        # purge: keep train rows whose label window [ts, t1_end] clears the
        # embargoed test window entirely
        clear = (t1_end < (t0 - embargo)) | (ts > (t1 + embargo))
        train_mask = (clear & ~test_mask) | (~is_real)
        folds.append({"k": k, "test": np.flatnonzero(test_mask),
                      "train": np.flatnonzero(train_mask),
                      "t0": t0, "t1": t1})
    return folds
