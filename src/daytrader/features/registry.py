"""Feature assembly: named groups → one aligned float32 matrix per source.

`build_features_from_1m` is the single code path used by the pipeline, the
lookahead tests, and (eventually) the forward-test harness — so what we test
is exactly what we trade.
"""
import numpy as np
import pandas as pd

from ..config import experiment, instrument
from ..data.loader import load_bars
from ..data.resample import resample_bars
from ..utils import paths
from ..utils.hashio import sha256_obj
from ..utils.log import get_logger
from . import base, daily_context, market_structure, timefeat, zigzag

log = get_logger("features")

FEATURE_VERSION = 4  # bump to invalidate caches when feature code changes

GROUPS = ("base", "time", "ms", "zz", "daily", "cal", "tape")

BAR_COLS = ["ts", "avail_ts", "day", "open", "high", "low", "close",
            "tickvol", "spread_pts"]

WARMUP_5M_BARS = 1100  # covers ema200/atr96 warmup + 21 completed 4h bars


def build_features_from_1m(df1m: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """1m bars → decision-TF bar frame + feature columns (float32)."""
    ex = experiment()
    dec_min = ex["timeframes"]["decision_minutes"]
    ctx_min = ex["timeframes"]["context_minutes"]

    df5 = resample_bars(df1m, dec_min)
    htf = {m: resample_bars(df1m, m) for m in ctx_min}
    atr5 = base.atr(df5, ex["labels"]["atr_period"])

    parts = [df5[BAR_COLS]]
    if "base" in groups:
        parts.append(base.compute(df5))
    if "time" in groups:
        parts.append(timefeat.compute(df5))
    if "ms" in groups:
        parts.append(market_structure.compute(df5, htf, atr5))
    if "zz" in groups:
        parts.append(zigzag.compute(df5, atr5))
    if "daily" in groups:
        parts.append(daily_context.daily(df5))
    if "cal" in groups:
        parts.append(daily_context.cal(df5))
    if "tape" in groups:
        parts.append(daily_context.tape(df5))
    out = pd.concat(parts, axis=1)
    out = out.iloc[WARMUP_5M_BARS:].reset_index(drop=True)
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Model-facing columns (excludes bar columns and _helpers)."""
    return [c for c in df.columns if c not in BAR_COLS and not c.startswith("_")]


def _cache_key(groups: list[str]) -> str:
    ex = experiment()
    payload = {
        "v": FEATURE_VERSION,
        "groups": sorted(groups),
        "tf": ex["timeframes"],
        "atr_period": ex["labels"]["atr_period"],
        "instrument": instrument()["symbol"],
    }
    return sha256_obj(payload)[:12]


def build_features(source: str, groups: list[str] | None = None,
                   refresh: bool = False) -> pd.DataFrame:
    groups = groups or list(experiment()["features"]["groups"])
    key = _cache_key(groups)
    p = paths.DATA_DIR / "features" / f"{source}_{key}.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    log.info(f"building features for {source} (groups={groups}) …")
    df = build_features_from_1m(load_bars(source), groups)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    n_feat = len(feature_cols(df))
    log.info(f"{source}: {len(df):,} decision bars × {n_feat} features → {p.name}")
    return df
