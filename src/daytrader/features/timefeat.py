"""`time` feature group — session clock, weekday, US-DST flag."""
import numpy as np
import pandas as pd

from ..config import instrument

MAX_LOOKBACK = 0


def _us_dst(dates: pd.Series) -> np.ndarray:
    """US daylight-saving flag (2nd Sunday of March → 1st Sunday of November).
    Matters because the US cash open shifts by 1h in broker time."""
    years = dates.dt.year.to_numpy()
    out = np.zeros(len(dates), dtype=bool)
    d64 = dates.dt.normalize().to_numpy()
    for y in np.unique(years):
        mar1 = np.datetime64(f"{y}-03-01")
        dow = (pd.Timestamp(mar1).dayofweek + 1) % 7  # days since Sunday
        second_sun = mar1 + np.timedelta64((6 - dow) % 7 + 7, "D")
        nov1 = np.datetime64(f"{y}-11-01")
        dow = (pd.Timestamp(nov1).dayofweek + 1) % 7
        first_sun = nov1 + np.timedelta64((6 - dow) % 7, "D")
        m = years == y
        out[m] = (d64[m] >= second_sun) & (d64[m] < first_sun)
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    sess = instrument()["session"]
    ts = df["ts"]
    out = pd.DataFrame(index=df.index)
    mod = ts.dt.hour * 60 + ts.dt.minute
    out["tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    dow = ts.dt.dayofweek
    for d in range(5):
        out[f"dow_{d}"] = (dow == d).astype("float32")
    hh, mm = sess["force_flat_at"].split(":")
    flat_mod = int(hh) * 60 + int(mm)
    hh, mm = sess["day_start"].split(":")
    start_mod = int(hh) * 60 + int(mm)
    out["mins_since_open"] = ((mod - start_mod).clip(lower=0) / 1380).astype("float32")
    out["mins_to_flat"] = ((flat_mod - mod).clip(lower=0) / 1380).astype("float32")
    out["us_dst"] = _us_dst(ts).astype("float32")
    return out.astype("float32")
