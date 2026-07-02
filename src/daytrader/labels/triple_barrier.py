"""Triple-barrier labels per side (long/short), resolved on the 1m path.

For each eligible decision bar i (close known at avail_ts):
  entry  = open of the first 1m bar at/after avail_ts (same day)
  TP/SL  = entry ± tp_atr·ATR / ∓ sl_atr·ATR   (ATR of the decision TF at i)
  window = horizon_bars × decision_minutes 1m steps, capped at the session
           force-flat time of that day
  label  = 1 iff TP is touched strictly before SL; same-1m-bar ambiguity
           counts as SL first (pessimistic). Timeout → 0.

Also produced: label end times (for purged CV), overlap-uniqueness weights
(López de Prado), and timeout mark-to-market returns for diagnostics.
Labels are computed on RAW prices; costs enter via the EV decision gate and
the backtester, so one label set serves every cost scenario.
"""
import numpy as np
import pandas as pd

from ..config import experiment, instrument
from ..data.loader import load_bars
from ..data.resample import load_tf_bars
from ..features.base import atr as atr_fn
from ..utils import paths
from ..utils.hashio import sha256_obj
from ..utils.log import get_logger

log = get_logger("labels")

LABEL_VERSION = 3


def _chunk_rows(W: int) -> int:
    """Bound the per-chunk window-matrix memory (~80M cells)."""
    return max(2000, int(8e7 // max(W, 1)))


def _minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def _first_touch(win_hi, win_lo, tp, sl, allowed, long_side: bool):
    """First-touch step of TP/SL inside per-row allowed window lengths."""
    W = win_hi.shape[1]
    steps = np.arange(W)[None, :]
    valid = steps < allowed[:, None]
    if long_side:
        tp_hits = (win_hi >= tp[:, None]) & valid
        sl_hits = (win_lo <= sl[:, None]) & valid
    else:
        tp_hits = (win_lo <= tp[:, None]) & valid
        sl_hits = (win_hi >= sl[:, None]) & valid
    tp_any = tp_hits.any(axis=1)
    sl_any = sl_hits.any(axis=1)
    tp_idx = np.where(tp_any, tp_hits.argmax(axis=1), np.iinfo(np.int32).max)
    sl_idx = np.where(sl_any, sl_hits.argmax(axis=1), np.iinfo(np.int32).max)
    win = tp_any & (tp_idx < sl_idx)          # tie → SL first (pessimistic)
    end = np.minimum(np.minimum(tp_idx, sl_idx), allowed - 1).astype(np.int32)
    # exit code: 1=tp, 2=sl, 0=timeout
    code = np.zeros(len(win), dtype=np.int8)
    code[win] = 1
    code[sl_any & (sl_idx <= tp_idx)] = 2
    return win, end, code


def _uniqueness(entry_idx: np.ndarray, exit_idx: np.ndarray, n_1m: int) -> np.ndarray:
    """Overlap-adjusted uniqueness: mean over the label window of 1/concurrency."""
    conc = np.zeros(n_1m + 1, dtype=np.float64)
    np.add.at(conc, entry_idx, 1.0)
    np.add.at(conc, exit_idx + 1, -1.0)
    conc = np.cumsum(conc)[:-1]
    inv = 1.0 / np.maximum(conc, 1.0)
    S = np.concatenate([[0.0], np.cumsum(inv)])
    length = np.maximum(exit_idx - entry_idx + 1, 1)
    return ((S[exit_idx + 1] - S[entry_idx]) / length).astype(np.float32)


def compute_labels(source: str) -> pd.DataFrame:
    ex = experiment()
    lab = ex["labels"]
    sess = instrument()["session"]
    dec_min = ex["timeframes"]["decision_minutes"]
    W = lab["horizon_bars"] * dec_min

    df1 = load_bars(source)
    ts1 = df1["ts"].to_numpy()
    day1 = df1["ts"].dt.normalize().to_numpy()
    mod1 = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute).to_numpy()
    o1 = df1["open"].to_numpy(np.float64)
    h1 = df1["high"].to_numpy(np.float64)
    l1 = df1["low"].to_numpy(np.float64)
    c1 = df1["close"].to_numpy(np.float64)
    n1 = len(df1)

    # per-1m-bar index of that day's flat cutoff (first bar at/after force_flat)
    flat_mod = _minutes(sess["force_flat_at"])
    day_change = np.flatnonzero(np.concatenate([[True], day1[1:] != day1[:-1]]))
    day_start_idx = day_change
    day_end_idx = np.append(day_change[1:], n1)  # exclusive
    flat_idx_per_day = {}
    for s_, e_ in zip(day_start_idx, day_end_idx):
        after = np.searchsorted(mod1[s_:e_], flat_mod, side="left")
        flat_idx_per_day[day1[s_]] = s_ + after
    flat_idx_1m = np.array([flat_idx_per_day[d] for d in day1], dtype=np.int64)

    df5 = load_tf_bars(source, dec_min)
    a5 = atr_fn(df5, lab["atr_period"]).to_numpy(np.float64)
    close_time = df5["avail_ts"].to_numpy()
    day5 = df5["day"].to_numpy()
    mod5 = (df5["ts"].dt.hour * 60 + df5["ts"].dt.minute).to_numpy()

    entry_idx = np.searchsorted(ts1, close_time, side="left")
    eligible = entry_idx < n1
    ok = eligible.copy()
    ok &= np.where(eligible, np.take(day1, np.minimum(entry_idx, n1 - 1)) == day5, False)
    ok &= mod5 + dec_min <= _minutes(sess["no_entry_after"])
    win = lab.get("entry_window")
    if win:  # optional experiment-level entry window (e.g. US session focus)
        ok &= (mod5 >= _minutes(win[0])) & (mod5 <= _minutes(win[1]))
    if lab.get("overnight"):
        # swing mode: window runs across sessions, capped only by data end
        allowed = np.minimum(entry_idx + W, n1)
    else:
        allowed = np.minimum(entry_idx + W, np.take(flat_idx_1m, np.minimum(entry_idx, n1 - 1)))
    allowed = (allowed - entry_idx).astype(np.int32)
    ok &= allowed >= 5
    ok &= np.isfinite(a5) & (a5 > 0)

    n5 = len(df5)
    y_long = np.zeros(n5, dtype=np.int8)
    y_short = np.zeros(n5, dtype=np.int8)
    x_long = np.zeros(n5, dtype=np.int8)     # exit codes: 1 tp / 2 sl / 0 timeout
    x_short = np.zeros(n5, dtype=np.int8)
    end_long = np.zeros(n5, dtype=np.int32)
    end_short = np.zeros(n5, dtype=np.int32)
    r_end = np.zeros(n5, dtype=np.float32)   # window-end mark vs entry, ATR units

    # pad end so windows never run off the array
    pad = W + 1
    h1p = np.pad(h1, (0, pad), constant_values=-np.inf)
    l1p = np.pad(l1, (0, pad), constant_values=np.inf)
    c1p = np.pad(c1, (0, pad), constant_values=np.nan)
    o1p = np.pad(o1, (0, pad), constant_values=np.nan)

    win_hi_all = np.lib.stride_tricks.sliding_window_view(h1p, W)
    win_lo_all = np.lib.stride_tricks.sliding_window_view(l1p, W)

    idx_all = np.flatnonzero(ok)
    CHUNK = _chunk_rows(W)
    for lo in range(0, len(idx_all), CHUNK):
        sel = idx_all[lo : lo + CHUNK]
        e_idx = entry_idx[sel]
        entry = o1p[e_idx]
        atr_ = a5[sel]
        alw = allowed[sel]
        wh = win_hi_all[e_idx]
        wl = win_lo_all[e_idx]

        tp = entry + lab["tp_atr"] * atr_
        sl = entry - lab["sl_atr"] * atr_
        wL, eL, cL = _first_touch(wh, wl, tp, sl, alw, True)
        tp_s = entry - lab["tp_atr"] * atr_
        sl_s = entry + lab["sl_atr"] * atr_
        wS, eS, cS = _first_touch(wh, wl, tp_s, sl_s, alw, False)

        y_long[sel] = wL
        y_short[sel] = wS
        x_long[sel] = cL
        x_short[sel] = cS
        end_long[sel] = e_idx + eL
        end_short[sel] = e_idx + eS
        last_c = c1p[e_idx + alw - 1]
        r_end[sel] = ((last_c - entry) / atr_).astype(np.float32)

    out = pd.DataFrame({
        "ts": df5["ts"],
        "eligible": ok,
        "y_long": y_long,
        "y_short": y_short,
        "x_long": x_long,
        "x_short": x_short,
        "r_end_atr": r_end,
        "entry_open": np.where(ok, o1p[entry_idx], np.nan).astype("float32"),
        "atr_abs": a5.astype("float32"),
    })
    # uniqueness weights per side (eligible rows only)
    for side, endx in (("long", end_long), ("short", end_short)):
        w = np.zeros(n5, dtype=np.float32)
        w[ok] = _uniqueness(entry_idx[ok], endx[ok], n1)
        out[f"w_uniq_{side}"] = w
    # label end times for purged CV (as timestamps)
    ts1p = np.append(ts1, np.repeat(ts1[-1], pad))
    out["t1_long"] = pd.Series(ts1p[np.minimum(end_long, n1 - 1)]).where(ok, pd.NaT)
    out["t1_short"] = pd.Series(ts1p[np.minimum(end_short, n1 - 1)]).where(ok, pd.NaT)
    return out


def _cache_key() -> str:
    ex = experiment()
    payload = {"v": LABEL_VERSION, "labels": ex["labels"],
               "tf": ex["timeframes"], "sym": instrument()["symbol"],
               "flat": instrument()["session"]["force_flat_at"],
               "noentry": instrument()["session"]["no_entry_after"]}
    return sha256_obj(payload)[:12]


def build_labels(source: str, refresh: bool = False) -> pd.DataFrame:
    p = paths.DATA_DIR / "labels" / f"{source}_{_cache_key()}.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    log.info(f"building labels for {source} …")
    df = compute_labels(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    el = df["eligible"]
    log.info(
        f"{source}: {int(el.sum()):,}/{len(df):,} eligible | "
        f"base rates: long {df.loc[el, 'y_long'].mean():.3f}, "
        f"short {df.loc[el, 'y_short'].mean():.3f}"
    )
    return df
