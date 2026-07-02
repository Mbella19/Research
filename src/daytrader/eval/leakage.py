"""Synthetic↔real leakage audit.

Two independent probes:
1. Exact-window duplication: do synthetic universes contain verbatim copies of
   real 1m return sequences? (detects block-bootstrap provenance, and — the
   dangerous case — copies of VALIDATION-period material)
2. Day-path nearest-neighbour: for random synthetic days, the max correlation
   of their intraday 5m return path against every real validation day,
   benchmarked against real-training days as the null.

The locked OOS set is not touched (it postdates the synthetic files' creation;
assumption recorded in the report).
"""
import numpy as np
import pandas as pd

from ..config import synth_sources
from ..data.loader import load_bars
from ..utils.artifacts import new_run_dir
from ..utils.hashio import save_json
from ..utils.log import get_logger

log = get_logger("eval.leakage")

WIN = 120          # 1m bars per query window (2h)
N_QUERIES = 60     # windows sampled per synth universe
ROUND = 7          # decimals for float-exact matching


def _returns_1m(source: str) -> np.ndarray:
    df = load_bars(source)
    ts = pd.DatetimeIndex(df["ts"])
    day = ts.normalize().to_numpy()
    r = np.diff(np.log(df["close"].to_numpy(np.float64)), prepend=np.nan)
    r[day != np.roll(day, 1)] = np.nan
    return r


def _window_hashes(r: np.ndarray, win: int) -> set[bytes]:
    """Hashes of every finite `win`-length rounded return window."""
    import hashlib

    rr = np.round(r, ROUND)
    finite = np.isfinite(rr)
    ok = np.ones(len(rr) - win + 1, dtype=bool)
    bad = np.where(~finite)[0]
    for b in bad:                        # windows touching a NaN are excluded
        lo, hi = max(0, b - win + 1), min(len(ok), b + 1)
        ok[lo:hi] = False
    hs = set()
    idx = np.where(ok)[0]
    view = np.lib.stride_tricks.sliding_window_view(rr, win)
    for i in idx:
        hs.add(hashlib.blake2b(view[i].tobytes(), digest_size=12).digest())
    return hs


def _query_windows(r: np.ndarray, win: int, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rr = np.round(r, ROUND)
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        i = int(rng.integers(1, len(rr) - win))
        w = rr[i : i + win]
        if np.isfinite(w).all():
            out.append(w)
        tries += 1
    return out


def _day_paths_5m(source: str, max_days: int | None = None, seed: int = 0) -> np.ndarray:
    """Matrix of per-day standardized 5m return paths (fixed common length)."""
    df = load_bars(source)
    ts = pd.DatetimeIndex(df["ts"])
    c5 = pd.Series(df["close"].to_numpy(np.float64), index=ts).resample("5min").last().dropna()
    r5 = np.log(c5).diff()
    frame = pd.DataFrame({"d": c5.index.normalize(), "t": c5.index.time, "r": r5.values})
    piv = frame.pivot_table(index="d", columns="t", values="r")
    piv = piv.dropna(axis=1, thresh=int(len(piv) * 0.95)).dropna(axis=0)
    m = piv.to_numpy()
    if max_days and len(m) > max_days:
        rng = np.random.default_rng(seed)
        m = m[rng.choice(len(m), max_days, replace=False)]
    m = m - m.mean(axis=1, keepdims=True)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-12)


def run_leakage_audit() -> dict:
    run_dir = new_run_dir("leakage")
    log.info(f"artifacts → {run_dir}")

    log.info("hashing real training windows …")
    r_train = _returns_1m("real_training")
    train_hashes = _window_hashes(r_train, WIN)
    log.info(f"  {len(train_hashes):,} training windows hashed")
    r_valid = _returns_1m("real_validation")
    valid_hashes = _window_hashes(r_valid, WIN)
    log.info(f"  {len(valid_hashes):,} validation windows hashed")

    import hashlib

    results = {}
    for s in synth_sources().keys():
        qs = _query_windows(_returns_1m(s), WIN, N_QUERIES, seed=42)
        hits_train = sum(
            hashlib.blake2b(w.tobytes(), digest_size=12).digest() in train_hashes for w in qs
        )
        hits_valid = sum(
            hashlib.blake2b(w.tobytes(), digest_size=12).digest() in valid_hashes for w in qs
        )
        results[s] = {"queries": len(qs), "exact_hits_train": hits_train,
                      "exact_hits_valid": hits_valid}
        log.info(f"{s}: {hits_train}/{len(qs)} windows verbatim in TRAIN, "
                 f"{hits_valid}/{len(qs)} in VALIDATION")

    log.info("day-path nearest-neighbour audit vs validation …")
    v_paths = _day_paths_5m("real_validation")
    t_paths = _day_paths_5m("real_training", max_days=300)
    k = min(v_paths.shape[1], t_paths.shape[1])
    null_max = (t_paths[:, :k] @ v_paths[:, :k].T).max(axis=1)
    day_stats = {"null_train_vs_valid": {
        "p50": float(np.median(null_max)), "p99": float(np.quantile(null_max, 0.99)),
        "max": float(null_max.max())}}
    for s in synth_sources().keys():
        s_paths = _day_paths_5m(s, max_days=200, seed=1)
        mx = (s_paths[:, :k] @ v_paths[:, :k].T).max(axis=1)
        day_stats[s] = {"p50": float(np.median(mx)), "p99": float(np.quantile(mx, 0.99)),
                        "max": float(mx.max())}
        log.info(f"{s}: day-path max-corr vs validation p50={day_stats[s]['p50']:.2f} "
                 f"max={day_stats[s]['max']:.2f}")

    verdict = {
        "synth_contains_training_material": any(
            r["exact_hits_train"] > 0 for r in results.values()),
        "synth_contains_validation_material": any(
            r["exact_hits_valid"] > 0 for r in results.values()) or any(
            day_stats[s]["max"] > 0.999 for s in synth_sources().keys()),
    }
    out = {"window_audit": results, "day_path_audit": day_stats, "verdict": verdict}
    save_json(out, run_dir / "leakage_audit.json")
    log.info(f"verdict: {verdict}")
    return out
