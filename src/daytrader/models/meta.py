"""B5 (D-026): meta-label sizing layer.

A second, deliberately tiny LightGBM learns P(champion's trade wins) from the
REGIME feature blocks only (daily/cal/tape — information the primary model
sees but doesn't specialize on at the per-trade level). Trained on the
champion's OOF trades with time-purged folds (purge key = trade EXIT time),
so every sizing decision is out-of-fold twice over.

Pre-registered sizing map (no tuning): OOF p_win quintiles →
  size 0.0 below q20 · 1.0 between · 1.5 above q80.
Adopt iff the resized S1 daily stream's Sharpe improves ≥ +0.05 over flat
sizing. The v2 EV-rank refutation does not apply: different model, different
features, different target.
"""
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..portfolio.grid import stream_metrics
from ..utils.artifacts import new_run_dir
from ..utils.hashio import save_json
from ..utils.log import get_logger

log = get_logger("meta")

META_PARAMS = {"objective": "binary", "num_leaves": 7, "max_depth": 3,
               "min_child_samples": 200, "learning_rate": 0.05,
               "feature_fraction": 0.8, "bagging_fraction": 0.8,
               "bagging_freq": 1, "lambda_l2": 20.0, "verbose": -1, "seed": 7}
REGIME_PREFIXES = ("d_", "c_", "t_")


def run_meta(trades: pd.DataFrame, feat: pd.DataFrame, days: pd.DatetimeIndex,
             n_folds: int = 5, embargo_days: float = 2.0,
             tag: str = "meta") -> dict:
    """trades: engine OOF trades (entry_ts = decision-bar ts). Returns the
    adopt/reject verdict + per-trade OOF multipliers."""
    run_dir = new_run_dir(f"meta_{tag}")
    log.info(f"artifacts → {run_dir}")
    cols = [c for c in feat.columns if c.startswith(REGIME_PREFIXES)]
    fmap = feat.set_index("ts")[cols]
    X = fmap.reindex(trades["entry_ts"]).to_numpy(np.float32)
    y = (trades["R"] > 0).astype(int).to_numpy()
    t_ent = trades["entry_ts"].to_numpy()
    t_exit = trades["exit_ts"].to_numpy()
    n = len(trades)
    log.info(f"meta set: {n} trades × {len(cols)} regime features, "
             f"base win rate {y.mean():.3f}")

    edges = np.quantile(t_ent.astype("datetime64[ns]").astype(np.int64),
                        np.linspace(0, 1, n_folds + 1))
    emb = np.timedelta64(int(embargo_days * 86400 * 1e9), "ns")
    p_oof = np.full(n, np.nan)
    for k in range(n_folds):
        t0, t1 = edges[k].astype("datetime64[ns]"), edges[k + 1].astype("datetime64[ns]")
        te = (t_ent >= t0) & ((t_ent < t1) if k < n_folds - 1 else (t_ent <= t1))
        tr = (t_exit < t0 - emb) | (t_ent > t1 + emb)
        if te.sum() < 20 or tr.sum() < 200:
            continue
        d = lgb.Dataset(X[tr], label=y[tr])
        booster = lgb.train(META_PARAMS, d, num_boost_round=120)
        p_oof[te] = booster.predict(X[te])
        del booster, d
        gc.collect()

    ok = np.isfinite(p_oof)
    q20, q80 = np.nanquantile(p_oof, 0.2), np.nanquantile(p_oof, 0.8)
    mult = np.where(~ok, 1.0, np.where(p_oof < q20, 0.0,
                                       np.where(p_oof > q80, 1.5, 1.0)))
    # monotonicity diagnostic
    bins = pd.qcut(pd.Series(p_oof[ok]), 5, duplicates="drop")
    mono = pd.Series(trades["R"].to_numpy()[ok]).groupby(bins.values, observed=True).mean()
    log.info(f"p_win quintile → mean R: {[round(v, 3) for v in mono.tolist()]}")

    def stream(mults):
        r = (0.005 * pd.Series(trades["R"].to_numpy() * mults,
                               index=trades["exit_ts"].dt.normalize())
             .groupby(level=0).sum()).reindex(days).fillna(0.0)
        return stream_metrics(r)

    m_flat, m_meta = stream(np.ones(n)), stream(mult)
    verdict = bool(m_meta["sharpe"] >= m_flat["sharpe"] + 0.05)
    out = {"n": int(n), "flat_sharpe": round(m_flat["sharpe"], 3),
           "meta_sharpe": round(m_meta["sharpe"], 3),
           "flat_worst_year": round(m_flat["worst_year_pct"], 2),
           "meta_worst_year": round(m_meta["worst_year_pct"], 2),
           "quintile_meanR": [round(v, 4) for v in mono.tolist()],
           "killed_pct": round(float((mult == 0).mean() * 100), 1),
           "boosted_pct": round(float((mult == 1.5).mean() * 100), 1),
           "ADOPT": verdict}
    save_json(out, run_dir / "meta_verdict.json")
    log.info(f"META: flat {m_flat['sharpe']:.2f} → meta {m_meta['sharpe']:.2f} "
             f"| ADOPT = {verdict}")
    return {"verdict": out, "mult": mult, "run_dir": run_dir}
