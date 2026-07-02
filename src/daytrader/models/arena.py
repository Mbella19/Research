"""v3 champion arena (D-026): honest engine-OOF comparison of S1 candidates.

Every candidate's saved OUT-OF-FOLD outputs drive the real backtest engine
(futures-stressed execution) over the training era. Champion rule
(pre-registered): highest daily-stream Sharpe; ties (±0.05) break toward
better worst-year, then more trades. Gate thresholds are chosen per
candidate on its own OOF plateau (training-side, D-020 methodology):
among thresholds with ≥400 trades, max Sharpe, worst-year tiebreak.

Candidates:
  v1     — frozen classification model (113 features), drift-priced gate 0.20
  v3a    — classification re-search on 142 features, plateau-picked gate
  v3b    — same config at challenger geometry tp2/sl1/H24 (if trained)
  evreg  — EV-regression head (142 features), direct EV gate (if trained)
"""
import numpy as np
import pandas as pd

from ..backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
from ..backtest.signals import from_probabilities, _base
from ..config import experiment, override_experiment, clear_overrides
from ..data.loader import load_bars
from ..features.registry import build_features
from ..models.dataset import cost_atr, drift_atr
from ..portfolio.grid import stream_metrics
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger

log = get_logger("arena")

THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
GEOM_B = {"tp_atr": 2.0, "sl_atr": 1.0, "horizon_bars": 24}


def _engine_stream(sig: pd.DataFrame, df1m: pd.DataFrame, days: pd.DatetimeIndex):
    res = run_backtest(df1m, sig, make_cost_cfg(), make_risk_cfg())
    t = res["trades"]
    r = (0.005 * t.groupby(t["exit_ts"].dt.normalize())["R"].sum()).reindex(days).fillna(0.0)
    return t, r


def _sig_prob(feat, oof, thr, lab):
    dec = experiment()["decision"]
    m = feat.merge(oof[["ts", "p_long", "p_short"]], on="ts", how="inner")
    ca = cost_atr(m, profile=dec.get("gate_cost_profile"))
    return m, from_probabilities(m, np.nan_to_num(m["p_long"].to_numpy()),
                                 np.nan_to_num(m["p_short"].to_numpy()),
                                 lab["tp_atr"], lab["sl_atr"], ca, thr,
                                 dec["prob_floor"], "both", drift_atr(m))


def _sig_ev(feat, oof_ev, thr):
    dec = experiment()["decision"]
    m = feat.merge(oof_ev, on="ts", how="inner")
    ca = cost_atr(m, profile=dec.get("gate_cost_profile"))
    ev_l = np.nan_to_num(m["ev_long"].to_numpy(), nan=-9.0)
    ev_s = np.nan_to_num(m["ev_short"].to_numpy(), nan=-9.0)
    sig = _base(m)
    best = np.where(ev_l >= ev_s, 1, -1)
    best_ev = np.maximum(ev_l, ev_s)
    sig["side"] = np.where(best_ev - ca > thr, best, 0).astype(np.int8)
    sig["ev_atr"] = (best_ev - ca).astype(np.float32)
    return m, sig


def _evaluate(name, sig_fn, df1m, days) -> dict | None:
    best = None
    for thr in THRESHOLDS:
        m, sig = sig_fn(thr)
        t, r = _engine_stream(sig, df1m, days)
        if len(t) < 400:
            continue
        sm = stream_metrics(r)
        row = {"thr": thr, "n": len(t),
               "n_long": int((t.side > 0).sum()), "n_short": int((t.side < 0).sum()),
               "exp_R": round(float(t.R.mean()), 4), "sharpe": round(sm["sharpe"], 3),
               "worst_year": round(sm["worst_year_pct"], 2),
               "ann": round(sm["ann_ret_pct"], 2), "dd": round(sm["max_dd_pct"], 2),
               "yearly": sm["yearly"]}
        log.info(f"{name} thr {thr:.2f}: Sharpe {row['sharpe']:.2f} n={row['n']} "
                 f"({row['n_long']}L/{row['n_short']}S) exp {row['exp_R']:+.3f}R "
                 f"worst-yr {row['worst_year']:+.1f}%")
        if best is None or (row["sharpe"], row["worst_year"]) > (best["sharpe"], best["worst_year"]):
            best = row
    return best


def run_arena() -> None:
    run_dir = new_run_dir("arena_v3")
    log.info(f"artifacts → {run_dir}")
    df1m = load_bars("real_training")
    days = pd.DatetimeIndex(sorted(df1m["ts"].dt.normalize().unique()))
    feat = build_features("real_training")     # 142-col cache; ts-merge is what matters
    lab = experiment()["labels"]
    out = {}

    p1 = paths.MODELS_DIR / "lgbm_final" / "oof_real.parquet"
    if p1.exists():
        oof = pd.read_parquet(p1)
        out["v1"] = _evaluate("v1", lambda thr: _sig_prob(feat, oof, thr, lab), df1m, days)

    p3 = paths.MODELS_DIR / "lgbm_final_v3" / "oof_real.parquet"
    if p3.exists():
        oof3 = pd.read_parquet(p3)
        out["v3a"] = _evaluate("v3a", lambda thr: _sig_prob(feat, oof3, thr, lab), df1m, days)

    pb = paths.MODELS_DIR / "lgbm_final_v3b" / "oof_real.parquet"
    if pb.exists():
        oofb = pd.read_parquet(pb)
        override_experiment(labels={**experiment()["labels"], **GEOM_B})
        try:
            labB = experiment()["labels"]
            out["v3b"] = _evaluate("v3b", lambda thr: _sig_prob(feat, oofb, thr, labB),
                                   df1m, days)
        finally:
            clear_overrides()

    pe = paths.MODELS_DIR / "lgbm_evreg_v3" / "oof_ev.parquet"
    if pe.exists():
        oe = pd.read_parquet(pe)
        out["evreg"] = _evaluate("evreg", lambda thr: _sig_ev(feat, oe, thr), df1m, days)

    ranked = sorted((k for k in out if out[k]),
                    key=lambda k: (-out[k]["sharpe"], -out[k]["worst_year"], -out[k]["n"]))
    out["champion"] = ranked[0] if ranked else None
    save_json(out, run_dir / "arena.json")
    log.info(f"ARENA: champion = {out['champion']} | "
             f"{[(k, out[k]['sharpe'] if out[k] else None) for k in ('v1','v3a','v3b','evreg') if k in out]}")
    log.info(f"ARENA DONE → {run_dir}")
