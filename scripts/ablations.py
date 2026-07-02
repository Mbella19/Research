"""Training-side ablations at the winning geometry (no validation contact).

A: feature groups — base / base+time / +ms / +zz / all
B: training era — full history vs 2023+ (regime-matched), both scored on
   recent-era test rows for apples-to-apples comparison.
Scored by 3-fold purged OOF economics with the current best config.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daytrader.config import override_experiment
from daytrader.utils.hashio import load_json, save_json
from daytrader.utils import paths
from daytrader.utils.artifacts import new_run_dir
from daytrader.utils.log import get_logger

log = get_logger("ablations")


def group_cols(names, groups):
    out = []
    for c in names:
        is_ms = c.startswith("ms_")
        is_zz = c.startswith(("zz_", "pat_"))
        is_time = c.startswith(("tod_", "dow_", "mins_", "us_dst"))
        g = "ms" if is_ms else "zz" if is_zz else "time" if is_time else "base"
        if g in groups:
            out.append(c)
    return out


def restrict_bundle(bundle, cols=None, min_ts=None):
    b = dict(bundle)
    if cols is not None:
        idx = [bundle["feature_names"].index(c) for c in cols]
        b["X"] = bundle["X"][:, idx]
        b["feature_names"] = cols
    if min_ts is not None:
        keep = pd.DatetimeIndex(bundle["ts"]) >= pd.Timestamp(min_ts)
        for k in ("X", "is_real", "ts", "spread_now", "atr_abs", "cost_atr",
                  "y_long", "y_short", "w_long", "w_short", "t1_long", "t1_short",
                  "R_long", "R_short"):
            b[k] = b[k][keep.to_numpy() if hasattr(keep, "to_numpy") else keep]
    return b


def main():
    run_dir = new_run_dir("ablations")
    log.info(f"artifacts → {run_dir}")
    best_geom = load_json(paths.MODELS_DIR / "lgbm_barriers.json")["best_geom"]
    override_experiment(labels=best_geom)
    log.info(f"geometry: {best_geom}")

    from daytrader.models.dataset import assemble, purged_folds
    from daytrader.models.lgbm import run_oof
    cfg = load_json(paths.MODELS_DIR / "lgbm_candidates.json")["best"]["cfg"]

    bundle = assemble(w_synth=0.0)
    folds_all = purged_folds(bundle)
    folds3 = [folds_all[i] for i in (0, 2, 4)]
    results = {}

    # A: feature groups
    for tag, groups in [("base", ["base"]), ("base+time", ["base", "time"]),
                        ("base+time+ms", ["base", "time", "ms"]),
                        ("base+time+zz", ["base", "time", "zz"]),
                        ("all", ["base", "time", "ms", "zz"])]:
        cols = group_cols(bundle["feature_names"], groups)
        b = restrict_bundle(bundle, cols=cols)
        r = run_oof(b, folds3, cfg)
        results[tag] = {"n_feat": len(cols), "econ": r["econ"],
                        "auc": [round(r["auc_test_long"], 4), round(r["auc_test_short"], 4)]}
        log.info(f"{tag:>14} ({len(cols):3d}f): netEV {r['econ']['net_ev_sum']:+8.1f} "
                 f"({r['econ']['n_trades']:,} tr, {r['econ']['net_ev_mean']:+.4f}/tr) "
                 f"auc {r['auc_test_long']:.4f}/{r['auc_test_short']:.4f}")
        save_json(results, run_dir / "ablations.json")

    # B: era restriction — score both on 2023+ test rows
    b_recent = restrict_bundle(bundle, min_ts="2023-01-01")
    folds_recent = purged_folds(b_recent, n_folds=3)
    r_recent = run_oof(b_recent, folds_recent, cfg)
    results["train2023+"] = {"econ": r_recent["econ"],
                             "auc": [round(r_recent["auc_test_long"], 4),
                                     round(r_recent["auc_test_short"], 4)]}
    log.info(f"train2023+: netEV {r_recent['econ']['net_ev_sum']:+8.1f} "
             f"({r_recent['econ']['n_trades']:,} tr, {r_recent['econ']['net_ev_mean']:+.4f}/tr)")

    # full-history model, recent test rows only (from the 'all' run's folds)
    r_all = run_oof(bundle, folds3, cfg)
    recent_mask = pd.DatetimeIndex(bundle["ts"][r_all["test_idx"]]) >= pd.Timestamp("2023-01-01")
    from daytrader.models.lgbm import economic_score
    idx_recent = r_all["test_idx"][recent_mask.to_numpy()]
    e = economic_score(r_all["oof"]["long"][idx_recent], r_all["oof"]["short"][idx_recent],
                       bundle, idx_recent)
    results["full_hist_recent_slice"] = {"econ": e}
    log.info(f"full-hist model on 2023+ slice: netEV {e['net_ev_sum']:+8.1f} "
             f"({e['n_trades']:,} tr, {e['net_ev_mean']:+.4f}/tr)")
    save_json(results, run_dir / "ablations.json")
    log.info("ABLATIONS DONE")


if __name__ == "__main__":
    main()
