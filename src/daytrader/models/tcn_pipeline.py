"""TCN pipeline: purged-CV (3 folds × 3 seeds) + final deep ensemble.

Uses the SAME data recipe (synth weight) as the frozen GBT recipe so the
champion comparison is apples-to-apples. OOF predictions are saved for the
stacking blend; economics computed with the same cost-aware gate.
"""
import gc

import numpy as np
import pandas as pd
import torch

from ..config import experiment
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger
from .dataset import assemble, purged_folds
from .lgbm import blockwise_econ, economic_score
from .pipeline import RECIPE, _per_trade_lb
from .tcn import HybridTCN, CHANNELS, predict, source_ids, train_tcn

log = get_logger("models.tcn_pipeline")

TCN_OOF = paths.MODELS_DIR / "tcn_oof.parquet"
TCN_DIR = paths.MODELS_DIR / "tcn_final"


def _recipe_w() -> float:
    if RECIPE.exists():
        return float(load_json(RECIPE).get("w_synth", 0.0))
    return 0.0


def _es_from_fold(bundle, fold):
    from .lgbm import _es_split

    return _es_split(bundle, fold)


def run_tcn_cv(cv_folds: tuple = (0, 2, 4)) -> None:
    ex = experiment()
    run_dir = new_run_dir("tcn_cv")
    w = _recipe_w()
    log.info(f"artifacts → {run_dir} | w_synth={w}")
    bundle = assemble(w_synth=w)
    folds_all = purged_folds(bundle)
    folds = [folds_all[i] for i in cv_folds]
    src = source_ids(bundle)

    n = len(bundle["X"])
    acc = {s: np.zeros(n, dtype=np.float64) for s in ("long", "short")}
    cnt = np.zeros(n, dtype=np.int32)
    for fold in folds:
        fit_idx, es_idx = _es_from_fold(bundle, fold)
        for seed in ex["tcn"].get("cv_seeds", ex["tcn"]["seeds"]):
            model, info = train_tcn(bundle, fit_idx, es_idx, seed,
                                    log_prefix=f"fold{fold['k']} ")
            pr = predict(model, bundle, fold["test"], src)
            acc["long"][pr["rows"]] += pr["p_long"]
            acc["short"][pr["rows"]] += pr["p_short"]
            cnt[pr["rows"]] += 1
            del model
            gc.collect()
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        save_json({"done_fold": fold["k"]}, run_dir / "progress.json")

    rows = np.flatnonzero(cnt > 0)
    p_long = (acc["long"][rows] / cnt[rows]).astype(np.float32)
    p_short = (acc["short"][rows] / cnt[rows]).astype(np.float32)
    econ = economic_score(p_long, p_short, bundle, rows)
    lb = _per_trade_lb({"long": _scatter(p_long, rows, n),
                        "short": _scatter(p_short, rows, n)}, bundle, rows)
    from sklearn.metrics import roc_auc_score

    auc_l = float(roc_auc_score(bundle["y_long"][rows], p_long))
    auc_s = float(roc_auc_score(bundle["y_short"][rows], p_short))
    pd.DataFrame({"ts": bundle["ts"][rows], "p_long": p_long,
                  "p_short": p_short}).to_parquet(TCN_OOF, index=False)
    save_json({"econ": econ, "auc_long": auc_l, "auc_short": auc_s,
               "per_trade_lb": lb, "w_synth": w,
               "blocks": blockwise_econ(
                   {"long": _scatter(p_long, rows, n),
                    "short": _scatter(p_short, rows, n)}, bundle, rows, 10)},
              run_dir / "tcn_cv.json")
    log.info(f"TCN CV DONE: auc L/S {auc_l:.4f}/{auc_s:.4f} | "
             f"netEV {econ['net_ev_sum']:+.1f} ({econ['n_trades']:,} trades) "
             f"| LB {lb:+.4f} | oof → {TCN_OOF}")


def _scatter(vals, rows, n):
    out = np.full(n, np.nan, dtype=np.float32)
    out[rows] = vals
    return out


def run_tcn_final() -> None:
    ex = experiment()
    run_dir = new_run_dir("tcn_final")
    w = _recipe_w()
    log.info(f"artifacts → {run_dir} | w_synth={w}")
    bundle = assemble(w_synth=w)
    ts = pd.DatetimeIndex(bundle["ts"])
    is_real = bundle["is_real"]
    cut = ts[is_real].max() - pd.Timedelta(days=ex["lgbm"]["early_stop_tail_days"])
    es_idx = np.flatnonzero(is_real & (ts >= cut))
    fit_idx = np.setdiff1d(np.arange(len(ts)), es_idx)

    TCN_DIR.mkdir(parents=True, exist_ok=True)
    for seed in ex["tcn"]["seeds"]:
        model, info = train_tcn(bundle, fit_idx, es_idx, seed, log_prefix="final ")
        torch.save(model.state_dict(), TCN_DIR / f"seed{seed}.pt")
        log.info(f"final seed{seed}: best val {info['best_val']:.4f}")
        del model
        gc.collect()
    save_json({"w_synth": w, "seeds": ex["tcn"]["seeds"], "channels": CHANNELS,
               "n_tab": len(bundle["feature_names"]),
               "feature_names": bundle["feature_names"],
               "tcn_cfg": ex["tcn"]}, TCN_DIR / "meta.json")
    log.info(f"TCN FINAL DONE → {TCN_DIR}")


def load_tcn_ensemble() -> tuple[list, dict]:
    meta = load_json(TCN_DIR / "meta.json")
    models = []
    for seed in meta["seeds"]:
        m = HybridTCN(len(meta["channels"]), meta["n_tab"],
                      meta["tcn_cfg"]["hidden"], meta["tcn_cfg"]["blocks"],
                      meta["tcn_cfg"]["dropout"])
        m.load_state_dict(torch.load(TCN_DIR / f"seed{seed}.pt",
                                     map_location="cpu"))
        m.eval()
        models.append(m)
    return models, meta
