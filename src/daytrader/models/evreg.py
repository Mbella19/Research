"""B4 (D-026): EV-regression challenger — predict realized R per side directly.

The classification head asks "will a tp/sl bracket win?" and converts p to EV
through the bracket algebra plus a drift prior. This head skips the
conversion: LightGBM regression on the label-implied realized R (ATR units,
gross of costs) per side. The instrument's drift is embedded in the label
itself, so no prior term exists to mis-specify (the D-020/D-023 lesson).

Discipline identical to the classification path: same assemble/purged-folds/
embargoed-tail early stop, ONE config inherited from the classification
search winner (no second search surface), gate threshold chosen on the OOF
EV plateau (training-side), fixed-rounds final refit at fold-median capacity,
permutation collapse check. Signals: side = argmax EV_hat, trade iff
EV_hat − gate_cost > threshold.
"""
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import experiment
from ..utils import paths
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger
from .dataset import assemble, purged_folds
from .lgbm import _es_split

log = get_logger("evreg")

SIDES = ("long", "short")


def _params(cfg: dict) -> dict:
    p = {"objective": "regression", "metric": "l2", "verbose": -1,
         "bagging_freq": 1, "seed": 42, "feature_fraction_seed": 43,
         "bagging_seed": 44, "num_threads": 0}
    p.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
              for k, v in cfg.items()})
    p["num_leaves"] = int(cfg.get("num_leaves", 31))
    p["max_depth"] = int(cfg.get("max_depth", 5))
    p["min_child_samples"] = int(cfg.get("min_child_samples", 3000))
    return p


def _train_fold(bundle, fold, cfg, side, fixed_rounds=None):
    y = bundle[f"R_{side}"]
    w = bundle[f"w_{side}"]
    if fixed_rounds is not None:
        tr = fold["train"]
        dtrain = lgb.Dataset(bundle["X"][tr], label=y[tr], weight=w[tr],
                             feature_name=bundle["feature_names"], free_raw_data=True)
        booster = lgb.train(_params(cfg), dtrain, num_boost_round=int(fixed_rounds))
        del dtrain
        gc.collect()
        return booster, int(fixed_rounds)
    fit_idx, es_idx = _es_split(bundle, fold)
    dtrain = lgb.Dataset(bundle["X"][fit_idx], label=y[fit_idx], weight=w[fit_idx],
                         feature_name=bundle["feature_names"], free_raw_data=True)
    dvalid = lgb.Dataset(bundle["X"][es_idx], label=y[es_idx], weight=w[es_idx],
                         reference=dtrain, free_raw_data=True)
    booster = lgb.train(_params(cfg), dtrain, num_boost_round=1500,
                        valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    it = booster.best_iteration
    del dtrain, dvalid
    gc.collect()
    return booster, int(it)


def _inherited_cfg() -> dict:
    rp = paths.MODELS_DIR / "lgbm_recipe.json"
    if rp.exists():
        r = load_json(rp)
        for k in ("cfg", "best_cfg", "config"):
            if k in r:
                return r[k]
    return load_json(paths.MODELS_DIR / "lgbm_final" / "meta.json")["cfg"]


def _gate_scan(ev_l, ev_s, bundle, mask) -> list[dict]:
    """OOF net economics across gate thresholds (internal-score convention:
    bundle cost for both gate and net, like lgbm.economic_score)."""
    cost = bundle["cost_atr"]
    take_long = ev_l >= ev_s
    ev = np.where(take_long, ev_l, ev_s)
    Rn = np.where(take_long, bundle["R_long"], bundle["R_short"]) - cost
    yrs = pd.DatetimeIndex(bundle["ts"]).year
    rows = []
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        act = mask & (ev - cost > thr)
        if act.sum() < 50:
            rows.append({"thr": thr, "n": int(act.sum())})
            continue
        by_year = {int(y): {"n": int((act & (yrs == y)).sum()),
                            "mean": round(float(Rn[act & (yrs == y)].mean()), 4)
                            if (act & (yrs == y)).any() else None}
                   for y in np.unique(yrs)}
        n_long = int((act & take_long).sum())
        rows.append({"thr": thr, "n": int(act.sum()),
                     "net_mean": round(float(Rn[act].mean()), 4),
                     "net_sum": round(float(Rn[act].sum()), 2),
                     "n_long": n_long, "n_short": int(act.sum()) - n_long,
                     "long_mean": round(float(Rn[act & take_long].mean()), 4) if n_long else None,
                     "short_mean": round(float(Rn[act & ~take_long].mean()), 4)
                     if (act & ~take_long).any() else None,
                     "by_year": by_year})
    return rows


def run_evreg(out_name: str = "lgbm_evreg_v3") -> None:
    cfg = _inherited_cfg()
    log.info(f"EV-regression challenger, inherited cfg: {cfg}")
    bundle = assemble(0.0)
    folds = purged_folds(bundle)
    n = len(bundle["X"])
    oof = {s: np.full(n, np.nan, dtype=np.float32) for s in SIDES}
    iters: dict = {s: [] for s in SIDES}
    for fold in folds:
        for side in SIDES:
            booster, it = _train_fold(bundle, fold, cfg, side)
            te = fold["test"]
            oof[side][te] = booster.predict(
                bundle["X"][te], num_iteration=booster.best_iteration)
            iters[side].append(it)
            del booster
            gc.collect()
        log.info(f"fold {fold['k']}: iters {[iters[s][-1] for s in SIDES]}")

    mask = np.isfinite(oof["long"]) & np.isfinite(oof["short"])
    # OOF diagnostics: correlation with realized R + shuffled-label baseline
    diag = {}
    for side in SIDES:
        r = bundle[f"R_{side}"][mask]
        e = oof[side][mask]
        diag[f"corr_{side}"] = round(float(np.corrcoef(e, r)[0, 1]), 4)
    rng = np.random.default_rng(0)
    sh = rng.permutation(bundle["R_long"][mask])
    diag["corr_long_shuffled"] = round(float(np.corrcoef(oof["long"][mask], sh)[0, 1]), 4)
    log.info(f"OOF EV↔R correlation: {diag}")

    scan = _gate_scan(oof["long"], oof["short"], bundle, mask)
    for row in scan:
        log.info(f"gate {row['thr']:.2f}: {({k: v for k, v in row.items() if k != 'by_year'})}")

    # final refit at fold-median capacity on ALL training rows
    art = paths.MODELS_DIR / out_name
    art.mkdir(parents=True, exist_ok=True)
    all_fold = {"train": np.arange(n)}
    for side in SIDES:
        med = int(np.median(iters[side]))
        booster, _ = _train_fold(bundle, all_fold, cfg, side, fixed_rounds=med)
        booster.save_model(str(art / f"{side}.txt"))
        del booster
        gc.collect()
    pd.DataFrame({"ts": bundle["ts"], "ev_long": oof["long"],
                  "ev_short": oof["short"]}).to_parquet(art / "oof_ev.parquet")
    save_json({"cfg": cfg, "kind": "ev_regression", "diag": diag,
               "fold_iters": iters, "gate_scan": scan,
               "feature_names": bundle["feature_names"]}, art / "meta.json")
    log.info(f"EVREG DONE → {art}")
