"""LightGBM training: purged-CV random search, economic OOF scoring,
synthetic-pooling gates, diagnostics, final fit + calibration.

Stages (each checkpointed into the run dir):
  search : real-only random search — configs ranked by OOF net EV at the
           default cost-aware gate (AUC is diagnostic, never the objective).
  synth  : top config re-run with pooled synthetic at w ∈ grid; admitted only
           if it beats real-only on the same real folds (protocol D-002).
  final  : chosen recipe → full OOF preds, isotonic calibration, permutation
           test, diagnostics plots, final boosters saved.
"""
import gc
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from ..config import experiment
from ..utils import paths
from ..utils.hashio import save_json, sha256_obj
from ..utils.log import get_logger
from .dataset import assemble, purged_folds

log = get_logger("models.lgbm")

SIDES = ("long", "short")


# ── config space ─────────────────────────────────────────────────────
def sample_configs(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    grid = {
        "num_leaves": [15, 31, 63],
        "max_depth": [5, 7, 9],
        "min_child_samples": [300, 1000, 3000],
        "feature_fraction": [0.6, 0.75, 0.9],
        "bagging_fraction": [0.6, 0.8],
        "lambda_l2": [1.0, 5.0, 20.0],
        "learning_rate": [0.03, 0.05],
    }
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def _params(cfg: dict) -> dict:
    return {
        "objective": "binary", "verbose": -1, "bagging_freq": 1,
        "seed": 42, "feature_fraction_seed": 43, "bagging_seed": 44,
        "num_threads": 0,
        **{k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
           for k, v in cfg.items()},
    }


# ── fold training ────────────────────────────────────────────────────
def _es_split(bundle: dict, fold: dict) -> tuple[np.ndarray, np.ndarray]:
    """Early-stop set: last `early_stop_tail_days` of REAL train rows before
    the test block (falls back to earliest post-test rows)."""
    tail_days = experiment()["lgbm"]["early_stop_tail_days"]
    ts = pd.DatetimeIndex(bundle["ts"])
    tr = fold["train"]
    real_tr = tr[bundle["is_real"][tr]]
    pre = real_tr[ts[real_tr] < fold["t0"]]
    if len(pre) > 5000:
        cut = ts[pre].max() - pd.Timedelta(days=tail_days)
        es = pre[ts[pre] >= cut]
        fit = np.setdiff1d(tr, es, assume_unique=False)
    else:
        post = real_tr[ts[real_tr] > fold["t1"]]
        cut = ts[post].min() + pd.Timedelta(days=tail_days)
        es = post[ts[post] <= cut]
        fit = np.setdiff1d(tr, es, assume_unique=False)
    return fit, es


def train_fold(bundle: dict, fold: dict, cfg: dict, side: str,
               n_rounds: int = 1500,
               fixed_rounds: int | None = None) -> tuple[lgb.Booster, dict]:
    """fixed_rounds: train the full train side at a preset round count with no
    early stopping (used for final refits at CV-derived capacity)."""
    y = bundle[f"y_{side}"]
    w = bundle[f"w_{side}"]
    if fixed_rounds is not None:
        tr = fold["train"]
        dtrain = lgb.Dataset(bundle["X"][tr], label=y[tr], weight=w[tr],
                             feature_name=bundle["feature_names"], free_raw_data=True)
        booster = lgb.train(_params(cfg), dtrain, num_boost_round=int(fixed_rounds))
        del dtrain
        gc.collect()
        return booster, {"best_iter": int(fixed_rounds)}
    fit_idx, es_idx = _es_split(bundle, fold)
    dtrain = lgb.Dataset(bundle["X"][fit_idx], label=y[fit_idx], weight=w[fit_idx],
                         feature_name=bundle["feature_names"], free_raw_data=True)
    dvalid = lgb.Dataset(bundle["X"][es_idx], label=y[es_idx], weight=w[es_idx],
                         reference=dtrain, free_raw_data=True)
    booster = lgb.train(_params(cfg), dtrain, num_boost_round=n_rounds,
                        valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    info = {"best_iter": booster.best_iteration}
    del dtrain, dvalid
    gc.collect()
    return booster, info


# ── economic OOF scoring ─────────────────────────────────────────────
def economic_score(p_long, p_short, bundle, idx, min_ev=None, prob_floor=None) -> dict:
    """Label-implied net EV of the gated policy on rows idx (ATR units)."""
    dec = experiment()["decision"]
    lab = experiment()["labels"]
    min_ev = dec["min_ev_atr"] if min_ev is None else min_ev
    prob_floor = dec["prob_floor"] if prob_floor is None else prob_floor
    tp, sl = lab["tp_atr"], lab["sl_atr"]
    cost = bundle["cost_atr"][idx]
    ev_l = p_long * tp - (1 - p_long) * sl - cost
    ev_s = p_short * tp - (1 - p_short) * sl - cost
    take_long = ev_l >= ev_s
    ev = np.where(take_long, ev_l, ev_s)
    p = np.where(take_long, p_long, p_short)
    act = (ev > min_ev) & (p > prob_floor)
    R_net = np.where(take_long, bundle["R_long"][idx], bundle["R_short"][idx]) - cost
    net = R_net[act]
    return {
        "n_trades": int(act.sum()),
        "net_ev_sum": float(net.sum()),
        "net_ev_mean": float(net.mean()) if act.any() else 0.0,
        "hit_rate": float((net > 0).mean()) if act.any() else 0.0,
    }


def run_oof(bundle: dict, folds: list[dict], cfg: dict) -> dict:
    """OOF probabilities on real test rows for both sides + fold diagnostics."""
    n = len(bundle["X"])
    oof = {s: np.full(n, np.nan, dtype=np.float32) for s in SIDES}
    diags = []
    for fold in folds:
        d = {"k": fold["k"]}
        for side in SIDES:
            booster, info = train_fold(bundle, fold, cfg, side)
            te = fold["test"]
            oof[side][te] = booster.predict(
                bundle["X"][te], num_iteration=booster.best_iteration)
            tr_s = fold["train"][:: max(1, len(fold["train"]) // 20000)]
            p_tr = booster.predict(bundle["X"][tr_s], num_iteration=booster.best_iteration)
            d[f"auc_train_{side}"] = float(roc_auc_score(bundle[f"y_{side}"][tr_s], p_tr))
            d[f"auc_test_{side}"] = float(
                roc_auc_score(bundle[f"y_{side}"][te], oof[side][te]))
            d[f"best_iter_{side}"] = info["best_iter"]
            del booster
            gc.collect()
        diags.append(d)
    test_all = np.concatenate([f["test"] for f in folds])
    econ = economic_score(oof["long"][test_all], oof["short"][test_all], bundle, test_all)
    auc_l = float(np.mean([d["auc_test_long"] for d in diags]))
    auc_s = float(np.mean([d["auc_test_short"] for d in diags]))
    return {"oof": oof, "diags": diags, "econ": econ,
            "auc_test_long": auc_l, "auc_test_short": auc_s,
            "test_idx": test_all}


# ── per-block economics (for PBO / stability) ────────────────────────
def blockwise_econ(oof: dict, bundle: dict, test_idx: np.ndarray, n_blocks: int = 10) -> list[float]:
    ts = pd.DatetimeIndex(bundle["ts"][test_idx])
    order = np.argsort(ts.asi8)
    blocks = np.array_split(test_idx[order], n_blocks)
    return [economic_score(oof["long"][b], oof["short"][b], bundle, b)["net_ev_sum"]
            for b in blocks]
