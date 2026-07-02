"""Phase-3 to-EOD direction system (D-012).

y = sign of the window-end mark (r_end) at the H144 geometry — day-scale
direction, the strongest signal found (AUC ≈ 0.58). One binary model:
p_up. Gate: |2p−1|·M − cost_gate (M = training median |r_end|; gate keeps the
CFD-stressed liquidity term). Execution: hold to window end / session flat
with a safety SL (2.0·ATR, matching the cached label geometry), futures costs.
Realized economics from label plumbing: SL-code rows → −sl_atr, else ±r_end.
"""
import gc

import numpy as np
import pandas as pd

from ..config import clear_overrides, experiment, override_experiment
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger

log = get_logger("models.direction")

GEOM = {"tp_atr": 5.0, "sl_atr": 2.0, "horizon_bars": 144}
SAFETY_SL = 2.0
ART = paths.MODELS_DIR / "dir_final"


def _frame():
    override_experiment(labels=GEOM)
    from .dataset import cost_atr, load_xy

    df, fcols = load_xy("real_training")
    df["y_dir"] = (df["r_end_atr"] > 0).astype(np.int8)
    df["ca_gate"] = cost_atr(df, profile="cfd_stressed")
    df["ca_exec"] = cost_atr(df, profile="futures_proxy_stressed")
    # realized R of a to-EOD hold with safety SL, per side
    r_end = df["r_end_atr"].to_numpy(np.float32)
    df["R_up"] = np.where(df["x_long"].to_numpy() == 2, -SAFETY_SL, r_end)
    df["R_dn"] = np.where(df["x_short"].to_numpy() == 2, -SAFETY_SL, -r_end)
    return df, fcols


def _mini_bundle(df, fcols):
    return {
        "X": df[fcols].to_numpy(np.float32),
        "feature_names": fcols,
        "is_real": np.ones(len(df), dtype=bool),
        "ts": df["ts"].to_numpy(),
        "t1_long": df["t1_long"].to_numpy(),
        "t1_short": df["t1_short"].to_numpy(),
        "y_long": df["y_dir"].to_numpy(np.int8),      # reuse fold/ES plumbing
        "w_long": df["w_uniq_long"].to_numpy(np.float32),
        "y_short": df["y_dir"].to_numpy(np.int8),
        "w_short": df["w_uniq_long"].to_numpy(np.float32),
    }


def _econ(df, p_up, idx, M, gate):
    edge = np.abs(2 * p_up[idx] - 1) * M - df["ca_gate"].to_numpy()[idx]
    side_up = p_up[idx] >= 0.5
    act = edge > gate
    Rn = np.where(side_up, df["R_up"].to_numpy()[idx], df["R_dn"].to_numpy()[idx]) \
        - df["ca_exec"].to_numpy()[idx]
    r = Rn[act]
    return {"n": int(act.sum()), "sum": float(r.sum()),
            "mean": float(r.mean()) if act.any() else 0.0}


def run_search() -> None:
    from sklearn.metrics import roc_auc_score

    from .dataset import purged_folds
    from .lgbm import sample_configs, train_fold

    run_dir = new_run_dir("dir_search")
    log.info(f"artifacts → {run_dir}")
    df, fcols = _frame()
    b = _mini_bundle(df, fcols)
    folds = [purged_folds(b)[i] for i in (0, 2, 4)]
    M = float(np.median(np.abs(df["r_end_atr"])))
    log.info(f"{len(df):,} rows | base up-rate {df['y_dir'].mean():.3f} | M={M:.3f} ATR")

    results = []
    for i, cfg in enumerate(sample_configs(8, seed=99)):
        p_up = np.full(len(df), np.nan, dtype=np.float32)
        aucs = []
        for fold in folds:
            booster, _ = train_fold(b, fold, cfg, "long")
            te = fold["test"]
            p_up[te] = booster.predict(b["X"][te], num_iteration=booster.best_iteration)
            aucs.append(roc_auc_score(b["y_long"][te], p_up[te]))
            del booster
            gc.collect()
        test_idx = np.concatenate([f["test"] for f in folds])
        gates = {str(g): _econ(df, p_up, test_idx, M, g) for g in (0.05, 0.10, 0.15, 0.20, 0.30)}
        results.append({"cfg": cfg, "auc": float(np.mean(aucs)), "gates": gates})
        best_g = max(gates.items(), key=lambda kv: kv[1]["sum"] if kv[1]["n"] >= 300 else -1e9)
        log.info(f"[{i+1}/8] auc {np.mean(aucs):.4f} | best gate {best_g[0]}: "
                 f"n={best_g[1]['n']:,} sum {best_g[1]['sum']:+.1f} ({best_g[1]['mean']:+.4f}/tr)")
        save_json(results, run_dir / "dir_search.json")

    def key(r):
        vals = [v["sum"] for v in r["gates"].values() if v["n"] >= 300]
        return max(vals) if vals else -1e9

    best = max(results, key=key)
    save_json({"cfg": best["cfg"], "auc": best["auc"], "gates": best["gates"],
               "M": M}, paths.MODELS_DIR / "dir_candidates.json")
    log.info(f"DIR SEARCH DONE: auc {best['auc']:.4f} → {paths.MODELS_DIR / 'dir_candidates.json'}")
    clear_overrides()


def run_final() -> None:
    from .dataset import purged_folds
    from .lgbm import train_fold

    run_dir = new_run_dir("dir_final")
    log.info(f"artifacts → {run_dir}")
    cand = load_json(paths.MODELS_DIR / "dir_candidates.json")
    cfg = cand["cfg"]
    df, fcols = _frame()
    b = _mini_bundle(df, fcols)
    folds = purged_folds(b)

    p_up = np.full(len(df), np.nan, dtype=np.float32)
    iters = []
    for fold in folds:
        booster, info = train_fold(b, fold, cfg, "long")
        te = fold["test"]
        p_up[te] = booster.predict(b["X"][te], num_iteration=booster.best_iteration)
        iters.append(info["best_iter"])
        del booster
        gc.collect()
    test_idx = np.concatenate([f["test"] for f in folds])
    M = float(np.median(np.abs(df["r_end_atr"])))
    gate_curve = {str(g): _econ(df, p_up, test_idx, M, g)
                  for g in (0.05, 0.08, 0.10, 0.15, 0.20, 0.30)}
    # per-block stability at candidate gates
    blocks = {}
    ts_te = pd.DatetimeIndex(df["ts"].to_numpy()[test_idx])
    order = np.argsort(ts_te.asi8)
    for g in (0.10, 0.15, 0.20):
        chunks = np.array_split(test_idx[order], 10)
        blocks[str(g)] = [round(_econ(df, p_up, c, M, g)["sum"], 1) for c in chunks]

    final_fold = {"k": -1, "train": np.arange(len(df)), "test": np.array([], dtype=int),
                  "t0": pd.Timestamp.max, "t1": pd.Timestamp.max}
    booster, _ = train_fold(b, final_fold, cfg, "long",
                            fixed_rounds=max(int(np.median(iters)), 50))
    ART.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(ART / "dir.txt"))
    qs = np.linspace(0, 1, 1001)
    p_final = booster.predict(b["X"][test_idx])
    np.savez(ART / "qmap.npz", knots_final=np.quantile(p_final, qs),
             knots_oof=np.quantile(p_up[test_idx], qs))
    from sklearn.metrics import roc_auc_score

    meta = {"cfg": cfg, "geometry": GEOM, "safety_sl": SAFETY_SL, "M": M,
            "feature_names": fcols, "oof_auc": float(roc_auc_score(b["y_long"][test_idx], p_up[test_idx])),
            "gate_curve": gate_curve, "blocks": blocks,
            "fold_iters": iters}
    save_json(meta, ART / "meta.json")
    log.info(f"DIR FINAL: auc {meta['oof_auc']:.4f} | gates "
             + " | ".join(f"{g}: {v['sum']:+.0f}/{v['n']}" for g, v in gate_curve.items())
             + f" → {ART}")
    for g, bl in blocks.items():
        log.info(f"  blocks @gate {g}: +{sum(1 for x in bl if x > 0)}/10 {bl}")
    clear_overrides()


def predict_dir(feat: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    meta = load_json(ART / "meta.json")
    X = feat[meta["feature_names"]].to_numpy(np.float32)
    raw = lgb.Booster(model_file=str(ART / "dir.txt")).predict(X)
    q = np.load(ART / "qmap.npz")
    return np.interp(raw, q["knots_final"], q["knots_oof"]).astype(np.float32)


def validation_look(gate: float, risk_per_trade: float | None = None) -> None:
    """ONE ledgered validation look for the Phase-3 direction system."""
    from ..backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
    from ..data.loader import load_bars
    from ..eval.report import render_backtest
    from ..eval.validate import _count_looks, append_ledger, evaluate_criteria
    from ..features.registry import build_features
    from ..utils.hashio import save_json as sj
    from .dataset import cost_atr

    override_experiment(labels=GEOM)
    run_dir = new_run_dir("validate_dir")
    log.info(f"artifacts → {run_dir} | gate {gate}")
    meta = load_json(ART / "meta.json")
    feat = build_features("real_validation")
    p_up = predict_dir(feat)
    ca_gate = cost_atr(feat, profile="cfd_stressed")
    edge = np.abs(2 * p_up - 1) * meta["M"] - ca_gate
    side = np.where(edge > gate, np.where(p_up >= 0.5, 1, -1), 0).astype(np.int8)
    sig = feat[["ts", "avail_ts", "day"]].copy()
    from ..config import instrument

    sig["atr_abs"] = feat["_atr_points"] * float(instrument()["point_size"])
    sig["side"] = side
    log.info(f"signals: {int((side != 0).sum()):,}")
    df1m = load_bars("real_validation")
    res = run_backtest(df1m, sig, make_cost_cfg(),
                       make_risk_cfg(risk_per_trade),
                       tp_atr=99.0, sl_atr=SAFETY_SL)
    m = render_backtest(run_dir, "dir_system", res,
                        experiment()["backtest"]["equity0"], df1m)
    checks = evaluate_criteria(m, res["trades"], res["daily"], _count_looks() + 1)
    sj(checks, run_dir / "criteria.json")
    n_pass = sum(1 for k, v in checks.items() if not k.startswith("_") and v)
    n_all = sum(1 for k in checks if not k.startswith("_"))
    append_ledger(f"Phase-3 direction system (gate {gate}, risk {risk_per_trade})",
                  f"{m['n_trades']} trades, PF {m.get('profit_factor', 0):.3f}, "
                  f"Sharpe {m['sharpe']:.2f}, DD {m['max_dd_pct']:.1f}%, "
                  f"ret {m['total_return_pct']:+.1f}% (B&H {m.get('bh_total_return_pct', 0):+.1f}%). "
                  f"Criteria {n_pass}/{n_all} → {run_dir.name}")
    clear_overrides()
