"""LightGBM pipeline stages (search → synth gate → final), checkpointed.

Protocol (decisions ledger D-002):
  • all selection happens on REAL-fold OOF economics inside TRAINING;
  • real-only must show a positive bootstrap-LB edge before pooling synth;
  • synthetic pooling is admitted only if it beats real-only on the same folds;
  • validation is NOT touched by any stage here.
"""
import gc
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from ..config import experiment
from ..eval.metrics import bootstrap_lb
from ..eval.pbo import pbo_cscv
from ..eval.plotstyle import CRITICAL, GOOD, MUTED, SERIES, apply_style
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json, sha256_obj
from ..utils.log import get_logger
from .dataset import assemble, purged_folds
from .lgbm import (SIDES, blockwise_econ, economic_score, run_oof,
                   sample_configs, train_fold, _params)

log = get_logger("models.pipeline")

CANDIDATES = paths.MODELS_DIR / "lgbm_candidates.json"
RECIPE = paths.MODELS_DIR / "lgbm_recipe.json"


def _rank_key(res: dict) -> float:
    econ = res["econ"]
    if econ["n_trades"] < 300:          # no statistical mass → not rankable
        return -1e9
    return econ["net_ev_sum"]


def run_search(n_configs: int | None = None, search_folds: tuple = (0, 2, 4)) -> None:
    ex = experiment()
    n_configs = n_configs or ex["lgbm"]["n_search"]
    run_dir = new_run_dir("lgbm_search")
    log.info(f"artifacts → {run_dir}")

    bundle = assemble(w_synth=0.0)
    folds_all = purged_folds(bundle)
    folds = [folds_all[i] for i in search_folds]
    configs = sample_configs(n_configs, seed=ex["seed"])

    results = []
    for i, cfg in enumerate(configs):
        r = run_oof(bundle, folds, cfg)
        blocks = blockwise_econ(r["oof"], bundle, r["test_idx"], n_blocks=10)
        gap_l = np.mean([d["auc_train_long"] - d["auc_test_long"] for d in r["diags"]])
        gap_s = np.mean([d["auc_train_short"] - d["auc_test_short"] for d in r["diags"]])
        rec = {"cfg": cfg, "econ": r["econ"], "auc_test_long": r["auc_test_long"],
               "auc_test_short": r["auc_test_short"],
               "auc_gap_mean": float((gap_l + gap_s) / 2), "blocks": blocks}
        results.append(rec)
        save_json(results, run_dir / "search_results.json")
        log.info(
            f"[{i + 1}/{n_configs}] auc L/S {r['auc_test_long']:.4f}/{r['auc_test_short']:.4f} "
            f"gap {rec['auc_gap_mean']:.4f} | trades {r['econ']['n_trades']:,} "
            f"| netEV {r['econ']['net_ev_sum']:+.1f} ({r['econ']['net_ev_mean']:+.4f}/tr)"
        )
        del r
        gc.collect()

    M = np.array([r["blocks"] for r in results])
    pbo = pbo_cscv(M)
    order = np.argsort([-_rank_key(r) for r in results])
    best, second = results[order[0]], results[order[1]]
    save_json({"best": best, "second": second, "pbo": pbo,
               "n_configs": n_configs, "search_folds": list(search_folds),
               "run_dir": str(run_dir)}, CANDIDATES)

    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    aucs = [(r["auc_test_long"] + r["auc_test_short"]) / 2 for r in results]
    evs = [r["econ"]["net_ev_sum"] for r in results]
    axs[0].scatter(aucs, evs, color=SERIES[0], s=24)
    axs[0].scatter([(best["auc_test_long"] + best["auc_test_short"]) / 2],
                   [best["econ"]["net_ev_sum"]], color=GOOD, s=60)
    axs[0].set_xlabel("mean OOF AUC")
    axs[0].set_ylabel("OOF net EV (ATR)")
    axs[0].set_title("configs: AUC vs economics")
    axs[1].hist([r["auc_gap_mean"] for r in results], bins=15, color=SERIES[0])
    axs[1].set_title("train−test AUC gap (memorization watch)")
    fig.savefig(run_dir / "plots" / "search_overview.png")
    plt.close(fig)

    log.info(f"SEARCH DONE. best netEV {best['econ']['net_ev_sum']:+.1f} "
             f"({best['econ']['n_trades']:,} trades) | PBO {pbo['pbo']:.2f} "
             f"| candidates → {CANDIDATES}")


BARRIER_GRID = [
    {"tp_atr": 3.0, "sl_atr": 1.5, "horizon_bars": 48},   # day-trade control
    {"tp_atr": 4.0, "sl_atr": 1.5, "horizon_bars": 552,   # ~2-day swing
     "overnight": True},
    {"tp_atr": 6.0, "sl_atr": 2.0, "horizon_bars": 828,   # ~3-day swing
     "overnight": True},
    {"tp_atr": 8.0, "sl_atr": 2.5, "horizon_bars": 1380,  # ~5-day swing
     "overnight": True},
]


def run_barriers() -> None:
    """Training-side sweep of label geometry on the best search config.
    Labels are rebuilt per geometry; scored by real-fold OOF economics."""
    from ..config import clear_overrides, override_experiment
    from ..labels.triple_barrier import build_labels

    run_dir = new_run_dir("lgbm_barriers")
    log.info(f"artifacts → {run_dir}")
    cand = load_json(CANDIDATES)
    cfg = cand["best"]["cfg"]
    results = []
    for geom in BARRIER_GRID:
        clear_overrides()
        override_experiment(labels=geom)
        build_labels("real_training")           # builds/caches this geometry
        bundle = assemble(w_synth=0.0)
        folds = purged_folds(bundle)
        folds = [folds[i] for i in (0, 2, 4)]
        r = run_oof(bundle, folds, cfg)
        gate_curve = {}
        for g in (0.05, 0.10, 0.15, 0.20, 0.30):
            gate_curve[str(g)] = economic_score(
                r["oof"]["long"][r["test_idx"]], r["oof"]["short"][r["test_idx"]],
                bundle, r["test_idx"], min_ev=g)
        rec = {"geom": geom, "econ": r["econ"],
               "auc_test_long": r["auc_test_long"],
               "auc_test_short": r["auc_test_short"],
               "per_trade_lb": _per_trade_lb(r["oof"], bundle, r["test_idx"]),
               "gate_curve": gate_curve,
               "blocks": blockwise_econ(r["oof"], bundle, r["test_idx"], 10)}
        results.append(rec)
        save_json(results, run_dir / "barrier_results.json")
        log.info(f"geom {geom}: netEV {r['econ']['net_ev_sum']:+.1f} "
                 f"({r['econ']['n_trades']:,} trades, "
                 f"{r['econ']['net_ev_mean']:+.4f}/tr, LB {rec['per_trade_lb']:+.4f})")
        del bundle, r
        gc.collect()
    clear_overrides()
    best = max(results, key=lambda x: x["econ"]["net_ev_sum"]
               if x["econ"]["n_trades"] >= 300 else -1e9)
    save_json({"best_geom": best["geom"], "results": results},
              paths.MODELS_DIR / "lgbm_barriers.json")
    log.info(f"BARRIERS DONE: best {best['geom']} "
             f"netEV {best['econ']['net_ev_sum']:+.1f}")


def run_synth_gate(weights: tuple = (0.25, 0.5, 1.0)) -> None:
    ex = experiment()
    run_dir = new_run_dir("lgbm_synth")
    log.info(f"artifacts → {run_dir}")
    cand = load_json(CANDIDATES)
    cfg = cand["best"]["cfg"]

    # head-to-head baseline: real-only on the FULL 5 folds
    bundle = assemble(w_synth=0.0)
    folds = purged_folds(bundle)
    base = run_oof(bundle, folds, cfg)
    base_blocks = blockwise_econ(base["oof"], bundle, base["test_idx"], 10)
    base_lb = bootstrap_lb(np.array(base_blocks, dtype=np.float64) /
                           max(base["econ"]["n_trades"], 1) * 10, q=0.05)
    results = {"real_only": {"econ": base["econ"],
                             "auc_test_long": base["auc_test_long"],
                             "auc_test_short": base["auc_test_short"],
                             "blocks": base_blocks}}
    # gate 1: real-only must have positive edge (block-bootstrap LB of per-trade EV)
    rows_real = base["test_idx"]
    per_trade_lb = _per_trade_lb(base["oof"], bundle, rows_real)
    results["real_only"]["per_trade_ev_lb95"] = per_trade_lb
    log.info(f"real-only: netEV {base['econ']['net_ev_sum']:+.1f} "
             f"| per-trade EV LB95 {per_trade_lb:+.4f} ATR")
    del bundle, base
    gc.collect()

    if per_trade_lb <= 0:
        log.warning("GATE FAILED: real-only edge LB ≤ 0 — synthetic pooling NOT "
                    "attempted (it may not rescue a non-edge). Recording and stopping.")
        save_json(results, run_dir / "synth_gate.json")
        save_json({"cfg": cfg, "w_synth": 0.0, "gate": "real_only_failed",
                   "evidence": results}, RECIPE)
        return

    for w in weights:
        bundle = assemble(w_synth=w)
        folds = purged_folds(bundle)
        r = run_oof(bundle, folds, cfg)
        results[f"w_{w}"] = {"econ": r["econ"],
                             "auc_test_long": r["auc_test_long"],
                             "auc_test_short": r["auc_test_short"],
                             "blocks": blockwise_econ(r["oof"], bundle, r["test_idx"], 10)}
        log.info(f"w_synth={w}: netEV {r['econ']['net_ev_sum']:+.1f} "
                 f"({r['econ']['n_trades']:,} trades) "
                 f"auc L/S {r['auc_test_long']:.4f}/{r['auc_test_short']:.4f}")
        save_json(results, run_dir / "synth_gate.json")
        del bundle, r
        gc.collect()

    base_ev = results["real_only"]["econ"]["net_ev_sum"]
    best_w, best_ev = 0.0, base_ev
    for w in weights:
        ev = results[f"w_{w}"]["econ"]["net_ev_sum"]
        if ev > best_ev:
            best_w, best_ev = w, ev
    save_json({"cfg": cfg, "w_synth": best_w, "gate": "passed",
               "evidence": results}, RECIPE)
    log.info(f"SYNTH GATE DONE: chosen w_synth={best_w} "
             f"(netEV {best_ev:+.1f} vs real-only {base_ev:+.1f}) → {RECIPE}")


def _per_trade_lb(oof: dict, bundle: dict, idx: np.ndarray) -> float:
    dec = experiment()["decision"]
    lab = experiment()["labels"]
    p_l, p_s = oof["long"][idx], oof["short"][idx]
    cost = bundle["cost_atr"][idx]
    ev_l = p_l * lab["tp_atr"] - (1 - p_l) * lab["sl_atr"] - cost
    ev_s = p_s * lab["tp_atr"] - (1 - p_s) * lab["sl_atr"] - cost
    take_long = ev_l >= ev_s
    ev = np.maximum(ev_l, ev_s)
    p = np.where(take_long, p_l, p_s)
    act = (ev > dec["min_ev_atr"]) & (p > dec["prob_floor"])
    R = np.where(take_long, bundle["R_long"][idx], bundle["R_short"][idx]) - cost
    return bootstrap_lb(R[act], q=0.05) if act.sum() > 50 else float("-inf")


def run_final(w_override: float | None = None, out_name: str = "lgbm_final") -> None:
    ex = experiment()
    run_dir = new_run_dir(out_name)
    log.info(f"artifacts → {run_dir}")
    recipe = load_json(RECIPE)
    cfg = recipe["cfg"]
    w = recipe["w_synth"] if w_override is None else w_override
    log.info(f"final recipe: w_synth={w} cfg={cfg}")

    bundle = assemble(w_synth=w)
    folds = purged_folds(bundle)
    r = run_oof(bundle, folds, cfg)
    test_idx = r["test_idx"]

    # permutation sanity: shuffled labels must destroy the signal
    rng = np.random.default_rng(0)
    perm_bundle = dict(bundle)
    perm_bundle["y_long"] = bundle["y_long"].copy()
    real_rows = np.flatnonzero(bundle["is_real"])
    perm_bundle["y_long"][real_rows] = rng.permutation(bundle["y_long"][real_rows])
    booster, _ = train_fold(perm_bundle, folds[0], cfg, "long")
    te = folds[0]["test"]
    p_perm = booster.predict(bundle["X"][te], num_iteration=booster.best_iteration)
    perm_auc = float(roc_auc_score(bundle["y_long"][te], p_perm))
    del booster, perm_bundle
    gc.collect()

    # isotonic calibration on OOF (real rows) — for REPORTING only; the gate
    # path uses the quantile map below (the policy validated in CV gated on
    # raw fold-model probabilities, so the deployed distribution must match)
    calibrators = {}
    for side in SIDES:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(r["oof"][side][test_idx], bundle[f"y_{side}"][test_idx])
        calibrators[side] = iso

    # final boosters on ALL data at CV-derived capacity (no early stopping —
    # a calm ES tail would truncate the model far below the validated folds)
    final_fold = {"k": -1, "train": np.arange(len(bundle["X"])),
                  "test": np.array([], dtype=int),
                  "t0": pd.Timestamp.max, "t1": pd.Timestamp.max}
    boosters = {}
    for side in SIDES:
        med_iter = int(np.median([d[f"best_iter_{side}"] for d in r["diags"]]))
        boosters[side], info = train_fold(bundle, final_fold, cfg, side,
                                          fixed_rounds=max(med_iter, 50))
        log.info(f"final {side}: fixed_rounds {info['best_iter']} "
                 f"(fold medians {med_iter})")

    # rank-preserving quantile map: final-model raw output distribution →
    # OOF raw distribution (fit on TRAINING rows only)
    qs = np.linspace(0.0, 1.0, 1001)
    art_dir = paths.MODELS_DIR / out_name
    art_dir.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        p_final = boosters[side].predict(bundle["X"][test_idx])
        knots_final = np.quantile(p_final, qs)
        knots_oof = np.quantile(r["oof"][side][test_idx], qs)
        np.savez(art_dir / f"qmap_{side}.npz", knots_final=knots_final,
                 knots_oof=knots_oof)

    # ── persist artifact ─────────────────────────────────────────────
    meta = {"cfg": cfg, "w_synth": w, "feature_names": bundle["feature_names"],
            "oof_econ": r["econ"], "auc_test_long": r["auc_test_long"],
            "auc_test_short": r["auc_test_short"], "perm_auc_fold0": perm_auc,
            "fold_diags": r["diags"],
            "label_cfg": ex["labels"], "decision_cfg": ex["decision"]}
    for side in SIDES:
        boosters[side].save_model(str(art_dir / f"{side}.txt"))
        with open(art_dir / f"iso_{side}.pkl", "wb") as f:
            pickle.dump(calibrators[side], f)
    meta["hash"] = sha256_obj(meta)
    save_json(meta, art_dir / "meta.json")

    # OOF predictions for stacking (real rows only)
    oof_df = pd.DataFrame({
        "ts": bundle["ts"][test_idx],
        "p_long": r["oof"]["long"][test_idx],
        "p_short": r["oof"]["short"][test_idx],
        "y_long": bundle["y_long"][test_idx],
        "y_short": bundle["y_short"][test_idx],
    })
    oof_df.to_parquet(art_dir / "oof_real.parquet", index=False)

    # diagnostics plots
    apply_style()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for side, col in (("long", SERIES[0]), ("short", SERIES[5])):
        p = r["oof"][side][test_idx]
        y = bundle[f"y_{side}"][test_idx]
        bins = np.quantile(p, np.linspace(0, 1, 21))
        binned = pd.cut(p, np.unique(bins), include_lowest=True)
        obs = pd.Series(y).groupby(binned, observed=True).mean()
        mid = pd.Series(p).groupby(binned, observed=True).mean()
        axs[0, 0].plot(mid.values, obs.values, color=col, marker="o", markersize=3,
                       linewidth=1.2, label=side)
        axs[0, 1].hist(p, bins=60, color=col, alpha=0.55, label=side)
    lims = axs[0, 0].get_xlim()
    axs[0, 0].plot(lims, lims, color=MUTED, linewidth=0.8, linestyle="--")
    axs[0, 0].set_title("OOF calibration (pre-isotonic)")
    axs[0, 0].legend()
    axs[0, 1].set_title("OOF probability distribution")
    axs[0, 1].legend()

    imp = pd.Series(boosters["long"].feature_importance("gain"),
                    index=bundle["feature_names"]).nlargest(25)[::-1]
    axs[1, 0].barh(imp.index, imp.values, color=SERIES[0])
    axs[1, 0].set_title("feature importance (gain, long)")
    axs[1, 0].tick_params(labelsize=6)

    gaps = [d["auc_train_long"] - d["auc_test_long"] for d in r["diags"]]
    axs[1, 1].bar(range(len(gaps)), gaps, color=SERIES[0], width=0.6)
    axs[1, 1].axhline(0, color=MUTED, linewidth=0.8)
    axs[1, 1].set_title(f"per-fold AUC gap | perm AUC {perm_auc:.3f}")
    fig.savefig(run_dir / "plots" / "final_diagnostics.png")
    plt.close(fig)

    log.info(f"FINAL DONE: OOF netEV {r['econ']['net_ev_sum']:+.1f} "
             f"({r['econ']['n_trades']:,} trades) | perm AUC {perm_auc:.3f} "
             f"| artifact → {art_dir}")
