"""Robustness battery on the chosen recipe — ONE pre-registered validation
look (one recipe, many stress slices). Ledgered as a single entry.

Slices: cost multiplier curve (1.0→2.0×), slippage 2×, EV threshold ±20%,
volatility-tercile and session slices of the base run's trades, synthetic
U1–U5 cross-check (veto power only), and — if the recipe pools synth —
real-only survival via the `lgbm_final_realonly` artifact.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
from ..backtest.signals import from_probabilities
from ..config import experiment, synth_sources
from ..data.loader import load_bars
from ..features.registry import build_features
from ..labels.triple_barrier import build_labels
from ..models.dataset import cost_atr, realized_R
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger
from .metrics import compute_metrics
from .plotstyle import MUTED, SERIES, apply_style
from .validate import append_ledger, predict_champion

log = get_logger("eval.robustness")


def _signals(feat, p_l, p_s, min_ev=None):
    dec = experiment()["decision"]
    lab = experiment()["labels"]
    ca = cost_atr(feat, profile=dec.get("gate_cost_profile"))
    from ..models.dataset import drift_atr

    return from_probabilities(feat, np.nan_to_num(p_l), np.nan_to_num(p_s),
                              lab["tp_atr"], lab["sl_atr"], ca,
                              dec["min_ev_atr"] if min_ev is None else min_ev,
                              dec["prob_floor"],
                              allowed_sides=dec.get("allowed_sides", "both"),
                              drift_atr=drift_atr(feat))


def run_battery(which: str | None = None) -> None:
    run_dir = new_run_dir("robustness")
    log.info(f"artifacts → {run_dir}")
    eq0 = experiment()["backtest"]["equity0"]
    feat = build_features("real_validation")
    (p_l, p_s), name = predict_champion(feat, which)
    df1m = load_bars("real_validation")
    risk = make_risk_cfg()
    out: dict = {"champion": name}

    # cost multiplier curve (+ slippage stress at 1.5×)
    curve = []
    for cm in experiment()["robustness"]["cost_curve"]:
        res = run_backtest(df1m, _signals(feat, p_l, p_s),
                           make_cost_cfg(stress_mult=cm), risk)
        m = compute_metrics(res["trades"], res["daily"], eq0)
        curve.append({"cost_mult": cm, "pf": m.get("profit_factor"),
                      "exp_R": m.get("expectancy_R"), "ret": m["total_return_pct"],
                      "n": m["n_trades"]})
        log.info(f"cost×{cm}: PF {m.get('profit_factor', 0):.3f} "
                 f"exp {m.get('expectancy_R', 0):+.3f}R n={m['n_trades']}")
    out["cost_curve"] = curve

    res_slip = run_backtest(df1m, _signals(feat, p_l, p_s),
                            make_cost_cfg(stress_mult=1.5, slip_mult=2.0), risk)
    m_slip = compute_metrics(res_slip["trades"], res_slip["daily"], eq0)
    out["stress_1p5x_2xslip"] = {"pf": m_slip.get("profit_factor"),
                                 "exp_R": m_slip.get("expectancy_R"),
                                 "n": m_slip["n_trades"]}

    # threshold ±20%
    base_ev = experiment()["decision"]["min_ev_atr"]
    thr = {}
    for f_, tag in ((0.8, "-20%"), (1.0, "base"), (1.2, "+20%")):
        res = run_backtest(df1m, _signals(feat, p_l, p_s, min_ev=base_ev * f_),
                           make_cost_cfg(), risk)
        m = compute_metrics(res["trades"], res["daily"], eq0)
        thr[tag] = {"pf": m.get("profit_factor"), "exp_R": m.get("expectancy_R"),
                    "n": m["n_trades"], "ret": m["total_return_pct"]}
        if tag == "base":
            base_trades, base_daily = res["trades"], res["daily"]
    out["threshold"] = thr

    # regime slices of the base run
    if len(base_trades):
        t = base_trades.copy()
        t["day"] = t["entry_ts"].dt.normalize()
        day_atr = feat.groupby("day")["_atr_points"].mean()
        ter = pd.qcut(day_atr, 3, labels=["low_vol", "mid_vol", "high_vol"])
        t["vol_regime"] = t["day"].map(ter)
        t["session"] = pd.cut(t["entry_ts"].dt.hour,
                              bins=[0, 9, 15, 24],
                              labels=["asia", "europe", "us"], right=False)
        for col in ("vol_regime", "session"):
            out[f"slice_{col}"] = {
                str(k): {"n": int(len(g)), "sum_R": float(g["R"].sum()),
                         "exp_R": float(g["R"].mean())}
                for k, g in t.groupby(col, observed=True)}
        log.info(f"slices: {out['slice_vol_regime']} | {out['slice_session']}")

    # synthetic cross-check (veto only): label-implied net R on each universe
    synth_check = {}
    for s in synth_sources().keys():
        sf = build_features(s)
        sl_ = build_labels(s)
        m_ = sf.merge(sl_, on="ts", how="inner")
        m_ = m_[m_["eligible"]].reset_index(drop=True)
        (pl_, ps_), _ = predict_champion(m_, which)
        ca = cost_atr(m_)
        lab = experiment()["labels"]
        ev_l = np.nan_to_num(pl_) * lab["tp_atr"] - (1 - np.nan_to_num(pl_)) * lab["sl_atr"] - ca
        ev_s = np.nan_to_num(ps_) * lab["tp_atr"] - (1 - np.nan_to_num(ps_)) * lab["sl_atr"] - ca
        take_long = ev_l >= ev_s
        ev = np.maximum(ev_l, ev_s)
        act = ev > experiment()["decision"]["min_ev_atr"]
        Rn = np.where(take_long, realized_R(m_, "long"), realized_R(m_, "short")) - ca
        synth_check[s] = {"n": int(act.sum()),
                          "exp_R": float(Rn[act].mean()) if act.any() else 0.0}
        log.info(f"synth {s}: n={synth_check[s]['n']:,} exp {synth_check[s]['exp_R']:+.4f}")
    out["synth_check"] = synth_check
    evs = [v["exp_R"] for v in synth_check.values() if v["n"] > 50]
    out["synth_veto"] = bool(evs and (np.median(evs) < 0 and min(evs) < -0.15))

    # real-only survival (only meaningful if the recipe pooled synth)
    art = paths.MODELS_DIR / "lgbm_final_realonly"
    if art.exists():
        from .validate import predict_gbt_from

        rl, rs = predict_gbt_from(art, feat)
        res = run_backtest(df1m, _signals(feat, rl, rs), make_cost_cfg(), risk)
        m = compute_metrics(res["trades"], res["daily"], eq0)
        out["real_only_survival"] = {"pf": m.get("profit_factor"),
                                     "exp_R": m.get("expectancy_R"),
                                     "n": m["n_trades"], "ret": m["total_return_pct"]}
        log.info(f"real-only survival: {out['real_only_survival']}")
    save_json(out, run_dir / "battery.json")

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    cms = [c["cost_mult"] for c in curve]
    pfs = [c["pf"] if c["pf"] is not None else 0 for c in curve]
    ax.plot(cms, pfs, color=SERIES[0], marker="o")
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("cost multiplier")
    ax.set_ylabel("profit factor")
    ax.set_title("PF vs cost stress (needs graceful decay, no cliff)")
    fig.savefig(run_dir / "plots" / "cost_curve.png")
    plt.close(fig)

    append_ledger("robustness battery (single look)",
                  f"cost curve PF: {[(c['cost_mult'], None if c['pf'] is None else round(c['pf'], 3)) for c in curve]}; "
                  f"1.5×+2×slip: {out['stress_1p5x_2xslip']}; thresholds: {thr}; "
                  f"synth veto: {out['synth_veto']} → {run_dir.name}")
    log.info(f"BATTERY DONE → {run_dir}")
