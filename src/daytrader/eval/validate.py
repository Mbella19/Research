"""Validation gate — pre-registered looks ONLY (ledger D-002/D-003).

Every invocation runs the REAL backtester on real VALIDATION data, evaluates
the pre-registered criteria, and appends an entry to notes/decisions.md.
The number of prior looks feeds the Deflated Sharpe trial count.
"""
import pickle
import re
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
from ..backtest.signals import from_probabilities
from ..config import experiment, instrument
from ..data.loader import load_bars
from ..features.registry import build_features
from ..models.dataset import cost_atr
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger
from .metrics import (bootstrap_lb, buy_and_hold, compute_metrics,
                      deflated_sharpe, probabilistic_sharpe)
from .report import render_backtest

log = get_logger("eval.validate")

LEDGER = paths.NOTES_DIR / "decisions.md"


def _count_looks() -> int:
    if not LEDGER.exists():
        return 0
    return len(re.findall(r"^## VLOOK-", LEDGER.read_text(), flags=re.M))


def append_ledger(title: str, body: str) -> int:
    n = _count_looks() + 1
    entry = f"\n## VLOOK-{n:02d} · {date.today()} · {title}\n{body}\n"
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(entry)
    log.info(f"ledger: VLOOK-{n:02d} appended")
    return n


# ── model inference on an arbitrary source ───────────────────────────
def predict_gbt_from(art, feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Gate-path probabilities: raw booster output, quantile-mapped onto the
    OOF distribution the gate economics were validated on (isotonic exists in
    the artifact for reporting only)."""
    meta = load_json(art / "meta.json")
    X = feat[meta["feature_names"]].to_numpy(np.float32)
    out = []
    for side in ("long", "short"):
        booster = lgb.Booster(model_file=str(art / f"{side}.txt"))
        raw = booster.predict(X)
        qp = art / f"qmap_{side}.npz"
        if qp.exists():
            q = np.load(qp)
            raw = np.interp(raw, q["knots_final"], q["knots_oof"])
        out.append(raw.astype(np.float32))
    return out[0], out[1]


def predict_gbt(feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    art = experiment()["decision"].get("s1_artifact", "lgbm_final")
    return predict_gbt_from(paths.MODELS_DIR / art, feat)


def predict_tcn(feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from ..models.tcn import CHANNELS, device
    from ..models.tcn_pipeline import load_tcn_ensemble

    models, meta = load_tcn_ensemble()
    X = feat[meta["feature_names"]].to_numpy(np.float32)
    chan_cols = [meta["feature_names"].index(c) for c in CHANNELS]
    chan = np.ascontiguousarray(X[:, chan_cols])
    L = meta["tcn_cfg"]["seq_len"]
    n = len(X)
    dev = device()
    p_l = np.full(n, np.nan, dtype=np.float32)
    p_s = np.full(n, np.nan, dtype=np.float32)
    seqs = np.lib.stride_tricks.sliding_window_view(chan, (L, chan.shape[1]))[:, 0]
    valid = np.arange(L - 1, n)
    with torch.no_grad():
        for lo in range(0, len(valid), 4096):
            rows = valid[lo : lo + 4096]
            seq = torch.from_numpy(np.ascontiguousarray(
                seqs[rows - L + 1].transpose(0, 2, 1))).to(dev)
            tab = torch.from_numpy(X[rows]).to(dev)
            acc = None
            for m in models:
                m.to(dev)
                p = torch.sigmoid(m(seq, tab)).cpu().numpy()
                acc = p if acc is None else acc + p
            acc /= len(models)
            p_l[rows], p_s[rows] = acc[:, 0], acc[:, 1]
    return p_l, p_s


def predict_champion(feat: pd.DataFrame, which: str | None = None):
    champ = which or load_json(paths.MODELS_DIR / "champion.json")["champion"]
    if champ == "gbt":
        return predict_gbt(feat), "gbt"
    if champ == "tcn":
        return predict_tcn(feat), "tcn"
    gl, gs = predict_gbt(feat)
    tl, ts_ = predict_tcn(feat)
    coefs = load_json(paths.MODELS_DIR / "stacker.json")

    def stack(pg, pt, side):
        lg_ = np.log(np.clip(pg, 1e-5, 1 - 1e-5) / (1 - np.clip(pg, 1e-5, 1 - 1e-5)))
        lt = np.log(np.clip(pt, 1e-5, 1 - 1e-5) / (1 - np.clip(pt, 1e-5, 1 - 1e-5)))
        z = coefs[side]["coef"][0] * lg_ + coefs[side]["coef"][1] * lt + coefs[side]["intercept"]
        return 1 / (1 + np.exp(-z))

    return (stack(gl, tl, "long"), stack(gs, ts_, "short")), "blend"


# ── criteria ─────────────────────────────────────────────────────────
def evaluate_criteria(m: dict, trades: pd.DataFrame, daily: pd.DataFrame,
                      n_trials: int) -> dict:
    checks = {}
    r = trades["R"].to_numpy() if len(trades) else np.array([])
    checks["boot_lb_expectancy_pos"] = (bootstrap_lb(r) > 0) if len(r) else False
    psr = probabilistic_sharpe(daily["ret"])
    checks["psr_ge_0.95"] = bool(psr >= 0.95) if np.isfinite(psr) else False
    dsr = deflated_sharpe(daily["ret"], n_trials=max(n_trials, 1))
    checks["dsr_ge_0.95"] = bool(dsr >= 0.95) if np.isfinite(dsr) else False
    checks["pf_ge_1.2"] = m.get("profit_factor", 0) >= 1.2
    checks["sharpe_ge_1"] = m.get("sharpe", 0) >= 1.0
    checks["dd_le_15"] = m.get("max_dd_pct", 100) <= 15.0
    checks["trades_ge_200"] = m.get("n_trades", 0) >= 200
    mp, mt = m.get("months_positive", 0), m.get("months_total", 1)
    checks["months_pos_ge_5of7"] = mp / max(mt, 1) >= 5 / 7 - 1e-9
    checks["max_month_share_le_40"] = m.get("max_month_share", 1.0) <= 0.40
    checks["max_day_share_le_15"] = m.get("max_day_share", 1.0) <= 0.15
    checks["profitable_wo_top5"] = m.get("pnl_without_top5", -1) > 0
    if "bh_total_return_pct" in m:
        checks["ret_ge_bh"] = m["total_return_pct"] >= m["bh_total_return_pct"]
        checks["dd_le_half_bh"] = m["max_dd_pct"] <= min(15.0, 0.5 * m["bh_max_dd_pct"])
        checks["sharpe_gt_bh"] = m["sharpe"] > m["bh_sharpe"]
        checks["calmar_gt_bh"] = m["calmar"] > m["bh_calmar"]
    checks["_psr"] = round(float(psr), 4) if np.isfinite(psr) else None
    checks["_dsr"] = round(float(dsr), 4) if np.isfinite(dsr) else None
    checks["_n_trials"] = n_trials
    return checks


def _signals_for(source: str, which: str | None, min_ev: float | None = None):
    feat = build_features(source)
    (p_l, p_s), name = predict_champion(feat, which)
    dec = experiment()["decision"]
    lab = experiment()["labels"]
    ca = cost_atr(feat, profile=dec.get("gate_cost_profile"))
    from ..models.dataset import drift_atr

    sig = from_probabilities(feat, np.nan_to_num(p_l), np.nan_to_num(p_s),
                             lab["tp_atr"], lab["sl_atr"], ca,
                             dec["min_ev_atr"] if min_ev is None else min_ev,
                             dec["prob_floor"],
                             allowed_sides=dec.get("allowed_sides", "both"),
                             drift_atr=drift_atr(feat))
    return feat, sig, name


def run_champion_look(which: str | None = None) -> None:
    run_dir = new_run_dir("validate_champion")
    log.info(f"artifacts → {run_dir}")
    feat, sig, name = _signals_for("real_validation", which)
    df1m = load_bars("real_validation")
    res = run_backtest(df1m, sig, make_cost_cfg(), make_risk_cfg())
    m = render_backtest(run_dir, f"champion_{name}", res,
                        experiment()["backtest"]["equity0"], df1m)
    checks = evaluate_criteria(m, res["trades"], res["daily"], _count_looks() + 1)
    save_json(checks, run_dir / "criteria.json")
    n_pass = sum(1 for k, v in checks.items() if not k.startswith("_") and v)
    n_all = sum(1 for k in checks if not k.startswith("_"))
    body = (f"Champion `{name}` on VALIDATION, default gate. "
            f"{m['n_trades']} trades, PF {m.get('profit_factor', 0):.3f}, "
            f"Sharpe {m['sharpe']:.2f}, maxDD {m['max_dd_pct']:.1f}%, "
            f"ret {m['total_return_pct']:+.1f}% (B&H {m.get('bh_total_return_pct', 0):+.1f}%). "
            f"Criteria: {n_pass}/{n_all} PASS → {run_dir.name}")
    append_ledger(f"champion confirmation ({name})", body)
    log.info(f"criteria: {n_pass}/{n_all} PASS")


def run_sizing_look(which: str | None = None, min_ev: float | None = None,
                    grid=(0.005, 0.0075, 0.01, 0.015, 0.02)) -> None:
    """ONE pre-registered look: risk-per-trade grid vs the B&H-dominance goal
    (return ≥ B&H while maxDD ≤ min(15%, ½·B&H-DD))."""
    run_dir = new_run_dir("validate_sizing")
    log.info(f"artifacts → {run_dir}")
    feat, sig, name = _signals_for("real_validation", which, min_ev=min_ev)
    df1m = load_bars("real_validation")
    _, bh_m = buy_and_hold(df1m, experiment()["backtest"]["equity0"])
    dd_budget = min(15.0, 0.5 * bh_m["max_dd_pct"])
    rows = []
    for rpt in grid:
        res = run_backtest(df1m, sig, make_cost_cfg(), make_risk_cfg(rpt))
        m = compute_metrics(res["trades"], res["daily"],
                            experiment()["backtest"]["equity0"])
        rows.append({"risk": rpt, "ret": m["total_return_pct"],
                     "dd": m["max_dd_pct"], "sharpe": m["sharpe"],
                     "pf": m.get("profit_factor"),
                     "beats_bh": bool(m["total_return_pct"] >= bh_m["total_return_pct"]
                                      and m["max_dd_pct"] <= dd_budget)})
        log.info(f"risk {rpt}: ret {m['total_return_pct']:+.1f}% dd {m['max_dd_pct']:.1f}% "
                 f"sharpe {m['sharpe']:.2f} beats_bh_in_budget={rows[-1]['beats_bh']}")
    save_json({"bh_ret": bh_m["total_return_pct"], "bh_dd": bh_m["max_dd_pct"],
               "dd_budget": dd_budget, "grid": rows}, run_dir / "sizing.json")
    append_ledger("sizing/risk-knob look (single look)",
                  f"B&H {bh_m['total_return_pct']:+.1f}% DD {bh_m['max_dd_pct']:.1f}% "
                  f"(budget {dd_budget:.1f}%); " + "; ".join(
                      f"r={r['risk']}: {r['ret']:+.1f}%/{r['dd']:.1f}%dd"
                      f"{' ✓' if r['beats_bh'] else ''}" for r in rows) +
                  f" → {run_dir.name}")


def run_threshold_sweep(which: str | None = None,
                        grid=(0.05, 0.08, 0.10, 0.15, 0.20, 0.30)) -> None:
    """ONE pre-registered look: the EV-threshold curve on validation."""
    run_dir = new_run_dir("validate_threshold")
    log.info(f"artifacts → {run_dir}")
    df1m = load_bars("real_validation")
    rows = []
    for g in grid:
        feat, sig, name = _signals_for("real_validation", which, min_ev=g)
        res = run_backtest(df1m, sig, make_cost_cfg(), make_risk_cfg())
        m = compute_metrics(res["trades"], res["daily"],
                            experiment()["backtest"]["equity0"])
        rows.append({"min_ev": g, "n": m["n_trades"],
                     "pf": m.get("profit_factor"), "exp_R": m.get("expectancy_R"),
                     "sharpe": m["sharpe"], "dd": m["max_dd_pct"],
                     "ret": m["total_return_pct"]})
        log.info(f"min_ev {g}: {rows[-1]}")
    save_json(rows, run_dir / "threshold_curve.json")
    append_ledger("EV-threshold plateau sweep (single look)",
                  "curve: " + "; ".join(
                      f"ev≥{r['min_ev']}: n={r['n']} PF={r['pf'] if r['pf'] is None else round(r['pf'], 3)} "
                      f"ret={r['ret']:+.1f}%" for r in rows) + f" → {run_dir.name}")
