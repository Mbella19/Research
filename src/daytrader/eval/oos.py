"""LOCKED OOS single-shot runner + forward-test harness.

The OOS gate physically refuses to run twice (runs/OOS_EXECUTED.flag) and
requires --confirm-single-shot. It evaluates the FROZEN champion artifact,
untouched, on the locked file — the first cold forward test.

`run_forward` applies the same frozen artifact to ANY new MT5 export
(demo/paper trading exports) and reports divergence vs the validation trade
distribution — the tooling for the 2–3 month paper-trade protocol.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
from ..config import experiment
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json, sha256_file
from ..utils.log import get_logger
from .metrics import bootstrap_lb
from .report import render_backtest
from .validate import _signals_for, append_ledger, evaluate_criteria, _count_looks

log = get_logger("eval.oos")

FLAG = paths.RUNS_DIR / "OOS_EXECUTED.flag"


def run_oos(confirm: bool) -> None:
    if FLAG.exists():
        raise SystemExit(f"REFUSED: locked OOS was already executed once "
                         f"({FLAG.read_text().strip()}). There is no second shot.")
    if not confirm:
        raise SystemExit("REFUSED: pass --confirm-single-shot. This runs the "
                         "locked OOS exactly once, forever.")
    frozen = paths.MODELS_DIR / "FINAL_FROZEN.json"
    if not frozen.exists():
        raise SystemExit("REFUSED: no frozen artifact (models/FINAL_FROZEN.json). "
                         "Freeze the validated recipe first (daytrader freeze).")

    os.environ["DAYTRADER_UNLOCK_OOS"] = "1"
    run_dir = new_run_dir("OOS_SINGLE_SHOT")
    log.info(f"artifacts → {run_dir}")
    FLAG.write_text(f"executed {datetime.now().isoformat()} run={run_dir.name}\n")

    from ..data.loader import ingest_source, load_bars
    from ..config import real_sources

    src = real_sources(include_locked_oos=True)["real_locked_oos"]
    ingest_source("real_locked_oos", src)
    meta_f = load_json(frozen)
    feat, sig, name = _signals_for("real_locked_oos", meta_f["champion"])
    df1m = load_bars("real_locked_oos")
    res = run_backtest(df1m, sig, make_cost_cfg(), make_risk_cfg())
    m = render_backtest(run_dir, f"OOS_{name}", res,
                        experiment()["backtest"]["equity0"], df1m)
    checks = evaluate_criteria(m, res["trades"], res["daily"], _count_looks() + 1)
    save_json({"metrics": m, "criteria": checks, "frozen": meta_f},
              run_dir / "oos_result.json")
    n_pass = sum(1 for k, v in checks.items() if not k.startswith("_") and v)
    n_all = sum(1 for k in checks if not k.startswith("_"))
    append_ledger("LOCKED OOS — SINGLE SHOT",
                  f"{name}: {m['n_trades']} trades, PF {m.get('profit_factor', 0):.3f}, "
                  f"Sharpe {m['sharpe']:.2f}, DD {m['max_dd_pct']:.1f}%, "
                  f"ret {m['total_return_pct']:+.1f}% (B&H {m.get('bh_total_return_pct', 0):+.1f}%). "
                  f"Criteria {n_pass}/{n_all} → {run_dir.name}")
    log.info(f"OOS complete. Criteria {n_pass}/{n_all}. This gate is now closed forever.")


def run_forward(csv_path: str) -> None:
    """Frozen artifact on a fresh MT5 export + divergence control charts."""
    from ..data.loader import integrity_check, load_mt5_csv

    run_dir = new_run_dir("forward")
    log.info(f"artifacts → {run_dir}")
    frozen = load_json(paths.MODELS_DIR / "FINAL_FROZEN.json")
    p = paths.PROJECT_ROOT / csv_path if not str(csv_path).startswith("/") else __import__("pathlib").Path(csv_path)
    df1m = load_mt5_csv(p)
    rep = integrity_check(df1m, "forward")
    save_json({"integrity": rep, "csv_sha256": sha256_file(p)}, run_dir / "input_meta.json")

    from ..features.registry import build_features_from_1m
    from ..models.dataset import cost_atr
    from ..backtest.signals import from_probabilities
    from .validate import predict_champion

    feat = build_features_from_1m(df1m, list(experiment()["features"]["groups"]))
    (p_l, p_s), name = predict_champion(feat, frozen["champion"])
    dec, lab = experiment()["decision"], experiment()["labels"]
    from ..models.dataset import drift_atr

    sig = from_probabilities(feat, np.nan_to_num(p_l), np.nan_to_num(p_s),
                             lab["tp_atr"], lab["sl_atr"],
                             cost_atr(feat, profile=dec.get("gate_cost_profile")),
                             dec["min_ev_atr"], dec["prob_floor"],
                             allowed_sides=dec.get("allowed_sides", "both"),
                             drift_atr=drift_atr(feat))
    res = run_backtest(df1m, sig, make_cost_cfg(), make_risk_cfg())
    m = render_backtest(run_dir, f"forward_{name}", res,
                        experiment()["backtest"]["equity0"], df1m)

    # ── Sleeve-2 forward check (v2): realized overnight stream vs validation ──
    if (paths.MODELS_DIR / "FINAL_FROZEN_V2.json").exists():
        from ..portfolio.book import _real_daily_context, _s2_params
        from ..portfolio.grid import stream_metrics
        from ..portfolio.overnight import sleeve2_run

        s2_fw = sleeve2_run(df1m, _s2_params(), make_cost_cfg(),
                            context_daily=_real_daily_context())
        m2 = stream_metrics(s2_fw["daily_ret"])
        val_1m = pd.read_parquet(paths.PARQUET_DIR / "real_validation.parquet")
        s2_val = sleeve2_run(val_1m, _s2_params(), make_cost_cfg(),
                             context_daily=_real_daily_context())
        fw_r = s2_fw["trades"]["ret_net"].to_numpy()
        val_r = s2_val["trades"]["ret_net"].to_numpy()
        s2_div = {"n_fw": int(len(fw_r)),
                  "fw_mean_bp": float(fw_r.mean() * 1e4) if len(fw_r) else None,
                  "val_mean_bp": float(val_r.mean() * 1e4),
                  "fw_sharpe": m2["sharpe"],
                  "ks_p": (float(ks_2samp(val_r, fw_r).pvalue) if len(fw_r) >= 10 else None),
                  "halt_recommended": bool(len(fw_r) >= 20 and bootstrap_lb(fw_r) < -5e-4)}
        save_json(s2_div, run_dir / "divergence_s2.json")
        log.info(f"S2 forward divergence: {s2_div}")

    # divergence vs validation champion trades
    val_runs = sorted(paths.RUNS_DIR.glob("validate_champion_*/trades_champion_*.csv"))
    if val_runs and len(res["trades"]):
        val_R = pd.read_csv(val_runs[-1])["R"].to_numpy()
        fw_R = res["trades"]["R"].to_numpy()
        ks = ks_2samp(val_R, fw_R)
        div = {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
               "val_exp_R": float(val_R.mean()), "fw_exp_R": float(fw_R.mean()),
               "fw_exp_R_lb95": bootstrap_lb(fw_R),
               "halt_recommended": bool(bootstrap_lb(fw_R, q=0.95) < 0)}
        save_json(div, run_dir / "divergence.json")
        log.info(f"divergence vs validation: {div}")
    log.info(f"forward test done → {run_dir}")
