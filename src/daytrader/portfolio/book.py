"""Portfolio book — v2 two-sleeve assembly (D-021/D-022).

S1 (frozen v1 intraday GBT policy, risk-normalized R stream) + S2 (gated
overnight drift, per-unit-notional stream × vol-parity budget w2) settle on
one equity base, compounded daily. The pre-registered vol-targeting layer
was REFUTED on training OOF (Sharpe 1.09→0.87, deeper DD — EWMA vol of a
sparse trade stream is noise) and is dropped: FIXED budgets ship (D-022).

Stage map: oof (training, honest S1-OOF), validate --which s2|portfolio
(VLOOK each), robustness (one VLOOK), freeze-v2, finaltest (single shot
behind runs/FINALTEST_V2_EXECUTED.flag — ledgered as a SECOND look at the
Jan–Jun 2026 window; v1 consumed the virgin shot).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
from ..backtest.signals import from_probabilities
from ..config import experiment
from ..data.loader import load_bars
from ..eval.plotstyle import MUTED, SERIES, apply_style
from ..eval.validate import _signals_for, append_ledger
from ..features.registry import build_features
from ..models.dataset import cost_atr, drift_atr
from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger
from .grid import stream_metrics
from .overnight import S2Params, daily_frame, sleeve2_run

log = get_logger("portfolio")

_REAL_CTX: pd.DataFrame | None = None


def _real_daily_context() -> pd.DataFrame:
    """Daily OHLC across all real sources, for gate warm-up across window
    boundaries (live deployments always carry history)."""
    global _REAL_CTX
    if _REAL_CTX is None:
        oos_ok = ((paths.RUNS_DIR / "OOS_EXECUTED.flag").exists()
                  or os.environ.get("DAYTRADER_UNLOCK_OOS") == "1")
        sources = ["real_training", "real_validation"] + (
            ["real_locked_oos"] if oos_ok else [])   # same policy as load_bars
        parts = []
        for s in sources:
            p = paths.PARQUET_DIR / f"{s}.parquet"
            if p.exists():
                parts.append(daily_frame(pd.read_parquet(
                    p, columns=["ts", "open", "high", "low", "close"])))
        _REAL_CTX = pd.concat(parts).sort_index()
        _REAL_CTX = _REAL_CTX[~_REAL_CTX.index.duplicated()]
    return _REAL_CTX

FROZEN_V2 = paths.MODELS_DIR / "FINAL_FROZEN_V2.json"
FLAG_V2 = paths.RUNS_DIR / "FINALTEST_V2_EXECUTED.flag"
FROZEN_V3 = paths.MODELS_DIR / "FINAL_FROZEN_V3.json"


def _cfg() -> dict:
    return experiment()["sleeves"]


def _s2_params(cfg: dict | None = None) -> S2Params:
    c = (cfg or _cfg())["s2"]
    return S2Params(window=c["window"], gate=c["gate"],
                    volcap=bool(c["volcap"]), stop_atr=c["stop_atr"],
                    derisk=bool(c.get("derisk", False)))


def _s1_trades(source: str, stress_mult: float = 1.0, slip_mult: float = 1.0,
               commission_mult: float = 1.0) -> pd.DataFrame:
    """S1 trades from the frozen policy. On TRAINING the probabilities are
    the saved OUT-OF-FOLD ones (honest); elsewhere the frozen final model."""
    cost, risk = make_cost_cfg(stress_mult, slip_mult, commission_mult), make_risk_cfg()
    art = experiment()["decision"].get("s1_artifact", "lgbm_final")
    if source == "real_training":
        feat = build_features(source)
        oof = pd.read_parquet(paths.MODELS_DIR / art / "oof_real.parquet")
        m = feat.merge(oof[["ts", "p_long", "p_short"]], on="ts", how="inner")
        dec, lab = experiment()["decision"], experiment()["labels"]
        ca = cost_atr(m, profile=dec.get("gate_cost_profile"))
        sig = from_probabilities(m, np.nan_to_num(m["p_long"].to_numpy()),
                                 np.nan_to_num(m["p_short"].to_numpy()),
                                 lab["tp_atr"], lab["sl_atr"], ca,
                                 dec["min_ev_atr"], dec["prob_floor"],
                                 allowed_sides=dec.get("allowed_sides", "both"),
                                 drift_atr=drift_atr(m))
    else:
        _, sig, _ = _signals_for(source, None)
    return run_backtest(load_bars(source), sig, cost, risk)["trades"]


def build_streams(source: str, stress_mult: float = 1.0, slip_mult: float = 1.0,
                  swap_bp_night: float | None = None,
                  s2_params: S2Params | None = None,
                  commission_mult: float = 1.0) -> dict:
    cfg = _cfg()
    df1m = load_bars(source)
    ctx = _real_daily_context() if source.startswith("real") else None
    s2 = sleeve2_run(df1m, s2_params or _s2_params(cfg),
                     make_cost_cfg(stress_mult, slip_mult, commission_mult),
                     swap_bp_night=(cfg["swap_bp_night"] if swap_bp_night is None
                                    else swap_bp_night),
                     context_daily=ctx)
    days = s2["daily_ret"].index
    t1 = _s1_trades(source, stress_mult, slip_mult, commission_mult)
    r1 = (float(cfg["risk1"])
          * t1.groupby(t1["exit_ts"].dt.normalize())["R"].sum()).reindex(days).fillna(0.0)
    r2 = float(cfg["w2"]) * s2["daily_ret"]
    port = (r1 + r2).rename("port")
    close_d = df1m.groupby(df1m["ts"].dt.normalize())["close"].last()
    bh = close_d.pct_change().fillna(0.0).rename("bh")
    return {"r1": r1, "r2": r2, "port": port, "bh": bh,
            "t1": t1, "s2_trades": s2["trades"], "s2_exposure": s2["exposure"]}


def _bar_checks(m: dict, is_oof: bool) -> dict:
    """Pre-registered v2 bar (D-021)."""
    return {
        "sharpe": (m["sharpe"] >= (1.3 if is_oof else 1.0), round(m["sharpe"], 2)),
        "max_dd<=12": (m["max_dd_pct"] >= -12.0, round(m["max_dd_pct"], 1)),
        "worst_year>=-5": (m["worst_year_pct"] >= -5.0, round(m["worst_year_pct"], 1)),
        "years_pos>=5of6" if is_oof else "years_pos": (
            (m["years_pos"] >= 5) if is_oof else (m["years_pos"] >= m["years_n"]),
            f"{m['years_pos']}/{m['years_n']}"),
        "months_pos>=65": (m["months_pos_pct"] >= 65.0, round(m["months_pos_pct"], 0)),
        "worst_month>-6": (m["worst_month_pct"] > -6.0, round(m["worst_month_pct"], 1)),
    }


def _report(run_dir, tag: str, S: dict, is_oof: bool) -> dict:
    m_p = stream_metrics(S["port"])
    m_1 = stream_metrics(S["r1"])
    m_2 = stream_metrics(S["r2"])
    m_bh = stream_metrics(S["bh"])
    eq_p, eq_bh = (1 + S["port"]).cumprod(), (1 + S["bh"]).cumprod()
    corr = float(S["r1"].corr(S["r2"]))
    checks = _bar_checks(m_p, is_oof)
    checks["beats_bh_return"] = (float(eq_p.iloc[-1]) >= float(eq_bh.iloc[-1]),
                                 f"{(eq_p.iloc[-1]-1)*100:+.1f}% vs {(eq_bh.iloc[-1]-1)*100:+.1f}%")
    checks["dd<=half_bh"] = (m_p["max_dd_pct"] >= m_bh["max_dd_pct"] / 2,
                             f"{m_p['max_dd_pct']:.1f}% vs B&H {m_bh['max_dd_pct']:.1f}%")
    out = {"tag": tag, "portfolio": m_p, "s1": m_1, "s2": m_2, "bh": m_bh,
           "sleeve_corr": round(corr, 3),
           "total_return_pct": round(float((eq_p.iloc[-1] - 1) * 100), 2),
           "bh_total_return_pct": round(float((eq_bh.iloc[-1] - 1) * 100), 2),
           "s1_trades": int(len(S["t1"])), "s2_trades": int(len(S["s2_trades"])),
           "s2_exposure_pct": round(S["s2_exposure"] * 100, 1),
           "criteria": {k: {"pass": bool(v[0]), "value": v[1]} for k, v in checks.items()},
           "criteria_passed": f"{sum(int(v[0]) for v in checks.values())}/{len(checks)}"}
    save_json(out, run_dir / f"portfolio_{tag}.json")

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.5]})
    ax = axes[0]
    ax.plot(eq_p.index, eq_p, color=SERIES[0], linewidth=2, label="portfolio")
    ax.plot(eq_bh.index, eq_bh, color=MUTED, linewidth=1.2, linestyle="--", label="buy & hold")
    ax.plot(eq_p.index, (1 + S["r1"]).cumprod(), color=SERIES[1], linewidth=1, label="S1 intraday")
    ax.plot(eq_p.index, (1 + S["r2"]).cumprod(), color=SERIES[2], linewidth=1, label="S2 overnight")
    ax.set_title(f"{tag}: portfolio vs B&H (Sharpe {m_p['sharpe']:.2f}, "
                 f"DD {m_p['max_dd_pct']:.1f}% vs B&H {m_bh['max_dd_pct']:.1f}%)")
    ax.legend(fontsize=8)
    dd_p, dd_bh = eq_p / eq_p.cummax() - 1, eq_bh / eq_bh.cummax() - 1
    axes[1].fill_between(dd_p.index, dd_p * 100, 0, color=SERIES[0], alpha=0.6, linewidth=0)
    axes[1].plot(dd_bh.index, dd_bh * 100, color=MUTED, linewidth=1, linestyle="--")
    axes[1].set_ylabel("drawdown %")
    y_p = pd.Series(m_p["yearly"]); y_bh = pd.Series(m_bh["yearly"])
    x = np.arange(len(y_p))
    axes[2].bar(x - 0.2, y_p.values, width=0.4, color=SERIES[0], label="portfolio")
    axes[2].bar(x + 0.2, y_bh.reindex(y_p.index).values, width=0.4, color=MUTED, label="B&H")
    axes[2].set_xticks(x, [str(k) for k in y_p.index])
    axes[2].axhline(0, color=MUTED, linewidth=0.8)
    axes[2].set_ylabel("year %"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / f"bt_{tag}.png")
    plt.close(fig)

    monthly = (1 + S["port"]).groupby([S["port"].index.year, S["port"].index.month]).prod() - 1
    monthly.rename("ret").to_csv(run_dir / f"monthly_{tag}.csv")
    S["t1"].to_csv(run_dir / f"trades_S1_{tag}.csv", index=False)
    S["s2_trades"].to_csv(run_dir / f"trades_S2_{tag}.csv", index=False)
    log.info(f"{tag}: port Sharpe {m_p['sharpe']:.2f} ann {m_p['ann_ret_pct']:+.2f}% "
             f"DD {m_p['max_dd_pct']:.1f}% | S1 {m_1['sharpe']:.2f} S2 {m_2['sharpe']:.2f} "
             f"corr {corr:+.2f} | B&H {out['bh_total_return_pct']:+.1f}%/{m_bh['max_dd_pct']:.1f}%DD "
             f"| criteria {out['criteria_passed']}")
    return out


def run_oof() -> None:
    run_dir = new_run_dir("portfolio_oof")
    log.info(f"artifacts → {run_dir}")
    S = build_streams("real_training")
    out = _report(run_dir, "TRAIN_OOF", S, is_oof=True)
    for k, v in out["criteria"].items():
        log.info(f"  {'PASS' if v['pass'] else 'FAIL':4s} {k}: {v['value']}")
    log.info(f"OOF DONE → {run_dir}")


def run_validate(which: str = "s2") -> None:
    run_dir = new_run_dir(f"portfolio_val_{which}")
    log.info(f"artifacts → {run_dir}")
    if which == "s2":
        S = build_streams("real_validation")
        m2 = stream_metrics(S["r2"] / max(_cfg()["w2"], 1e-9))   # unit-notional basis
        save_json(m2, run_dir / "s2_validation.json")
        t = S["s2_trades"]
        append_ledger("S2 overnight sleeve — validation confirm (v2 look 1)",
                      f"{_s2_params().tag()}: {len(t)} trades, exposure "
                      f"{S['s2_exposure']*100:.0f}%, Sharpe {m2['sharpe']:.2f}, "
                      f"ann {m2['ann_ret_pct']:+.2f}% (unit notional), maxDD "
                      f"{m2['max_dd_pct']:.1f}%, worst month {m2['worst_month_pct']:+.1f}%, "
                      f"months+ {m2['months_pos_pct']:.0f}% → {run_dir.name}")
        log.info(f"S2 validation: Sharpe {m2['sharpe']:.2f} ann {m2['ann_ret_pct']:+.2f}% "
                 f"DD {m2['max_dd_pct']:.1f}% n={len(t)}")
    else:
        S = build_streams("real_validation")
        out = _report(run_dir, "VALIDATION", S, is_oof=False)
        append_ledger("v2 portfolio — validation confirm (v2 look 2)",
                      f"Sharpe {out['portfolio']['sharpe']:.2f}, ret "
                      f"{out['total_return_pct']:+.2f}% (B&H {out['bh_total_return_pct']:+.2f}%), "
                      f"DD {out['portfolio']['max_dd_pct']:.1f}% (B&H "
                      f"{out['bh']['max_dd_pct']:.1f}%), corr {out['sleeve_corr']:+.2f}, "
                      f"criteria {out['criteria_passed']} → {run_dir.name}")
    log.info(f"VALIDATE DONE → {run_dir}")


def run_robustness() -> None:
    run_dir = new_run_dir("portfolio_robustness")
    log.info(f"artifacts → {run_dir}")
    out: dict = {}
    curve = []
    for cm in (1.0, 1.25, 1.5, 1.75, 2.0):
        S = build_streams("real_validation", stress_mult=cm)
        m = stream_metrics(S["port"])
        curve.append({"cost_mult": cm, "sharpe": round(m["sharpe"], 2),
                      "ann_ret_pct": round(m["ann_ret_pct"], 2)})
        log.info(f"cost×{cm}: Sharpe {m['sharpe']:.2f} ann {m['ann_ret_pct']:+.2f}%")
    out["cost_curve"] = curve
    S = build_streams("real_validation", stress_mult=1.5, slip_mult=2.0)
    out["stress_1p5x_2xslip"] = {"sharpe": round(stream_metrics(S["port"])["sharpe"], 2),
                                 "ann_ret_pct": round(stream_metrics(S["port"])["ann_ret_pct"], 2)}
    S = build_streams("real_validation", swap_bp_night=2.5)
    m_swap = stream_metrics(S["r2"] / max(_cfg()["w2"], 1e-9))
    out["s2_cfd_swap_2p5bp"] = {"sharpe": round(m_swap["sharpe"], 2),
                                "ann_ret_pct": round(m_swap["ann_ret_pct"], 2)}
    gates = {}
    df1m = load_bars("real_validation")
    ctx = _real_daily_context()
    val_days = daily_frame(df1m).index
    for n in (40, 50, 60):
        c = ctx["close"]
        on = ((c > c.rolling(n).mean()).astype(bool).shift(1, fill_value=False)
              .reindex(val_days, fill_value=False))
        s2 = sleeve2_run(df1m, _s2_params(), make_cost_cfg(), gate_override=on)
        m = stream_metrics(s2["daily_ret"])
        gates[f"sma{n}"] = {"sharpe": round(m["sharpe"], 2),
                            "ann_ret_pct": round(m["ann_ret_pct"], 2),
                            "n": int(len(s2["trades"]))}
        log.info(f"gate sma{n}: {gates[f'sma{n}']}")
    out["gate_perturbation"] = gates
    synth = {}
    from ..config import synth_sources
    for s in synth_sources().keys():
        df_s = load_bars(s)
        s2 = sleeve2_run(df_s, _s2_params(), make_cost_cfg())
        m = stream_metrics(s2["daily_ret"])
        synth[s] = {"sharpe": round(m["sharpe"], 2), "ann_ret_pct": round(m["ann_ret_pct"], 2)}
    out["synth_screen"] = synth
    save_json(out, run_dir / "battery.json")
    append_ledger("v2 portfolio robustness battery (v2 look 3)",
                  f"cost curve {[(c['cost_mult'], c['sharpe']) for c in curve]}; "
                  f"1.5×+2×slip {out['stress_1p5x_2xslip']}; S2 CFD-swap 2.5bp/night "
                  f"{out['s2_cfd_swap_2p5bp']}; gate ±10d {gates}; "
                  f"synth {synth} → {run_dir.name}")
    log.info(f"ROBUSTNESS DONE → {run_dir}")


def run_fullhistory() -> None:
    """Full-history rendering: the honest streams stitched end-to-end
    (2020-01 → 2026-06). Training era uses OUT-OF-FOLD S1 probabilities;
    validation + final-test use the frozen model. Post-final-test reporting
    only — refuses to run before the final test is consumed, and makes no
    decisions (both windows are already ledgered looks)."""
    if not FLAG_V2.exists():
        raise SystemExit("full-history rendering only AFTER the final test is "
                         "consumed (it re-renders spent windows, never fresh ones).")
    os.environ.setdefault("DAYTRADER_UNLOCK_OOS", "1")   # post-consumption re-read
    run_dir = new_run_dir("fullhistory")
    log.info(f"artifacts → {run_dir}")

    sources = ["real_training", "real_validation", "real_locked_oos"]
    S = {k: pd.concat([build_streams(s)[k] for s in sources])
         for k in ("r1", "r2", "port")}
    close_d = pd.concat([load_bars(s).pipe(
        lambda b: b.groupby(b["ts"].dt.normalize())["close"].last()) for s in sources])
    bh = close_d.pct_change().fillna(0.0).rename("bh")
    port, r1, r2 = S["port"], S["r1"], S["r2"]

    m_p, m_bh = stream_metrics(port), stream_metrics(bh)
    m_1, m_2 = stream_metrics(r1), stream_metrics(r2)
    eq_p, eq_bh = (1 + port).cumprod(), (1 + bh).cumprod()
    seg = {"train_oof": ("2020-01-01", "2025-05-31"),
           "validation": ("2025-06-01", "2025-12-31"),
           "final_test": ("2026-01-01", "2026-06-30")}
    per_set = {k: stream_metrics(port.loc[a:b]) for k, (a, b) in seg.items()}
    out = {"overall": m_p, "bh": m_bh, "s1": m_1, "s2": m_2,
           "per_set": per_set,
           "total_return_pct": round(float((eq_p.iloc[-1] - 1) * 100), 2),
           "bh_total_return_pct": round(float((eq_bh.iloc[-1] - 1) * 100), 2),
           "sleeve_corr": round(float(r1.corr(r2)), 3)}
    save_json(out, run_dir / "fullhistory.json")
    log.info(f"FULL 6.5y: port {out['total_return_pct']:+.1f}% "
             f"(Sharpe {m_p['sharpe']:.2f}, DD {m_p['max_dd_pct']:.1f}%) vs "
             f"B&H {out['bh_total_return_pct']:+.1f}% (DD {m_bh['max_dd_pct']:.1f}%)")

    apply_style()
    bounds = [pd.Timestamp(seg["validation"][0]), pd.Timestamp(seg["final_test"][0])]

    # ── FIG 1: compounded equity + drawdown + rolling Sharpe ─────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 11),
                             gridspec_kw={"height_ratios": [3, 1.3, 1.1]})
    ax = axes[0]
    ax.plot(eq_p.index, eq_p, color=SERIES[0], linewidth=2, label="portfolio")
    ax.plot(eq_bh.index, eq_bh, color=MUTED, linewidth=1.2, linestyle="--", label="buy & hold")
    ax.plot(eq_p.index, (1 + r1).cumprod(), color=SERIES[1], linewidth=1, label="S1 intraday")
    ax.plot(eq_p.index, (1 + r2).cumprod(), color=SERIES[2], linewidth=1, label="S2 overnight")
    for b_ in bounds:
        for a_ in axes[:2]:
            a_.axvline(b_, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(pd.Timestamp("2022-06-01"), ax.get_ylim()[1] * 0.02 + 2.4,
            "training era (honest out-of-fold)", fontsize=8, color=MUTED)
    ax.text(bounds[0], 2.55, " validation", fontsize=8, color=MUTED)
    ax.text(bounds[1], 2.4, " final test", fontsize=8, color=MUTED)
    ax.set_title(f"Full real history 2020-01 → 2026-06 — portfolio {out['total_return_pct']:+.0f}% "
                 f"(DD {m_p['max_dd_pct']:.1f}%)  vs  B&H {out['bh_total_return_pct']:+.0f}% "
                 f"(DD {m_bh['max_dd_pct']:.1f}%)")
    ax.legend(fontsize=8, loc="upper left")
    dd_p, dd_bh = eq_p / eq_p.cummax() - 1, eq_bh / eq_bh.cummax() - 1
    axes[1].fill_between(dd_p.index, dd_p * 100, 0, color=SERIES[0], alpha=0.6, linewidth=0)
    axes[1].plot(dd_bh.index, dd_bh * 100, color=MUTED, linewidth=1, linestyle="--")
    axes[1].set_ylabel("drawdown %")
    roll = port.rolling(126).mean() / port.rolling(126).std() * np.sqrt(252)
    axes[2].plot(roll.index, roll, color=SERIES[0], linewidth=1.2)
    axes[2].axhline(0, color=MUTED, linewidth=0.8)
    axes[2].set_ylabel("rolling 6-mo Sharpe")
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "fullhistory_equity.png")
    plt.close(fig)

    # ── FIG 2: monthly NON-COMPOUNDING heatmap (simple sums of daily) ────
    mo = port.groupby([port.index.year, port.index.month]).sum() * 100
    grid = mo.unstack(level=1).reindex(columns=range(1, 13))
    grid["Year"] = grid.sum(axis=1, skipna=True)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    vals = grid.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals))
    im = ax.imshow(vals, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                              "Aug", "Sep", "Oct", "Nov", "Dec", "YEAR"])
    ax.set_yticks(range(len(grid)), [str(y) for y in grid.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:+.1f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(vals[i, j]) > vmax * 0.55 else "#222222")
    ax.axvline(11.5, color="#222222", linewidth=1.2)
    ax.set_title("portfolio monthly returns %, NON-compounding (sum of daily returns "
                 "on a fixed base — months are additive)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "fullhistory_monthly_heatmap.png")
    plt.close(fig)
    grid.round(2).to_csv(run_dir / "monthly_noncompound.csv")

    # ── FIG 3: non-compounded cumulative + yearly bars vs B&H ────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5),
                             gridspec_kw={"height_ratios": [2, 1.4]})
    axes[0].plot(port.index, port.cumsum() * 100, color=SERIES[0], linewidth=2,
                 label="portfolio (non-compounded)")
    axes[0].plot(bh.index, bh.cumsum() * 100, color=MUTED, linewidth=1.2,
                 linestyle="--", label="B&H (non-compounded)")
    for b_ in bounds:
        axes[0].axvline(b_, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("cumulative % (additive)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("non-compounding view: sum of daily returns (fixed risk base)")
    y_p = port.groupby(port.index.year).sum() * 100
    y_bh = bh.groupby(bh.index.year).sum() * 100
    x = np.arange(len(y_p))
    axes[1].bar(x - 0.2, y_p.values, width=0.4, color=SERIES[0], label="portfolio")
    axes[1].bar(x + 0.2, y_bh.values, width=0.4, color=MUTED, label="B&H")
    for xi, v in zip(x, y_p.values):
        axes[1].text(xi - 0.2, v + (1.2 if v >= 0 else -3.2), f"{v:+.1f}",
                     ha="center", fontsize=7.5)
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].set_xticks(x, [str(k) for k in y_p.index])
    axes[1].set_ylabel("year % (non-compounded)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "fullhistory_noncompound.png")
    plt.close(fig)
    log.info(f"FULL-HISTORY RENDER DONE → {run_dir}")


def run_freeze_v2() -> None:
    if FROZEN_V2.exists():
        raise SystemExit("FINAL_FROZEN_V2.json exists — v2 already frozen. Refusing.")
    from ..utils.hashio import sha256_file

    cfg = _cfg()
    v1 = load_json(paths.MODELS_DIR / "FINAL_FROZEN.json")
    art = paths.MODELS_DIR / "lgbm_final"
    frozen = {
        "version": 2,
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sleeves": cfg,
        "s1": {"frozen_v1": v1, "artifacts": {f.name: sha256_file(f)
                                              for f in sorted(art.glob("*"))
                                              if f.is_file()}},
        "s2_source_sha": sha256_file(paths.PROJECT_ROOT / "src" / "daytrader"
                                     / "portfolio" / "overnight.py"),
        "grid_run": "sleeve2_grid_20260702_160652",
    }
    save_json(frozen, FROZEN_V2)
    log.info(f"FROZEN v2 → {FROZEN_V2}")


def run_freeze_v3() -> None:
    if FROZEN_V3.exists():
        raise SystemExit("FINAL_FROZEN_V3.json exists — v3 already frozen. Refusing.")
    from ..utils.hashio import sha256_file

    ex = experiment()
    art = paths.MODELS_DIR / ex["decision"]["s1_artifact"]
    frozen = {
        "version": 3,
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sleeves": ex["sleeves"],
        "decision": ex["decision"],
        "labels": ex["labels"],
        "s1_artifacts": {f.name: sha256_file(f) for f in sorted(art.glob("*"))
                         if f.is_file()},
        "s2_source_sha": sha256_file(paths.PROJECT_ROOT / "src" / "daytrader"
                                     / "portfolio" / "overnight.py"),
        "lineage": {"arena": "arena_v3_20260702_220058",
                    "search": "lgbm_search_20260702_213920",
                    "gate_decision": "D-028"},
    }
    save_json(frozen, FROZEN_V3)
    log.info(f"FROZEN v3 → {FROZEN_V3}")


def run_finaltest(confirm: bool = False) -> None:
    if FLAG_V2.exists():
        raise SystemExit("FINALTEST_V2_EXECUTED.flag exists — the final test was "
                         "consumed. There is no second run, by design.")
    if not FROZEN_V2.exists():
        raise SystemExit("freeze v2 first (portfolio --stage freeze-v2).")
    if not confirm:
        raise SystemExit("pass --confirm-single-shot to consume the v2 final test.")
    if os.environ.get("DAYTRADER_UNLOCK_OOS") != "1":
        raise SystemExit("set DAYTRADER_UNLOCK_OOS=1 to confirm intent.")
    run_dir = new_run_dir("FINALTEST_V2")
    log.info(f"artifacts → {run_dir}")
    S = build_streams("real_locked_oos")
    out = _report(run_dir, "FINALTEST", S, is_oof=False)
    FLAG_V2.write_text(f"consumed {pd.Timestamp.now().isoformat()} → {run_dir.name}\n")
    append_ledger("v2 FINAL TEST — Jan–Jun 2026 (SECOND look at this window; "
                  "v1 consumed the virgin shot)",
                  f"portfolio Sharpe {out['portfolio']['sharpe']:.2f}, ret "
                  f"{out['total_return_pct']:+.2f}% (B&H {out['bh_total_return_pct']:+.2f}%), "
                  f"DD {out['portfolio']['max_dd_pct']:.1f}% (B&H {out['bh']['max_dd_pct']:.1f}%), "
                  f"S1 {out['s1']['sharpe']:.2f} / S2 {out['s2']['sharpe']:.2f}, corr "
                  f"{out['sleeve_corr']:+.2f}, criteria {out['criteria_passed']} → {run_dir.name}")
    log.info(f"FINAL TEST DONE (flag written) → {run_dir}")
