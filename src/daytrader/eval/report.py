"""Backtest reporting: metrics, equity/drawdown/R plots, B&H comparison."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.hashio import save_json
from ..utils.log import get_logger
from .metrics import bootstrap_lb, buy_and_hold, compute_metrics, probabilistic_sharpe
from .plotstyle import CRITICAL, GOOD, MUTED, SERIES, apply_style

log = get_logger("eval.report")

KEY_ORDER = ["n_trades", "profit_factor", "win_rate", "expectancy_R", "sharpe",
             "sortino", "max_dd_pct", "total_return_pct", "cagr_pct", "calmar",
             "trades_per_day", "months_positive", "months_total",
             "max_day_share", "max_month_share", "pnl_without_top5",
             "exp_R_boot_lb95", "psr"]


def render_backtest(run_dir, name: str, res: dict, equity0: float,
                    df1m: pd.DataFrame | None = None) -> dict:
    apply_style()
    trades, daily = res["trades"], res["daily"]
    m = compute_metrics(trades, daily, equity0)
    if len(trades):
        m["exp_R_boot_lb95"] = bootstrap_lb(trades["R"].to_numpy(), q=0.05)
        m["psr"] = probabilistic_sharpe(daily["ret"])
    bh_daily = bh_m = None
    if df1m is not None:
        bh_daily, bh_m = buy_and_hold(df1m, equity0)
        m["bh_total_return_pct"] = bh_m["total_return_pct"]
        m["bh_max_dd_pct"] = bh_m["max_dd_pct"]
        m["bh_sharpe"] = bh_m["sharpe"]
        m["bh_calmar"] = bh_m["calmar"]

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    ax = axs[0, 0]
    ax.plot(daily.index, daily["equity"], color=SERIES[0], linewidth=1.8)
    if bh_daily is not None:
        ax.plot(bh_daily.index, bh_daily["equity"], color=MUTED, linewidth=1.2)
        ax.annotate("buy & hold", xy=(bh_daily.index[-1], bh_daily["equity"].iloc[-1]),
                    color=MUTED, fontsize=8, xytext=(-60, 8), textcoords="offset points")
        ax.annotate(name, xy=(daily.index[-1], daily["equity"].iloc[-1]),
                    color=SERIES[0], fontsize=8, xytext=(-40, -14), textcoords="offset points")
    ax.set_title("equity")
    ax.tick_params(axis="x", labelrotation=30)

    ax = axs[0, 1]
    dd = (1 - daily["equity"] / daily["equity"].cummax()) * 100
    ax.fill_between(daily.index, -dd, 0, color=CRITICAL, alpha=0.35, linewidth=0)
    ax.set_title("drawdown %")
    ax.tick_params(axis="x", labelrotation=30)

    ax = axs[1, 0]
    if len(trades):
        ax.hist(trades["R"], bins=60, color=SERIES[0])
        ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("trade R distribution")

    ax = axs[1, 1]
    if len(trades):
        monthly = trades.groupby(trades["exit_ts"].dt.to_period("M"))["pnl"].sum()
        colors = [GOOD if v > 0 else CRITICAL for v in monthly]
        ax.bar(range(len(monthly)), monthly.values, color=colors, width=0.62)
        ax.set_xticks(range(len(monthly)),
                      [str(p) for p in monthly.index], rotation=45, fontsize=7)
    ax.set_title("monthly P&L ($)")
    fig.suptitle(f"{name}")
    fig.savefig(run_dir / "plots" / f"bt_{name}.png")
    plt.close(fig)

    save_json(m, run_dir / f"metrics_{name}.json")
    if len(trades):
        trades.to_csv(run_dir / f"trades_{name}.csv", index=False)
    pf = m.get("profit_factor", 0)
    log.info(
        f"{name}: {m['n_trades']} trades | PF {pf if pf == np.inf else round(pf, 3)} "
        f"| exp {m.get('expectancy_R', 0):+.3f}R (LB95 {m.get('exp_R_boot_lb95', float('nan')):+.3f}) "
        f"| Sharpe {m['sharpe']:.2f} | maxDD {m['max_dd_pct']:.1f}% "
        f"| ret {m['total_return_pct']:+.1f}%"
        + (f" | B&H ret {m['bh_total_return_pct']:+.1f}% DD {m['bh_max_dd_pct']:.1f}%"
           if df1m is not None else "")
    )
    return m
