"""Sleeve-2 pre-registered 24-combo grid on TRAINING (D-021).

Selection rule (pre-registered BEFORE this module ran): among combos with
worst calendar year ≥ −5%, pick max Sharpe; ties (±0.05) break toward
parsimony (no stop → no volcap → simpler gate) then more exposed days.
Honesty artifacts: full grid table, picked-vs-median Sharpe spread, winner
year table + equity plot vs the unconditional window (no-gate baseline).
"""
import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest.engine import make_cost_cfg
from ..data.loader import load_bars
from ..eval.plotstyle import MUTED, SERIES, apply_style
from ..utils.artifacts import new_run_dir
from ..utils.hashio import save_json
from ..utils.log import get_logger
from .overnight import GATES, WINDOWS, S2Params, sleeve2_run

log = get_logger("sleeve2.grid")

TRADING_DAYS = 252


def stream_metrics(daily_ret: pd.Series) -> dict:
    """Metrics of a per-unit-notional daily return stream (0 on flat days)."""
    r = daily_ret
    eq = (1.0 + r).cumprod()
    years = max(len(r) / TRADING_DAYS, 1e-9)
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    yearly = ((1.0 + r).groupby(r.index.year).prod() - 1.0)
    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    return {
        "ann_ret_pct": float((eq.iloc[-1] ** (1 / years) - 1) * 100),
        "ann_vol_pct": float(r.std() * np.sqrt(TRADING_DAYS) * 100),
        "sharpe": sharpe,
        "max_dd_pct": float(((eq / eq.cummax()) - 1).min() * 100),
        "worst_year_pct": float(yearly.min() * 100),
        "years_pos": int((yearly > 0).sum()),
        "years_n": int(len(yearly)),
        "months_pos_pct": float((monthly > 0).mean() * 100),
        "worst_month_pct": float(monthly.min() * 100),
        "yearly": {int(k): round(float(v) * 100, 2) for k, v in yearly.items()},
    }


def _parsimony_key(p: S2Params) -> tuple:
    return (p.stop_atr is not None, p.volcap, GATES.index(p.gate), WINDOWS.index(p.window))


def run_grid(source: str = "real_training") -> None:
    run_dir = new_run_dir("sleeve2_grid")
    log.info(f"artifacts → {run_dir}")
    df1m = load_bars(source)
    cost = make_cost_cfg()          # active profile = futures_proxy_stressed
    combos = [S2Params(w, g, v, s)
              for w, g, v, s in itertools.product(WINDOWS, GATES, (False, True),
                                                  (None, 2.5))]
    rows, streams = [], {}
    for p in combos:
        out = sleeve2_run(df1m, p, cost)
        m = stream_metrics(out["daily_ret"])
        m.update(tag=p.tag(), n_trades=int(len(out["trades"])),
                 exposure_pct=round(out["exposure"] * 100, 1),
                 stop_exits=int((out["trades"]["reason"] == "stop").sum()))
        rows.append(m)
        streams[p.tag()] = out["daily_ret"]
        log.info(f"{p.tag():44s} Sharpe {m['sharpe']:5.2f}  ann {m['ann_ret_pct']:+6.2f}% "
                 f"DD {m['max_dd_pct']:5.1f}%  worst-yr {m['worst_year_pct']:+6.2f}%  "
                 f"n={m['n_trades']}")

    grid = pd.DataFrame(rows).set_index("tag")
    grid.drop(columns=["yearly"]).to_csv(run_dir / "grid.csv")

    # pre-registered pick
    eligible = grid[grid["worst_year_pct"] >= -5.0]
    pool = eligible if len(eligible) else grid
    best_sharpe = pool["sharpe"].max()
    near = pool[pool["sharpe"] >= best_sharpe - 0.05]
    by_tag = {p.tag(): p for p in combos}
    winner_tag = sorted(near.index, key=lambda t: (_parsimony_key(by_tag[t]),
                                                   -grid.loc[t, "n_trades"]))[0]
    winner = by_tag[winner_tag]
    picked = grid.loc[winner_tag]
    log.info(f"WINNER {winner_tag} (eligible {len(eligible)}/24; "
             f"picked Sharpe {picked['sharpe']:.2f} vs grid median "
             f"{grid['sharpe'].median():.2f})")

    # no-gate baseline for the winner's window (honesty reference)
    base = sleeve2_run(df1m, S2Params(window=winner.window, gate="sma50"), cost,
                       gate_override=pd.Series(True, index=streams[winner_tag].index))
    base_m = stream_metrics(base["daily_ret"])

    save_json({"winner": {"window": winner.window, "gate": winner.gate,
                          "volcap": winner.volcap, "stop_atr": winner.stop_atr},
               "winner_metrics": rows[[r["tag"] for r in rows].index(winner_tag)],
               "grid_sharpe_median": float(grid["sharpe"].median()),
               "grid_sharpe_iqr": [float(grid["sharpe"].quantile(q)) for q in (0.25, 0.75)],
               "eligible_n": int(len(eligible)),
               "nogate_baseline": base_m},
              run_dir / "grid_result.json")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    top = grid["sharpe"].nlargest(4).index
    for i, tag in enumerate(top):
        eq = (1 + streams[tag]).cumprod()
        axes[0].plot(eq.index, eq, color=SERIES[i % len(SERIES)],
                     linewidth=2 if tag == winner_tag else 1.2,
                     label=("★ " if tag == winner_tag else "") + tag)
    eqb = (1 + base["daily_ret"]).cumprod()
    axes[0].plot(eqb.index, eqb, color=MUTED, linewidth=1, linestyle="--",
                 label="no gate (baseline)")
    axes[0].set_title("S2 grid — top variants (unlevered, unit notional)")
    axes[0].legend(fontsize=7)
    yearly = pd.Series(rows[[r["tag"] for r in rows].index(winner_tag)]["yearly"])
    axes[1].bar(yearly.index.astype(str), yearly.values,
                color=[SERIES[0] if v >= 0 else SERIES[3] for v in yearly.values])
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].set_title(f"winner year returns %  ({winner_tag})")
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "grid.png")
    plt.close(fig)
    log.info(f"GRID DONE → {run_dir}")
