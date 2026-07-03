"""THE D-032 acceptance gate: the real LiveLoop, driven bar-by-bar through
SimBus over ~90 trading days INCLUDING the FWD-01 cold window, must
reproduce the batch engine (`run_backtest`) / S2 executor (`sleeve2_run`)
trade-for-trade for BOTH frozen policies.

Slow (four full replays + references).  pytest -m slow -k parity
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daytrader.backtest.engine import make_cost_cfg, make_risk_cfg, run_backtest
from daytrader.config import clear_overrides
from daytrader.live.replay import run_replay
from daytrader.portfolio.overnight import daily_frame, sleeve2_run

START = "2026-02-23"                      # ~90 trading days incl. cold window
END = "2026-07-02 23:59"
NEW_CSV = Path.home() / "Downloads" / "new data.csv"


@pytest.fixture(autouse=True)
def _clean():
    yield
    clear_overrides()


def _win(df, tcol="ts"):
    return df[(df[tcol] >= pd.Timestamp(START))
              & (df[tcol] <= pd.Timestamp(END))].reset_index(drop=True)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["v2", "v3"])
def test_s1_parity(name):
    """Two-phase gate (D-032a). The engine settles an intrabar stop at bar e
    and re-enters at e's open — physically unknowable live ("samebar" class,
    ~2-4%% of trades); live captures those ≤1 bar late, and the fill drift
    cascades (equity→lots; shifted exits→concurrency windows). Therefore:
    Phase A: EXACT trade equality on the longest samebar-free stretch
             (fresh replay, equity aligned at the stretch start);
    Phase B: full window — every deviation must postdate the first samebar
             event, and aggregates stay tight."""
    r = run_replay(name, START, END, sleeves=("s1",),
                   new_csv=NEW_CSV if NEW_CSV.exists() else None)
    ref = run_backtest(_win(r["bars"]), _win(r["signals"]),
                       make_cost_cfg(), make_risk_cfg())["trades"]
    exits = set(ref["exit_ts"])
    entry_bar = ref["entry_ts"] + pd.Timedelta(minutes=5)
    sb_ts = sorted(ref.loc[entry_bar.isin(exits), "entry_ts"])
    print(f"\n[{name}/S1] engine {len(ref)} trades, samebar events "
          f"{len(sb_ts)} ({len(sb_ts) / max(len(ref), 1):.1%})")
    assert len(sb_ts) / max(len(ref), 1) <= 0.06

    # ── Phase A: exact equality on the longest event-free stretch ────────
    bounds = ([pd.Timestamp(START)] + [t for t in sb_ts]
              + [pd.Timestamp(END)])
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    a0, a1 = max(spans, key=lambda ab: ab[1] - ab[0])
    a0 = (a0 + pd.Timedelta(days=1)).normalize()      # start flat, next day
    a1 = a1.normalize() - pd.Timedelta(minutes=1)     # end before the event
    print(f"[{name}/S1] Phase A stretch: {a0} → {a1} "
          f"({(a1 - a0).days} calendar days)")
    assert (a1 - a0).days >= 20, "no usable samebar-free stretch"
    ra = run_replay(name, str(a0), str(a1), sleeves=("s1",),
                    new_csv=NEW_CSV if NEW_CSV.exists() else None)

    def _w(df, t="ts"):
        return df[(df[t] >= a0) & (df[t] <= a1)].reset_index(drop=True)

    refa = run_backtest(_w(ra["bars"]), _w(ra["signals"]),
                        make_cost_cfg(), make_risk_cfg())["trades"]
    tra = ra["trades"]
    tra = tra[tra.sleeve == "s1"].copy()
    tra["decision_ts"] = pd.to_datetime(tra["decision_ts"])
    tra = tra.rename(columns={c: c + "_l" for c in tra.columns
                              if c != "decision_ts"})
    ma = refa.merge(tra, left_on="entry_ts", right_on="decision_ts",
                    how="outer", indicator=True)
    bad = ma[ma["_merge"] != "both"]
    print(f"[{name}/S1] Phase A: engine {len(refa)} vs loop {len(tra)}, "
          f"unmatched {len(bad)}")
    assert len(bad) == 0, bad[["entry_ts", "reason", "decision_ts",
                               "reason_l"]].to_string()
    assert (ma["side"] == ma["side_l"]).all()
    assert (ma["exit_ts"] == pd.to_datetime(ma["exit_ts_l"])).all()
    assert (ma["reason"] == ma["reason_l"]).all()
    for ce, cl, tol in [("entry", "entry_l", 1e-6), ("exit", "exit_l", 1e-6),
                        ("lots", "lots_l", 1e-9), ("R", "R_or_ret_l", 1e-5)]:
        d = float((ma[ce].astype(float) - ma[cl].astype(float)).abs().max())
        print(f"[{name}/S1] Phase A max|Δ {ce}| = {d:.3g}")
        assert d <= tol, f"{ce} diverged in the event-free stretch: {d}"

    # ── Phase B: full window, deviations traceable + aggregates tight ────
    tr = r["trades"]
    tr = tr[tr.sleeve == "s1"].copy()
    tr["decision_ts"] = pd.to_datetime(tr["decision_ts"])
    tr = tr.rename(columns={c: c + "_l" for c in tr.columns
                            if c != "decision_ts"})
    m = ref.merge(tr, left_on="entry_ts", right_on="decision_ts",
                  how="outer", indicator=True)
    first_sb = sb_ts[0] if sb_ts else pd.Timestamp(END)
    both = m[m["_merge"] == "both"]
    pre = both[both["entry_ts"] < first_sb]
    if len(pre):
        exact = ((pre["entry"].astype(float)
                  - pre["entry_l"].astype(float)).abs() < 1e-6)
        assert exact.all(), "pre-event trades must be exact"
    odd = m[m["_merge"] != "both"]
    when = odd["entry_ts"].fillna(odd["decision_ts"])
    assert (when >= first_sb).all(), \
        f"unexplained deviation BEFORE the first samebar event:\n{odd}"
    n_e, n_l = len(ref), len(tr)
    sum_e = float(ref["R"].sum())
    sum_l = float(tr["R_or_ret_l"].astype(float).sum())
    n_sb = len(sb_ts)
    print(f"[{name}/S1] Phase B: counts {n_e} vs {n_l}, "
          f"ΣR {sum_e:.2f} vs {sum_l:.2f} (drift {abs(sum_e - sum_l):.2f}R "
          f"over {n_sb} samebar events), deviations {len(odd)} "
          f"(all ≥ first samebar {first_sb})")
    assert abs(n_e - n_l) <= max(3, int(0.04 * n_e))
    # per-event outcome swing is bounded by the barrier span (tp+sl = 4.5R):
    # a re-entry one bar into a stop cascade can flip tp↔sl. Measured drift
    # is ledgered (D-032a); anything beyond the bound = machinery bug.
    assert abs(sum_e - sum_l) <= max(1.5, 4.5 * n_sb)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["v2", "v3"])
def test_s2_parity(name):
    r = run_replay(name, START, END, sleeves=("s2",),
                   new_csv=NEW_CSV if NEW_CSV.exists() else None)
    p = r["policy"]
    ref = sleeve2_run(_win(r["bars"]), p.s2p, make_cost_cfg(),
                      swap_bp_night=0.0,
                      context_daily=daily_frame(r["bars"]))["trades"]
    tr = r["trades"]
    tr = tr[tr.sleeve == "s2"].copy() if len(tr) else tr
    tr["entry_ts"] = pd.to_datetime(tr["entry_ts"])
    # the final replay hold may be open-ended (engine skips a last-day entry
    # it can't see an exit day for) — exempt trailing engine-only rows on the
    # last entry day and trailing loop-open positions (D-032)
    m = ref.merge(tr, on="entry_ts", suffixes=("_e", "_l"),
                  how="outer", indicator=True)
    tail_day = max(ref["entry_ts"].max(), tr["entry_ts"].max()).normalize()
    core = m[(m["_merge"] == "both") | (m["entry_ts"].dt.normalize() < tail_day)]
    unmatched = core[core["_merge"] != "both"]
    print(f"\n[{name}/S2] engine {len(ref)} vs loop {len(tr)} holds, "
          f"unmatched(core) {len(unmatched)}")
    assert len(unmatched) == 0, unmatched.to_string()
    b = m[m["_merge"] == "both"]
    for ce, cl, tol in [("entry_e", "entry_l", 1e-6),
                        ("exit_e", "exit_l", 1e-6),
                        ("expo_e", "expo_l", 1e-9)]:
        d = float((b[ce].astype(float) - b[cl].astype(float)).abs().max())
        print(f"[{name}/S2] max|Δ {ce}| = {d:.3g}")
        assert d <= tol
    # engine ret_net = expo × per-notional return; loop logs per-notional
    d = float((b["ret_net"].astype(float)
               - b["expo_e"].astype(float) * b["R_or_ret"].astype(float))
              .abs().max())
    print(f"[{name}/S2] max|Δ ret| = {d:.3g}")
    assert d <= 1e-5
    # stop fires must agree (v3 has the catastrophe stop, v2 has none)
    eng_stops = set(ref.loc[ref["reason"] == "stop", "entry_ts"])
    loop_stops = set(b.loc[b["reason_l"].astype(str).str.contains("sl"),
                           "entry_ts"])
    assert eng_stops == loop_stops, (eng_stops, loop_stops)
    if name == "v2":
        assert tr["sl"].isna().all() or (tr["sl"].astype(str) == "").all()
