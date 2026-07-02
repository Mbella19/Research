"""Live signal interface — the deployable surface of the frozen system.

`daytrader signal --csv <fresh MT5 export> [--equity N]` rebuilds features on
the export, applies the FROZEN policy, and emits the decision for the latest
completed bar: side, EV, entry/SL/TP levels, and lot sizing for the given
equity. Append-logged to runs/live_signals.csv for the paper-trade record.
"""
from datetime import datetime
from pathlib import Path

import numpy as np

from ..backtest.signals import from_probabilities
from ..config import experiment, instrument
from ..models.dataset import cost_atr
from ..utils import paths
from ..utils.hashio import load_json
from ..utils.log import get_logger

log = get_logger("live")

SIGNAL_LOG = paths.RUNS_DIR / "live_signals.csv"


def run_signal(csv_path: str, equity: float | None = None) -> None:
    frozen_p = paths.MODELS_DIR / "FINAL_FROZEN.json"
    if not frozen_p.exists():
        raise SystemExit("REFUSED: no frozen artifact — live signals only come "
                         "from a frozen, validated policy (daytrader freeze).")
    frozen = load_json(frozen_p)

    from ..data.loader import load_mt5_csv
    from ..features.registry import build_features_from_1m
    from .validate import predict_champion

    p = Path(csv_path) if csv_path.startswith("/") else paths.PROJECT_ROOT / csv_path
    df1m = load_mt5_csv(p)
    feat = build_features_from_1m(df1m, list(experiment()["features"]["groups"]))
    (p_l, p_s), name = predict_champion(feat, frozen["champion"])
    dec, lab = experiment()["decision"], experiment()["labels"]
    ca = cost_atr(feat, profile=dec.get("gate_cost_profile"))
    from ..models.dataset import drift_atr

    sig = from_probabilities(feat, np.nan_to_num(p_l), np.nan_to_num(p_s),
                             lab["tp_atr"], lab["sl_atr"], ca,
                             dec["min_ev_atr"], dec["prob_floor"],
                             allowed_sides=dec.get("allowed_sides", "both"),
                             drift_atr=drift_atr(feat))
    i = len(sig) - 1
    side = int(sig["side"].iloc[i])
    ts = sig["ts"].iloc[i]
    atr_abs = float(sig["atr_abs"].iloc[i])
    close = float(feat["close"].iloc[i])
    ins = instrument()
    bt = experiment()["backtest"]
    eq = equity if equity is not None else bt["equity0"]

    # live/backtest parity: the engine refuses entries whose decision bar
    # closes after session.no_entry_after — the live surface must too.
    sess = ins["session"]
    hh, mm = sess["no_entry_after"].split(":")
    no_entry_mod = int(hh) * 60 + int(mm)
    sig_mod = ts.hour * 60 + ts.minute + int(experiment()["timeframes"]["decision_minutes"])
    if sig_mod > no_entry_mod:
        side = 0
        msg = (f"{ts} [{name}] NO TRADE (past entry cutoff "
               f"{sess['no_entry_after']} — policy never enters this late)")
    elif side == 0:
        msg = f"{ts} [{name}] NO TRADE (EV below gate)"
    else:
        dir_txt = "LONG" if side > 0 else "SHORT"
        sl = close - side * lab["sl_atr"] * atr_abs
        tp = close + side * lab["tp_atr"] * atr_abs
        risk_d = eq * bt["risk_per_trade"]
        lots = risk_d / (lab["sl_atr"] * atr_abs * ins["value_per_price_unit_per_lot"])
        lots = float(np.floor(lots / ins["lot_step"]) * ins["lot_step"])
        msg = (f"{ts} [{name}] {dir_txt} @market (~{close:.1f}) "
               f"SL {sl:.1f} TP {tp:.1f} size {lots} lots "
               f"(risk {bt['risk_per_trade']:.1%} of {eq:,.0f}) "
               f"| horizon {lab['horizon_bars']} bars"
               f"{' (swing, hold overnight ok)' if lab.get('overnight') else ' (flat by session end)'}")
    log.info(msg)

    # ── Sleeve 2 (portfolio): gated overnight drift order ────────────────
    s2_state = ""
    frozen_v3 = paths.MODELS_DIR / "FINAL_FROZEN_V3.json"
    frozen_v2 = paths.MODELS_DIR / "FINAL_FROZEN_V2.json"
    frozen_pf = frozen_v3 if frozen_v3.exists() else frozen_v2
    if frozen_pf.exists():
        from ..portfolio.overnight import S2Params, daily_frame, gate_series

        cfg = load_json(frozen_pf)["sleeves"]
        s2p = S2Params(window=cfg["s2"]["window"], gate=cfg["s2"]["gate"],
                       volcap=bool(cfg["s2"]["volcap"]),
                       stop_atr=cfg["s2"]["stop_atr"],
                       derisk=bool(cfg["s2"].get("derisk", False)))
        d = daily_frame(df1m)
        gate_on = bool(gate_series(d, s2p).iloc[-1])   # today's gate: closes ≤ D−1
        w2 = float(cfg["w2"])
        expo = 1.0
        if s2p.derisk:
            from ..portfolio.overnight import derisk_exposure

            expo = float(derisk_exposure(d).iloc[-1])
        lots2 = float(np.floor((expo * w2 * eq
                                / (close * ins["value_per_price_unit_per_lot"]))
                               / ins["lot_step"]) * ins["lot_step"])
        if gate_on:
            when = ("enter LONG at 23:00 broker time, exit tomorrow 16:30"
                    if s2p.window == "usC" else
                    "enter LONG at 01:00 broker time, exit 16:30 same day")
            stop_txt = ""
            if s2p.stop_atr:
                from ..portfolio.overnight import _daily_atr

                a = float(_daily_atr(d).iloc[-1])
                stop_txt = f", catastrophe stop {close - s2p.stop_atr * a:.1f}"
            s2_state = (f"S2 overnight sleeve GATE ON → {when}, "
                        f"size {lots2} lots (= {expo:.2f} de-risk × {w2:.2f}× equity)"
                        f"{stop_txt}")
        else:
            s2_state = "S2 overnight sleeve GATE OFF → stay flat overnight"
        log.info(s2_state)

    header = not SIGNAL_LOG.exists()
    with open(SIGNAL_LOG, "a", encoding="utf-8") as f:
        if header:
            f.write("logged_at,bar_ts,champion,side,close,atr,ev_gate,s2_gate\n")
        f.write(f"{datetime.now().isoformat()},{ts},{name},{side},{close:.2f},"
                f"{atr_abs:.2f},{dec['min_ev_atr']},"
                f"{int('GATE ON' in s2_state) if s2_state else ''}\n")
    log.info(f"appended → {SIGNAL_LOG}")
