"""Live-loop orchestration: config, seeding, executor handshake, run.

`daytrader live-loop --policy v2|v3 [--dry-run|--once]`
`daytrader live-referee --policy v2|v3`
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.hashio import load_yaml
from ..data.loader import load_mt5_csv
from ..utils import paths
from ..utils.log import get_logger
from .bus import FileBus
from .history import LiveHistory, stored_real_frames
from .loop import LiveLoop
from .policy import PolicyRuntime

log = get_logger("live.runner")


def accounts_cfg() -> dict:
    return load_yaml(paths.CONFIG_DIR / "live_accounts.yaml")


def _expand(p: str) -> Path:
    return Path(p.replace("~", str(Path.home())))


def _verify_symbol_spec(status: dict) -> None:
    """The broker's contract must match what every backtest assumed —
    refuse to trade on ANY mismatch (D-032 parity guard)."""
    from ..config import instrument
    ins = instrument()
    sym = status.get("symbol", {})
    checks = [
        ("point", float(sym.get("point", 0)), float(ins["point_size"])),
        ("digits", int(sym.get("digits", -1)), int(ins["digits"])),
    ]
    for name, got, want in checks:
        if got != want:
            raise SystemExit(f"REFUSED: symbol {name}={got} but the frozen "
                             f"policies assume {want} (instrument.yaml)")
    step = float(sym.get("volume_step", 0))
    if step != float(ins["lot_step"]):
        # broker sizing granularity differs from the backtest assumption —
        # adopted for live lot flooring (R-neutral size noise; D-032c)
        log.warning(f"broker volume_step {step} != assumed "
                    f"{ins['lot_step']} — live sizing floors to {step}")
    tick_v, tick_s = float(sym.get("tick_value", 0)), float(sym.get("tick_size", 0))
    if tick_s > 0:
        vpu = tick_v / tick_s
        if abs(vpu - float(ins["value_per_price_unit_per_lot"])) > 0.02 * vpu:
            raise SystemExit(f"REFUSED: value/unit/lot ≈ {vpu:.4f} but frozen "
                             f"policies assume {ins['value_per_price_unit_per_lot']}"
                             f" (account currency? contract change?)")
    if not status.get("trade_allowed", False):
        log.warning("terminal AutoTrading is OFF — orders may be rejected; "
                    "enable the AutoTrading button")


def prepare(policy_name: str) -> tuple:
    cfg = accounts_cfg()
    acct = cfg["accounts"][policy_name]
    login = int(acct["login"])
    data_dir = _expand(acct["data_dir_native"])
    bus = FileBus(data_dir, login)

    hist = LiveHistory(str(login))
    if not hist.load():
        log.info(f"{policy_name}: seeding history from stored sources …")
        frames = stored_real_frames()
        for c in cfg.get("seed", {}).get("extra_csvs", []):
            p = _expand(c)
            if p.exists():
                log.info(f"  + {p}")
                frames.append(load_mt5_csv(p))
        hist.seed(frames)
        hist.save()

    # handshake: wait for a live status file
    t0 = time.time()
    while bus.status() is None or bus.status_age_s() > 30:
        if time.time() - t0 > 90:
            raise SystemExit(
                f"REFUSED: no fresh status from the executor in "
                f"{data_dir} — start it first (scripts/live_up.sh) and make "
                f"sure MT5 '{acct['terminal']}' is running and logged into "
                f"{login}.")
        time.sleep(2)
    st = bus.status()
    if int(st.get("login", -1)) != login:
        raise SystemExit(f"REFUSED: executor reports login {st.get('login')} "
                         f"but this policy is bound to {login}")
    _verify_symbol_spec(st)

    # bridge any bar gap: rolling bars first, then a deep backfill if needed
    b = bus.bars()
    if b is not None and len(b):
        rep = hist.append(b)
        log.info(f"gap bridge: +{rep['n_new']} bars from rolling export "
                 f"(gap {rep['gap_s'] / 60:.0f}m)")
    gap_s = (pd.Timestamp(st["server_time"])
             - hist.bars["ts"].iloc[-1]).total_seconds()
    weekend_ok = pd.Timestamp(st["server_time"]).dayofweek >= 5
    if gap_s > 3 * 86400 and not weekend_ok:
        n = int(cfg.get("seed", {}).get("backfill_bars", 60000))
        log.info(f"history gap {gap_s / 3600:.1f}h — requesting backfill({n})")
        bus.request_backfill(n, f"{policy_name}-backfill-{int(time.time())}")
        t0 = time.time()
        while time.time() - t0 < 180:
            deep = bus.bars_deep()
            if deep is not None and len(deep) and \
                    deep["ts"].iloc[0] <= hist.bars["ts"].iloc[-1]:
                rep = hist.append(deep)
                log.info(f"backfill merged: +{rep['n_new']} bars")
                break
            time.sleep(3)
    hist.save()
    policy = PolicyRuntime.load(policy_name)
    # adopt the BROKER's sizing granularity (orders reject otherwise)
    sym = st.get("symbol", {})
    if sym:
        policy.lot_step = float(sym["volume_step"])
        policy.min_lot = float(sym["volume_min"])
        policy.max_lot = min(policy.max_lot, float(sym["volume_max"]))
        log.info(f"sizing granularity: step {policy.lot_step} "
                 f"min {policy.min_lot} max {policy.max_lot}")
    return policy, hist, bus


def run_live(policy_name: str, dry_run: bool = False, once: bool = False):
    policy, hist, bus = prepare(policy_name)
    # keep the Mac awake while the loop lives (idle sleep kills Wine + feed)
    try:
        import os
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except OSError:
        log.warning("caffeinate unavailable — keep the Mac awake manually")
    loop = LiveLoop(policy, hist, bus, dry_run=dry_run)
    if once:
        loop.step()
        loop._save_state()
        return
    # periodic history persistence piggybacked on the run loop
    orig_step = loop.step
    state = {"last_save": time.time()}

    def step_and_save():
        r = orig_step()
        if time.time() - state["last_save"] > 1800:
            state["last_save"] = time.time()
            hist.save()
        return r

    loop.step = step_and_save
    loop.run()


def run_referee(policy_name: str) -> dict:
    """Weekly retrospective: recompute every decision from FULL history with
    the frozen policy and assert side-equality against decisions.csv; plus a
    trades summary. Writes runs/live/<policy>/referee_<date>.json."""
    from ..live.features_live import FeatureEngine

    cfg = accounts_cfg()
    acct = cfg["accounts"][policy_name]
    hist = LiveHistory(str(acct["login"]))
    if not hist.load():
        raise SystemExit("no live history yet")
    policy = PolicyRuntime.load(policy_name)
    eng = FeatureEngine(hist, policy.groups)
    sig = policy.s1_signals(eng.full_frame()).set_index("ts")

    out_dir = paths.RUNS_DIR / "live" / policy_name
    dec_f = out_dir / "decisions.csv"
    rep = {"policy": policy_name, "when": str(pd.Timestamp.now()),
           "n_decisions": 0, "n_compared": 0, "side_mismatches": [],
           "trades": {}}
    if dec_f.exists():
        dec = pd.read_csv(dec_f)
        dec["ts"] = pd.to_datetime(dec["ts"])
        acted = dec[dec["action"].isin(["OPEN", "no_trade"])]
        rep["n_decisions"] = int(len(dec))
        for _, row in acted.iterrows():
            if row["ts"] not in sig.index:
                continue
            rep["n_compared"] += 1
            want = int(sig.loc[row["ts"], "side"])
            got = int(row["side"]) if str(row["side"]).strip() else 0
            if want != got:
                rep["side_mismatches"].append(
                    {"ts": str(row["ts"]), "logged": got, "recomputed": want})
    tr_f = out_dir / "trades.csv"
    if tr_f.exists():
        tr = pd.read_csv(tr_f)
        for sl in ("s1", "s2"):
            t = tr[tr["sleeve"] == sl]
            if len(t):
                rep["trades"][sl] = {
                    "n": int(len(t)),
                    "sum_R_or_ret": float(t["R_or_ret"].sum()),
                    "win_rate": float((t["R_or_ret"] > 0).mean()),
                    "sum_pnl": float(t["pnl"].sum()),
                    "sum_swap": float(t["swap"].sum()),
                }
    ok = not rep["side_mismatches"]
    rep["ok"] = ok
    out = out_dir / f"referee_{pd.Timestamp.now():%Y%m%d}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    log.info(f"referee {policy_name}: decisions={rep['n_decisions']} "
             f"compared={rep['n_compared']} mismatches="
             f"{len(rep['side_mismatches'])} → {out}")
    if not ok:
        log.error("SIDE MISMATCHES FOUND — investigate before continuing "
                  "the paper trade")
    return rep
