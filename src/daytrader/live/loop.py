"""Live event loop — one policy, one account, one process (D-032).

The loop is a pure state machine over three inputs (bars, status, reports)
and one output (orders). It is driven identically by the real FileBus and
by the replay SimBus, so the replay parity test exercises THIS exact code.

Engine-parity rules implemented here (sources in parentheses):
- decisions only on COMPLETED 5m bins; the decidable bin for last 1m bar L
  is floor_5m(L − 4min) (a bin is complete when its 5th minute closed);
- entry guard order == engine.py:147-166: settle → concurrent → day reset →
  day_trades/day_R → ATR finite → cutoff (decision close > 22:45) → sizing
  (a sizing-floor skip does NOT count as a trade);
- if a close order is due at/before the entry bar, the loop WAITS for its
  fill report before evaluating the entry (engine settles exits with
  exit_idx ≤ e first) — bounded by the avail+90s MISSED rule;
- S1 horizon exit = 240 RECEIVED 1m bars after the entry bar (engine counts
  frame indices); force-flat at 23:30; EARLY_FLAT if feed dies late in the
  session with an open position;
- S2: enter at first bar ≥ 23:00 when gate ON (no bar that day ⇒ no entry),
  exit at first bar ≥ 16:30 of the next day PRESENT IN DATA (weekend/holiday
  holds match sleeve2_run's day indexing); gate/expo/dailyATR captured at
  entry; v3 broker stop = fill − 5×dailyATR, v2 no stop;
- MISSED rule: an entry decision older than avail_ts + 90s is skipped and
  logged, never chased. Freshness: stale bars ⇒ no entries, exits still run.
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import paths
from ..utils.log import get_logger
from .features_live import FeatureEngine
from .history import LiveHistory
from .policy import S2_ENTRY_MOD, S2_EXIT_MOD, PolicyRuntime

log = get_logger("live.loop")

MISSED_AFTER_S = 90.0
STALE_BARS_S = 90.0
EARLY_FLAT_STALE_S = 600.0
EARLY_FLAT_AFTER_MOD = 21 * 60
ORDER_TTL_S = 120.0
DEBOUNCE_GONE = 3          # consecutive statuses without the position
MAX_CONSEC_REJECTS = 3


def _mod(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _floor5(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("5min")


def complete_bin(last_1m_ts: pd.Timestamp) -> pd.Timestamp:
    """Newest COMPLETE 5m bin given the newest received 1m bar."""
    return _floor5(last_1m_ts - pd.Timedelta(minutes=4))


@dataclass
class OpenTrade:
    sleeve: str                 # "s1" | "s2"
    order_id: str
    ticket: int | None
    side: int
    lots: float
    entry_fill: float | None
    sl: float | None
    tp: float | None
    atr_abs: float | None       # s1
    risk_dollars: float | None  # s1
    expo: float | None          # s2
    datr: float | None          # s2
    decision_ts: str            # s1: decision bin; s2: entry day
    entry_bar_ts: str | None    # first 1m bar at/after the fill
    entry_day: str
    filled: bool = False
    close_confirmed: bool = False   # a CLOSE fill reported; awaiting reconcile


@dataclass
class DayCounters:
    date: str = ""
    trades: int = 0
    R: float = 0.0


class LiveLoop:
    def __init__(self, policy: PolicyRuntime, hist: LiveHistory, bus,
                 out_dir: Path | None = None, dry_run: bool = False,
                 sim_signals: pd.DataFrame | None = None,
                 sleeves: tuple = ("s1", "s2")):
        self.p = policy
        self.hist = hist
        self.bus = bus
        self.dry = dry_run
        self.sleeves = sleeves
        self.out = out_dir or (paths.RUNS_DIR / "live" / policy.name)
        self.out.mkdir(parents=True, exist_ok=True)
        self.eng = FeatureEngine(hist, policy.groups)
        # replay mode: precomputed signal frame (ts-indexed); rowwise purity +
        # prefix stability make this identical to per-step rebuilds
        self._sim_sig = (sim_signals.set_index("ts")
                         if sim_signals is not None else None)

        self.state_path = self.out / "state.json"
        self.order_seq = 0
        self.last_decision_ts: pd.Timestamp | None = None
        self.s1: OpenTrade | None = None
        self.s2: OpenTrade | None = None
        self.closing: list[OpenTrade] = []     # close-in-flight, to finalize
        self.pending: dict[str, dict] = {}     # order_id -> meta
        self.day = DayCounters()
        self.entries_halted = False
        self.halt_reason: str | None = None
        self.consec_rejects = 0
        self._gone_count: dict[str, int] = {}  # sleeve -> consecutive absent
        self._s2_attempted_day: str = ""
        self._last_report_idx = 0
        self._dirty = False
        self._decisions_f = self.out / "decisions.csv"
        self._trades_f = self.out / "trades.csv"
        self._equity_f = self.out / "equity.csv"
        self._load_state()

    # ── persistence ──────────────────────────────────────────────────────
    def _load_state(self):
        if not self.state_path.exists():
            return
        st = json.loads(self.state_path.read_text())
        self.order_seq = st.get("order_seq", 0)
        self.last_decision_ts = (pd.Timestamp(st["last_decision_ts"])
                                 if st.get("last_decision_ts") else None)
        self.s1 = OpenTrade(**st["s1"]) if st.get("s1") else None
        self.s2 = OpenTrade(**st["s2"]) if st.get("s2") else None
        self.closing = [OpenTrade(**c) for c in st.get("closing", [])]
        self.pending = st.get("pending", {})
        d = st.get("day", {})
        self.day = DayCounters(**d) if d else DayCounters()
        self.entries_halted = st.get("entries_halted", False)
        self.halt_reason = st.get("halt_reason")
        self._s2_attempted_day = st.get("s2_attempted_day", "")
        self._last_report_idx = st.get("last_report_idx", 0)

    def _save_state(self):
        if not self._dirty:
            return
        self._dirty = False
        st = {
            "policy": self.p.name, "order_seq": self.order_seq,
            "last_decision_ts": (str(self.last_decision_ts)
                                 if self.last_decision_ts is not None else None),
            "s1": asdict(self.s1) if self.s1 else None,
            "s2": asdict(self.s2) if self.s2 else None,
            "closing": [asdict(c) for c in self.closing],
            "pending": self.pending, "day": asdict(self.day),
            "entries_halted": self.entries_halted,
            "halt_reason": self.halt_reason,
            "s2_attempted_day": self._s2_attempted_day,
            "last_report_idx": self._last_report_idx,
        }
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def _csv(self, path: Path, header: str, row: str):
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(header + "\n")
            f.write(row + "\n")

    def _log_decision(self, ts, side, ev, action, reason=""):
        self._csv(self._decisions_f,
                  "ts,side,ev_atr,action,reason",
                  f"{ts},{side},{'' if ev is None else f'{ev:.6f}'},{action},{reason}")

    def _log_trade(self, t: OpenTrade, exit_px, exit_ts, pnl, swap,
                   commission, reason, r_or_ret):
        self._csv(self._trades_f,
                  "sleeve,decision_ts,side,lots,entry_ts,entry,exit_ts,exit,"
                  "sl,tp,pnl,swap,commission,R_or_ret,reason,order_id,expo",
                  f"{t.sleeve},{t.decision_ts},{t.side},{t.lots},"
                  f"{t.entry_bar_ts},{t.entry_fill},{exit_ts},{exit_px},"
                  f"{t.sl},{t.tp},{pnl:.2f},{swap:.2f},{commission:.2f},"
                  f"{r_or_ret:.6f},{reason},{t.order_id},"
                  f"{'' if t.expo is None else t.expo}")

    # ── order plumbing ───────────────────────────────────────────────────
    def _next_id(self) -> str:
        self.order_seq += 1
        self._dirty = True
        return f"{self.p.name}-{self.order_seq:06d}"

    def _send(self, kind: str, srv_now: pd.Timestamp, **kw) -> str:
        oid = self._next_id()
        order = {"id": oid, "policy": self.p.name,
                 "created_srv": str(srv_now), **kw}
        self.pending[oid] = {"kind": kind, "created_srv": str(srv_now),
                             **{k: v for k, v in kw.items()
                                if k in ("magic", "ticket")}}
        if self.dry:
            log.info(f"[DRY] would send {kind}: {order}")
            self.pending.pop(oid)
            return oid
        self.bus.send_order(order)
        log.info(f"sent {kind} {oid}: {order}")
        return oid

    def _close_position(self, t: OpenTrade, srv_now, reason: str):
        already = any(m.get("kind", "").endswith("_close")
                      and m.get("ticket") == t.ticket for m in self.pending.values())
        if already or not t.filled or t.close_confirmed:
            return
        self._send(f"{t.sleeve}_close", srv_now, action="CLOSE",
                   magic=(self.p.magic_s1 if t.sleeve == "s1" else self.p.magic_s2),
                   ticket=t.ticket, lots=t.lots, comment=reason)

    # ── report / reconciliation processing ───────────────────────────────
    def _process_reports(self, srv_now):
        reps = self.bus.reports(self._last_report_idx)
        for rep in reps:
            self._last_report_idx = rep["_idx"] + 1
            oid = rep.get("id")
            meta = self.pending.pop(oid, None)
            if meta is None:
                continue
            kind = meta["kind"]
            self._dirty = True
            if not rep.get("ok"):
                err = str(rep.get("error", ""))
                # 10027/10026: AutoTrading not armed — an operator condition,
                # not an order fault; entries are simply MISSED until armed
                unarmed = str(rep.get("retcode")) in ("10027", "10026")
                benign = unarmed or (kind.endswith("_close")
                          and ("not found" in err or "expired" in err))
                if unarmed:
                    log.error("*** ENABLE THE AutoTrading BUTTON in this "
                              "policy's MT5 terminal — orders are rejected "
                              "until it is ON ***")
                log.error(f"order {oid} ({kind}) REJECTED: {err} "
                          f"retcode={rep.get('retcode')}")
                if kind == "s1_open":
                    self.s1 = None
                elif kind == "s2_open":
                    self.s2 = None
                if not benign:   # close-vs-SL races resolve via reconciliation
                    self.consec_rejects += 1
                    if self.consec_rejects >= MAX_CONSEC_REJECTS:
                        self._halt_entries(f"{MAX_CONSEC_REJECTS} consecutive "
                                           f"order rejections")
                continue
            self.consec_rejects = 0
            if kind in ("s1_open", "s2_open"):
                t = self.s1 if kind == "s1_open" else self.s2
                if t is None or t.order_id != oid:
                    log.error(f"fill for unknown open {oid}")
                    continue
                t.ticket = int(rep["ticket"])
                t.entry_fill = float(rep["fill_price"])
                t.lots = float(rep.get("fill_volume", t.lots))
                t.sl = rep.get("sl")
                t.tp = rep.get("tp")
                t.entry_bar_ts = str(pd.Timestamp(rep["srv_time"]).floor("1min"))
                t.filled = True
                log.info(f"{kind} filled @ {t.entry_fill} lots {t.lots} "
                         f"sl {t.sl} tp {t.tp} ticket {t.ticket}")
            elif kind in ("s1_close", "s2_close"):
                # position disappearance path finalizes with deal data; the
                # latch stops duplicate closes during the reconcile debounce
                for t in (self.s1, self.s2, *self.closing):
                    if t is not None and t.ticket == rep.get("ticket"):
                        t.close_confirmed = True
        # order TTL: pending too long = executor never saw / rejected stale
        for oid in list(self.pending):
            m = self.pending[oid]
            age = (srv_now - pd.Timestamp(m["created_srv"])).total_seconds()
            if age > ORDER_TTL_S * 2:
                log.error(f"order {oid} ({m['kind']}) unreported for {age:.0f}s "
                          f"— dropping (executor rejects stale orders)")
                if m["kind"] == "s1_open" and self.s1 and self.s1.order_id == oid:
                    self.s1 = None
                if m["kind"] == "s2_open" and self.s2 and self.s2.order_id == oid:
                    self.s2 = None
                self.pending.pop(oid)

    def _reconcile_positions(self, status: dict, srv_now):
        live_tickets = {int(p["ticket"]) for p in status.get("positions", [])}
        for t in [x for x in (self.s1, self.s2, *self.closing)
                  if x is not None and x.filled]:
            key = str(t.ticket)
            if t.ticket in live_tickets:
                self._gone_count[key] = 0
                continue
            # an absent position with a closing deal on record is EXPLAINED
            # (SL/TP or our close) — finalize immediately so the very next
            # decision sees the slot free, exactly like the engine settling
            # exits with exit_idx ≤ e (engine.py:147-155). The debounce only
            # escalates the deal-less case (transient positions_get() glitch:
            # position reappears, counter resets; truly gone with no deal ⇒
            # keep retrying and log).
            self._gone_count[key] = self._gone_count.get(key, 0) + 1
            if self._finalize_from_deals(t, srv_now):
                if t is self.s1:
                    self.s1 = None
                elif t is self.s2:
                    self.s2 = None
                else:
                    self.closing.remove(t)
                self._gone_count.pop(key, None)
            elif self._gone_count[key] >= DEBOUNCE_GONE:
                log.error(f"{t.sleeve} ticket {t.ticket} absent for "
                          f"{self._gone_count[key]} statuses with NO closing "
                          f"deal — retrying")
        # orphans: our magic at the broker, unknown to us
        known = {t.ticket for t in (self.s1, self.s2, *self.closing)
                 if t and t.ticket}
        for pos in status.get("positions", []):
            if (int(pos["magic"]) in (self.p.magic_s1, self.p.magic_s2)
                    and int(pos["ticket"]) not in known):
                self._halt_entries(f"ORPHAN position ticket {pos['ticket']} "
                                   f"magic {pos['magic']} — adopt/flatten "
                                   f"manually")

    def _finalize_from_deals(self, t: OpenTrade, srv_now) -> bool:
        deals = self.bus.deals(ticket=t.ticket)
        outs = [d for d in deals if d.get("entry") == "out"]
        if not outs:
            log.error(f"{t.sleeve} ticket {t.ticket} gone but no closing deal "
                      f"yet — will retry")
            return False
        px = float(np.average([d["price"] for d in outs],
                              weights=[d["volume"] for d in outs]))
        pnl = sum(float(d["profit"]) for d in outs)
        # commission/swap accrue on BOTH legs (MT5 books entry commission on
        # the in-deal) — engine charges 2× per round trip (engine.py:218)
        swap = sum(float(d.get("swap", 0.0)) for d in deals)
        comm = sum(float(d.get("commission", 0.0)) for d in deals)
        when = str(max(pd.Timestamp(d["time"]) for d in outs))
        reason = outs[-1].get("reason", "")
        closed_by_us = any(m.get("ticket") == t.ticket and
                           m["kind"].endswith("_close")
                           for m in self.pending.values())
        self._dirty = True
        if t.sleeve == "s1":
            r = (pnl + swap + comm) / max(t.risk_dollars or 1e-9, 1e-9)
            self._roll_day(pd.Timestamp(when).normalize())
            self.day.R += r
            self._log_trade(t, px, when, pnl, swap, comm,
                            reason or ("close" if closed_by_us else "sl/tp"), r)
            log.info(f"S1 closed @ {px} pnl {pnl:.2f} R {r:.3f} "
                     f"(day R {self.day.R:.2f})")
        else:
            base = (t.entry_fill or px) * t.lots * self.p.vpu
            ret = (pnl + swap + comm) / max(base, 1e-9)
            self._log_trade(t, px, when, pnl, swap, comm,
                            reason or ("window" if closed_by_us else "stop"), ret)
            log.info(f"S2 closed @ {px} pnl {pnl:.2f} ret {ret:.4%}")
        # drop any stale close order for this ticket
        for oid in [o for o, m in self.pending.items()
                    if m.get("ticket") == t.ticket]:
            self.pending.pop(oid)
        return True

    def _halt_entries(self, reason: str):
        if not self.entries_halted:
            self.entries_halted = True
            self.halt_reason = reason
            self._dirty = True
            log.error(f"ENTRIES HALTED: {reason}")

    def _roll_day(self, day: pd.Timestamp):
        ds = str(day.date())
        if self.day.date != ds:
            self.day = DayCounters(date=ds)
            self._dirty = True

    # ── decision path ────────────────────────────────────────────────────
    def _signal_row(self, bin_ts: pd.Timestamp):
        """Signal for a completed bin: precomputed (replay) or fresh full
        rebuild (live). Rowwise purity + prefix stability make them equal."""
        if self._sim_sig is not None:
            if bin_ts not in self._sim_sig.index:
                return None
            r = self._sim_sig.loc[bin_ts]
            return {"side": int(r["side"]), "ev_atr": float(r["ev_atr"]),
                    "atr_abs": float(r["atr_abs"])}
        feat = self.eng.decision_frame()
        rows = feat[feat["ts"] == bin_ts]
        if not len(rows):
            return None
        sig = self.p.s1_signals(feat, tail_rows=600)
        m = sig[sig["ts"] == bin_ts]
        if not len(m):
            return None
        r = m.iloc[-1]
        return {"side": int(r["side"]), "ev_atr": float(r["ev_atr"]),
                "atr_abs": float(r["atr_abs"])}

    def _maybe_decide_s1(self, srv_now, status):
        last_ts = self.hist.bars["ts"].iloc[-1]
        b = complete_bin(last_ts)
        if self.last_decision_ts is not None and b <= self.last_decision_ts:
            return
        # engine guard order (engine.py:147-166). Cheap guards precede the
        # feature build; every skip is logged for the retrospective referee.
        self.last_decision_ts = b
        self._dirty = True
        avail = b + pd.Timedelta(minutes=5)
        self._roll_day(b.normalize())
        if self.entries_halted:
            self._log_decision(b, "", None, "skip", "entries_halted")
            return
        if _mod(b) + 5 > self.p.no_entry_mod:         # decision close > 22:45
            self._log_decision(b, "", None, "skip", "past_entry_cutoff")
            return                                    # engine.py:144
        # engine settles exits with exit_idx ≤ e before evaluating the entry
        # (engine.py:147-155): a position whose close is already in flight at
        # this bar counts as CLOSED for concurrency. Its R lands in day_R a
        # step later — the pre-registered D-032 micro-divergence.
        if self.s1 is not None:
            closing_now = (self.s1.close_confirmed or any(
                m["kind"] == "s1_close" and m.get("ticket") == self.s1.ticket
                for m in self.pending.values()))
            if closing_now:
                self.closing.append(self.s1)
                self.s1 = None
            elif (srv_now - avail).total_seconds() <= MISSED_AFTER_S:
                # the engine settles exits with exit_idx == e and re-enters
                # at the SAME bar (engine.py:147-155) — unknowable at bar
                # open when the exit is an intrabar stop. Retry inside the
                # missed-window: if the stop fires within decision latency,
                # the entry is captured (≤1 bar late; D-032 samebar class).
                self.last_decision_ts = b - pd.Timedelta(minutes=5)
                return
            else:                                     # max_concurrent = 1
                self._log_decision(b, "", None, "skip", "position_open")
                return
        if self.day.trades >= 6:
            self._log_decision(b, "", None, "skip", "max_trades_per_day")
            return
        if self.day.R <= -3.0:
            self._log_decision(b, "", None, "skip", "daily_stop_R")
            return
        if (srv_now - avail).total_seconds() > MISSED_AFTER_S:
            self._log_decision(b, "", None, "skip", "MISSED_late")
            return
        row = self._signal_row(b)
        if row is None:
            self._log_decision(b, "", None, "skip", "no_bin")
            return
        side, ev, atr = row["side"], row["ev_atr"], row["atr_abs"]
        if side == 0:
            self._log_decision(b, 0, ev, "no_trade", "gate")
            return
        if not np.isfinite(atr) or atr <= 0:
            self._log_decision(b, side, ev, "skip", "atr_not_finite")
            return
        eq = float(status["equity"])
        close_px = float(self.hist.bars["close"].iloc[-1])
        lots, risk_d = self.p.s1_lots(eq, atr, close_px)
        if lots <= 0:
            self._log_decision(b, side, ev, "skip", "sizing_floor")
            return   # does NOT count toward day_trades (engine.py:214-216)
        if not self._margin_ok(status, lots, close_px):
            self._log_decision(b, side, ev, "skip", "margin_guard")
            return
        sl_dist = self.p.labels["sl_atr"] * atr
        tp_dist = self.p.labels["tp_atr"] * atr
        # distances at full precision — broker rounding happens at the
        # executor (the ledgered 0.1-step micro-divergence lives THERE only)
        oid = self._send("s1_open", srv_now, action="OPEN",
                         magic=self.p.magic_s1, side=side, lots=lots,
                         sl_dist=float(sl_dist), tp_dist=float(tp_dist),
                         comment=f"{self.p.name}-s1")
        self.day.trades += 1
        self.s1 = OpenTrade(sleeve="s1", order_id=oid, ticket=None, side=side,
                            lots=lots, entry_fill=None, sl=None, tp=None,
                            atr_abs=atr, risk_dollars=risk_d, expo=None,
                            datr=None, decision_ts=str(b), entry_bar_ts=None,
                            entry_day=str(b.normalize().date()))
        self._log_decision(b, side, ev, "OPEN", f"lots={lots}")

    def _margin_ok(self, status, lots, px) -> bool:
        lev = float(status.get("leverage", 100.0)) or 100.0
        need = lots * px * self.p.vpu / lev
        free = float(status.get("margin_free", status["equity"]))
        return need <= 0.6 * free

    # ── exits ────────────────────────────────────────────────────────────
    def _check_s1_exits(self, srv_now, bars_fresh):
        t = self.s1
        if t is None or not t.filled:
            return
        n_since = self._bars_since(pd.Timestamp(t.entry_bar_ts))
        if n_since >= self.p.horizon_1m_bars():
            self._close_position(t, srv_now, "time")
            return
        if _mod(srv_now) >= self.p.flat_mod and bars_fresh:
            self._close_position(t, srv_now, "flat")
            return
        # early-close day without a 23:30 bar: never hold S1 across days
        # (engine clips at the day's last bar, engine.py:114-119/202-206)
        if self.hist.bars["ts"].iloc[-1].normalize() > pd.Timestamp(t.entry_day):
            self._close_position(t, srv_now, "day_rollover")
            return
        if (_mod(srv_now) >= EARLY_FLAT_AFTER_MOD and not bars_fresh
                and self._bars_age(srv_now) > EARLY_FLAT_STALE_S):
            self._close_position(t, srv_now, "early_flat")

    def _check_s2(self, srv_now, status, bars_fresh):
        mod = _mod(srv_now)
        today = srv_now.normalize()
        t = self.s2
        # exit: first fresh-bar day AFTER entry day, at/after 16:30
        if t is not None and t.filled:
            entry_day = pd.Timestamp(t.entry_day)
            last_bar_day = self.hist.bars["ts"].iloc[-1].normalize()
            if (last_bar_day > entry_day and bars_fresh
                    and mod >= S2_EXIT_MOD):
                self._close_position(t, srv_now, "window")
            return
        if t is not None:      # sent, awaiting fill
            return
        # entry: 23:00–23:59, gate on, fresh bars, once per day
        if self.entries_halted or not bars_fresh:
            return
        if not (S2_ENTRY_MOD <= mod < 24 * 60):
            return
        if self._s2_attempted_day == str(today.date()):
            return
        self._s2_attempted_day = str(today.date())
        self._dirty = True
        st2 = self.p.s2_state(self.hist.daily_ohlc())
        if not st2["gate"]:
            self._log_decision(srv_now.floor("1min"), "", None,
                               "s2_no_trade", "gate_off")
            return
        eq = float(status["equity"])
        close_px = float(self.hist.bars["close"].iloc[-1])
        lots = self.p.s2_lots(eq, close_px, st2["expo"])
        if lots < self.p.min_lot:
            self._log_decision(srv_now.floor("1min"), "", None,
                               "s2_skip", "sizing_floor")
            return
        sl_dist = (self.p.s2p.stop_atr * st2["datr"]
                   if self.p.s2p.stop_atr and np.isfinite(st2["datr"]) else None)
        oid = self._send("s2_open", srv_now, action="OPEN",
                         magic=self.p.magic_s2, side=1, lots=lots,
                         sl_dist=(float(sl_dist) if sl_dist else None),
                         tp_dist=None, comment=f"{self.p.name}-s2")
        self.s2 = OpenTrade(sleeve="s2", order_id=oid, ticket=None, side=1,
                            lots=lots, entry_fill=None, sl=None, tp=None,
                            atr_abs=None, risk_dollars=None,
                            expo=st2["expo"], datr=st2["datr"],
                            decision_ts=str(today.date()), entry_bar_ts=None,
                            entry_day=str(today.date()))
        self._log_decision(srv_now.floor("1min"), 1, None, "S2_OPEN",
                           f"lots={lots} expo={st2['expo']:.3f}")

    # ── helpers ──────────────────────────────────────────────────────────
    def _bars_since(self, ts: pd.Timestamp) -> int:
        a = self.hist.bars["ts"].to_numpy()
        return int(len(a) - np.searchsorted(a, np.datetime64(ts), "left"))

    def _bars_age(self, srv_now) -> float:
        return (srv_now - self.hist.bars["ts"].iloc[-1]).total_seconds() - 60.0

    # ── main step ────────────────────────────────────────────────────────
    def step(self) -> bool:
        """One iteration. Returns False when the bus is exhausted (replay)."""
        status = self.bus.status()
        if status is None:
            return False
        srv_now = pd.Timestamp(status["server_time"])
        b = self.bus.bars()
        if b is not None and len(b):
            self.hist.append(b)
        if (self.out / "HALT").exists() and not self.entries_halted:
            self._halt_entries("HALT file")
            for t in (self.s1, self.s2):
                if t and t.filled:
                    self._close_position(t, srv_now, "halt")
        self._process_reports(srv_now)
        self._reconcile_positions(status, srv_now)
        bars_fresh = self._bars_age(srv_now) <= STALE_BARS_S
        if "s1" in self.sleeves:
            self._check_s1_exits(srv_now, bars_fresh)
        if "s2" in self.sleeves:
            self._check_s2(srv_now, status, bars_fresh)
        if "s1" in self.sleeves and bars_fresh and not self.entries_halted:
            self._maybe_decide_s1(srv_now, status)
        elif "s1" in self.sleeves and not bars_fresh:
            # decisions while stale are MISSED, never chased
            last_ts = self.hist.bars["ts"].iloc[-1]
            bn = complete_bin(last_ts)
            if self.last_decision_ts is None or bn > self.last_decision_ts:
                self.last_decision_ts = bn
                self._dirty = True
                self._log_decision(bn, "", None, "skip", "stale_feed")
        self._save_state()
        return True

    def run(self, poll_s: float = 2.0):
        log.info(f"live loop {self.p.name} starting "
                 f"(dry_run={self.dry}, out={self.out})")
        eq_last = 0.0
        while True:
            try:
                alive = self.step()
                if not alive:
                    time.sleep(poll_s)
                    continue
                st = self.bus.status()
                if st and abs(float(st["equity"]) - eq_last) > 1e-9:
                    eq_last = float(st["equity"])
                    self._csv(self._equity_f, "logged,server_time,equity,balance",
                              f"{pd.Timestamp.now()},{st['server_time']},"
                              f"{st['equity']},{st.get('balance', '')}")
            except KeyboardInterrupt:
                log.info("interrupted — saving state")
                self._save_state()
                self.hist.save()
                return
            except Exception:
                log.exception("loop step failed — continuing")
            time.sleep(poll_s)
