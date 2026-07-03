"""Replay harness — the D-032 parity gate.

SimBus drives the REAL LiveLoop bar-by-bar over historical 1m data and fills
its orders with the ENGINE fill model (engine.py:169-226 / overnight.py:
167-182 semantics), so the replayed trade list is directly comparable to
`run_backtest` / `sleeve2_run` on the same window:

  entry:  market orders queue and fill at the NEXT completing bar's open
          (long: o+spread+slip, short: o−slip) — the engine's first-1m-bar-
          at/after-avail fill;
  SL/TP:  scanned per bar AFTER queued fills (a close due at the bar's open
          leaves nothing for the intrabar stop — matches the engine's
          exclusive exit-bar scan); long barriers on bid, short on ask;
          same-bar SL+TP → SL; gap-through = worse of (level, open) ∓ slip;
          TP is a limit, fills exactly at the level;
  deals:  MT5 sign conventions (gross profit; commission negative on the
          closing deal; swap 0 — the frozen policies price no swap).

ReplayHistory exposes the full pre-seeded history through a moving cursor so
the loop sees exactly what it would have seen live (no re-concat per bar).
"""
import numpy as np
import pandas as pd

from ..backtest.engine import CostCfg, make_cost_cfg
from ..data.resample import resample_bars
from ..features.daily_context import day_aggs
from ..portfolio.overnight import daily_frame
from ..utils.log import get_logger

log = get_logger("live.replay")


class ReplayHistory:
    """LiveHistory-compatible view over a fully pre-seeded frame + cursor."""

    def __init__(self, all_bars: pd.DataFrame, start_i: int):
        self._all = all_bars.reset_index(drop=True)
        self._ts = self._all["ts"].to_numpy()
        self.cursor = int(start_i)
        self._dohlc_full = daily_frame(self._all)     # per-day, causal via cursor
        self._aggs_full = day_aggs(resample_bars(self._all, 5))

    @property
    def bars(self) -> pd.DataFrame:
        return self._all.iloc[:self.cursor]

    def append(self, df_new):                          # bus delivers via cursor
        return {"n_new": 0, "revised": 0, "gap_s": 0.0}

    def save(self):
        pass

    def _upto_day(self, table: pd.DataFrame) -> pd.DataFrame:
        """Full-history daily tables clipped causally: completed days fully,
        plus TODAY's partial row rebuilt from bars ≤ cursor."""
        now_day = pd.Timestamp(self._ts[self.cursor - 1]).normalize()
        head = table[table.index < now_day]
        i0 = int(np.searchsorted(self._ts, np.datetime64(now_day)))
        part = self._all.iloc[i0:self.cursor]
        if len(part):
            if "absmove" in table.columns:
                tail = day_aggs(resample_bars(part, 5))
            else:
                tail = daily_frame(part)
            return pd.concat([head, tail])
        return head

    def daily_ohlc(self) -> pd.DataFrame:
        return self._upto_day(self._dohlc_full)

    def feat_daily_ctx(self) -> pd.DataFrame:
        return self._upto_day(self._aggs_full)


class SimBus:
    """Engine-semantics broker + bar feed for LiveLoop."""

    def __init__(self, hist: ReplayHistory, cost: CostCfg,
                 equity0: float = 10_000.0, end_i: int | None = None,
                 vpu: float = 1.0):
        self.h = hist
        self.cost = cost
        self.vpu = vpu
        df = hist._all
        self.ts = hist._ts
        self.o = df["open"].to_numpy(np.float64)
        self.hi = df["high"].to_numpy(np.float64)
        self.lo = df["low"].to_numpy(np.float64)
        self.c = df["close"].to_numpy(np.float64)
        self.sp = (np.maximum(df["spread_pts"].to_numpy(np.float64),
                              cost.spread_floor_pts)
                   * cost.spread_mult * cost.point_size)
        self.slip = cost.slippage_pts * cost.point_size
        self.end_i = int(end_i if end_i is not None else len(df))
        self.equity = float(equity0)
        self.queued: list[dict] = []       # accepted orders awaiting next bar
        self.positions: dict[int, dict] = {}
        self._reports: list[dict] = []
        self._deals: list[dict] = []
        self._ticket_seq = 1000
        self._done = False
        self._pre_j: int | None = None

    # ── bus interface (consumed by LiveLoop) ─────────────────────────────
    def status(self):
        if self._done:
            return None
        if self._pre_j is not None:        # bar _pre_j opening now
            srv = str(pd.Timestamp(self.ts[self._pre_j]))
        else:
            srv = str(pd.Timestamp(self.ts[self.h.cursor - 1])
                      + pd.Timedelta(minutes=1))
        return {
            "server_time": srv,
            "equity": self.equity, "balance": self.equity,
            "margin_free": self.equity, "leverage": 100.0,
            "positions": [
                {"ticket": t, "magic": p["magic"], "type": p["side"],
                 "volume": p["lots"], "price_open": p["entry"],
                 "sl": p.get("sl"), "tp": p.get("tp"), "time": str(p["ts_open"])}
                for t, p in self.positions.items()],
        }

    def bars(self):
        return None                        # history advances via the cursor

    def send_order(self, order: dict) -> str:
        j = self._pre_j if self._pre_j is not None else self.h.cursor - 1
        day_created = pd.Timestamp(self.ts[j]).normalize()
        self.queued.append({**order, "_day": day_created})
        return order["id"]

    def reports(self, since_idx: int):
        out = []
        for i in range(since_idx, len(self._reports)):
            out.append({**self._reports[i], "_idx": i})
        return out

    def deals(self, ticket: int | None = None):
        return [d for d in self._deals
                if ticket is None or d["position"] == ticket]

    # ── simulation stepping (two-phase clock) ────────────────────────────
    # pre_open(j): server time = ts[j] (bar j is opening NOW; not delivered).
    #   Time-triggered decisions (23:00 S2, 23:30 flat, 240-bar horizon) fire
    #   here so their market orders fill at o[j] — the engine's exit/entry
    #   bar open. deliver(j): fill the queue at o[j], run the intrabar SL/TP
    #   scan, hand bar j to the history.
    def pre_open(self) -> bool:
        if self.h.cursor >= self.end_i:
            self._done = True
            return False
        self._pre_j = self.h.cursor
        return True

    def deliver(self) -> None:
        j = self.h.cursor
        self.h.cursor += 1
        self._pre_j = None
        bar_day = pd.Timestamp(self.ts[j]).normalize()
        for q in list(self.queued):
            self.queued.remove(q)
            if q["_day"] != bar_day:
                if q["action"] == "OPEN":  # engine: entry bar must share the
                    self._report(q, ok=False,   # signal's day (engine.py:142)
                                 error="expired: next bar on a different day")
                    continue
            if q["action"] == "OPEN":
                self._fill_open(q, j)
            elif q["action"] == "CLOSE":
                self._fill_close(q, j)
        self._scan_barriers(j)

    def _report(self, order, ok, **kw):
        self._reports.append({"id": order["id"], "ok": ok,
                              "srv_time": str(self.ts[order.get('_fill_j', 0)])
                              if ok else str(self.ts[self.h.cursor - 1]), **kw})

    def _fill_open(self, q, j):
        side = int(q["side"])
        if side > 0:
            entry = self.o[j] + self.sp[j] + self.slip     # buy at ask
        else:
            entry = self.o[j] - self.slip                  # sell at bid
        sl = tp = None                       # engine-exact levels, unrounded
        if q.get("sl_dist"):
            sl = entry - side * float(q["sl_dist"])
        if q.get("tp_dist"):
            tp = entry + side * float(q["tp_dist"])
        t = self._ticket_seq = self._ticket_seq + 1
        self.positions[t] = {"magic": q["magic"], "side": side,
                             "lots": float(q["lots"]), "entry": entry,
                             "sl": sl, "tp": tp, "ts_open": self.ts[j],
                             "j_open": j}
        self._deals.append({"position": t, "time": str(self.ts[j]),
                            "entry": "in", "price": entry,
                            "volume": q["lots"], "profit": 0.0, "swap": 0.0,
                            "commission": -self.cost.commission_per_lot_side
                                          * float(q["lots"]),
                            "magic": q["magic"], "reason": ""})
        q["_fill_j"] = j
        self._report(q, ok=True, ticket=t, fill_price=entry,
                     fill_volume=q["lots"], sl=sl, tp=tp)

    def _fill_close(self, q, j, reason="close"):
        t = int(q["ticket"])
        p = self.positions.pop(t, None)
        if p is None:
            self._report(q, ok=False, error="position not found")
            return
        if p["side"] > 0:
            exit_px = self.o[j] - self.slip                # sell at bid
        else:
            exit_px = self.o[j] + self.sp[j] + self.slip   # cover at ask
        self._book_out(t, p, exit_px, j, q.get("comment", reason))
        q["_fill_j"] = j
        self._report(q, ok=True, ticket=t, fill_price=exit_px,
                     fill_volume=p["lots"], sl=p["sl"], tp=p["tp"])

    def _book_out(self, ticket, p, exit_px, j, reason):
        gross = (exit_px - p["entry"]) * p["side"] * p["lots"] * self.vpu
        comm = -self.cost.commission_per_lot_side * p["lots"]
        self._deals.append({"position": ticket, "time": str(self.ts[j]),
                            "entry": "out", "price": exit_px,
                            "volume": p["lots"], "profit": gross,
                            "swap": 0.0, "commission": comm,
                            "magic": p["magic"], "reason": reason})
        # closed-PnL-only compounding (engine.py:151/230)
        self.equity += gross + comm + self._in_comm(ticket)

    def _in_comm(self, ticket):
        return sum(d["commission"] for d in self._deals
                   if d["position"] == ticket and d["entry"] == "in")

    def _scan_barriers(self, j):
        """Engine barrier semantics on bar j (engine.py:175-200)."""
        for t, p in list(self.positions.items()):
            s, sl_lvl, tp_lvl = p["side"], p["sl"], p["tp"]
            if s > 0:
                sl_hit = sl_lvl is not None and self.lo[j] <= sl_lvl
                tp_hit = tp_lvl is not None and self.hi[j] >= tp_lvl
            else:
                ask_h = self.hi[j] + self.sp[j]
                ask_l = self.lo[j] + self.sp[j]
                sl_hit = sl_lvl is not None and ask_h >= sl_lvl
                tp_hit = tp_lvl is not None and ask_l <= tp_lvl
            if sl_hit:                                     # same-bar → SL
                if s > 0:
                    lvl = min(sl_lvl, self.o[j]) if self.o[j] < sl_lvl else sl_lvl
                    exit_px = lvl - self.slip
                else:
                    ask_open = self.o[j] + self.sp[j]
                    lvl = max(sl_lvl, ask_open) if ask_open > sl_lvl else sl_lvl
                    exit_px = lvl + self.slip
                self.positions.pop(t)
                self._book_out(t, p, exit_px, j, "sl")
            elif tp_hit:
                self.positions.pop(t)
                self._book_out(t, p, tp_lvl, j, "tp")


def run_replay(policy_name: str, start: str, end: str | None = None,
               sleeves: tuple = ("s1",), out_dir=None, new_csv=None,
               equity0: float = 10_000.0) -> dict:
    """Drive the REAL LiveLoop over history with SimBus fills. Returns the
    replayed trades plus everything the parity comparator needs."""
    import tempfile
    from pathlib import Path

    from ..data.loader import load_mt5_csv
    from ..features.registry import build_features_from_1m
    from .history import LiveHistory, stored_real_frames
    from .loop import LiveLoop
    from .policy import PolicyRuntime

    p = PolicyRuntime.load(policy_name)
    frames = stored_real_frames()
    if new_csv is not None and Path(new_csv).exists():
        frames.append(load_mt5_csv(Path(new_csv)))
    store = LiveHistory("replayseed", root=Path(tempfile.mkdtemp()))
    store.seed(frames)
    allb = store.bars
    if end is not None:
        allb = allb[allb["ts"] <= pd.Timestamp(end)].reset_index(drop=True)
    start_i = int(np.searchsorted(allb["ts"].to_numpy(),
                                  np.datetime64(pd.Timestamp(start))))
    sig = feat = None
    if "s1" in sleeves:
        feat = build_features_from_1m(allb, p.groups)
        sig = p.s1_signals(feat)
    hist = ReplayHistory(allb, start_i)
    bus = SimBus(hist, make_cost_cfg(), equity0=equity0, vpu=p.vpu)
    out = (Path(out_dir) if out_dir
           else Path(tempfile.mkdtemp()) / f"replay_{policy_name}")
    loop = LiveLoop(p, hist, bus, out_dir=out, sim_signals=sig,
                    sleeves=sleeves)
    steps = 0
    while bus.pre_open():
        loop.step()
        bus.deliver()
        loop.step()
        steps += 1
    trades = (pd.read_csv(out / "trades.csv")
              if (out / "trades.csv").exists() else pd.DataFrame())
    log.info(f"replay {policy_name} {sleeves}: {steps:,} bars, "
             f"{len(trades)} trades, equity {bus.equity:,.2f}")
    return {"trades": trades, "out": out, "steps": steps, "signals": sig,
            "bars": allb, "start_i": start_i, "policy": p, "bus": bus}
