"""Wine-side MT5 executor for the daytrader live bridge (D-032).

Runs under the SAME Wine Python 3.12 + MetaTrader5 package as the proven
tv-mt5-copier, one process per terminal:

    wine64 "C:\\Python312\\python.exe" mt5_executor_daytrader.py ^
        --config "C:\\daytrader\\accounts\\v2.json" [--once] [--dry]

It is a DUMB pipe: exports bars/status/deals, executes order files, applies
broker rounding to SL/TP anchored at the ACTUAL fill. All trading decisions
live on the Mac side. stdlib only (no numpy/pandas — Wine has pandas 3.x).

Attach-only battery (refuses rather than launching/logging in):
    terminal path+data_path match the portable install, login == login_guard,
    DEMO account, HEDGING margin mode, symbol visible.
"""
import calendar
import csv
import json
import os
import sys
import time

# ── pure helpers (unit-tested on the Mac side; no MT5 import needed) ──────
BAR_HEADER = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"
REPORT_FIELDS = ["id", "ok", "retcode", "ticket", "fill_price", "fill_volume",
                 "sl", "tp", "srv_time", "error"]
DEAL_FIELDS = ["position", "time", "entry", "price", "volume", "profit",
               "swap", "commission", "magic", "reason"]
FILLINGS = ["IOC", "FOK", "RETURN"]


def epoch_str(t: int) -> str:
    """MT5 server-tz epoch → 'YYYY.MM.DD<tab>HH:MM:SS' (broker wall time)."""
    st = time.gmtime(int(t))
    return (f"{st.tm_year:04d}.{st.tm_mon:02d}.{st.tm_mday:02d}\t"
            f"{st.tm_hour:02d}:{st.tm_min:02d}:{st.tm_sec:02d}")


def fmt_price(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def bar_line(rate, digits: int) -> str:
    """One MqlRates tuple → one export row (must round-trip load_mt5_csv)."""
    return (f"{epoch_str(rate['time'])}\t{fmt_price(rate['open'], digits)}\t"
            f"{fmt_price(rate['high'], digits)}\t{fmt_price(rate['low'], digits)}\t"
            f"{fmt_price(rate['close'], digits)}\t{int(rate['tick_volume'])}\t0\t"
            f"{int(rate['spread'])}")


def validate_order(o: dict, cfg: dict, done_ids: set, srv_now: float) -> str:
    """'' if valid, else the rejection reason."""
    if o.get("id") in done_ids:
        return "duplicate id"
    if int(o.get("login", -1)) != int(cfg["login_guard"]):
        return f"login mismatch {o.get('login')} != {cfg['login_guard']}"
    act = o.get("action")
    if act not in ("OPEN", "CLOSE", "FLATTEN", "BACKFILL"):
        return f"unknown action {act!r}"
    if act == "OPEN":
        if int(o.get("magic", 0)) not in cfg["magics"]:
            return f"magic {o.get('magic')} not ours"
        if not (0 < float(o.get("lots", 0)) <= float(cfg["max_lots"])):
            return f"lots {o.get('lots')} outside (0, {cfg['max_lots']}]"
        if int(o.get("side", 0)) not in (1, -1):
            return "side must be +1/-1"
    if act == "CLOSE" and not o.get("ticket"):
        return "CLOSE needs ticket"
    created = str(o.get("created_srv", ""))
    if act in ("OPEN", "CLOSE") and created:
        try:
            # MT5 epochs are server-wall-time-as-UTC; parse the same way so
            # the check is machine-timezone-independent
            t = calendar.timegm(time.strptime(created, "%Y-%m-%d %H:%M:%S"))
            if srv_now - t > 120.0:
                return f"stale order ({srv_now - t:.0f}s old)"
        except ValueError:
            return f"bad created_srv {created!r}"
    return ""


def next_filling(current: str) -> str | None:
    i = FILLINGS.index(current)
    return FILLINGS[i + 1] if i + 1 < len(FILLINGS) else None


def atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv(path: str, fields: list, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerow([row.get(k, "") for k in fields])


# ── executor proper (needs MetaTrader5; Wine only) ────────────────────────
def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Executor:
    def __init__(self, cfg: dict, dry: bool):
        import MetaTrader5 as mt5
        self.mt5 = mt5
        self.cfg = cfg
        self.dry = dry or bool(cfg.get("dry_run"))
        self.sym = cfg["symbol"]
        d = cfg["data_dir"]
        os.makedirs(os.path.join(d, "orders", "done"), exist_ok=True)
        self.f_bars = os.path.join(d, "bars.csv")
        self.f_deep = os.path.join(d, "bars_deep.csv")
        self.f_status = os.path.join(d, "status.json")
        self.f_deals = os.path.join(d, "deals.csv")
        self.f_reports = os.path.join(d, "reports.csv")
        self.d_orders = os.path.join(d, "orders")
        self.done_ids = self._load_done_ids()
        self.filling = "IOC"
        self.digits = 1
        self.last_bar_t = 0
        self.last_deals_t = 0.0

    def _load_done_ids(self) -> set:
        ids = set()
        if os.path.exists(self.f_reports):
            with open(self.f_reports, newline="") as f:
                for row in csv.DictReader(f):
                    ids.add(row.get("id"))
        return ids

    # ── connection ────────────────────────────────────────────────────────
    def connect(self) -> bool:
        mt5 = self.mt5
        if not mt5.initialize(self.cfg["terminal_path"], portable=True,
                              timeout=30000):
            log(f"initialize failed: {mt5.last_error()}")
            return False
        ti, ai = mt5.terminal_info(), mt5.account_info()
        if ti is None or ai is None:
            log("no terminal/account info")
            return False
        exp = self.cfg["expected_dir"].lower().rstrip("\\")
        if not (ti.path.lower().rstrip("\\") == exp
                and ti.data_path.lower().rstrip("\\") == exp):
            log(f"REFUSE: terminal path {ti.path!r} / data {ti.data_path!r} "
                f"!= expected portable dir {exp!r}")
            return False
        if int(ai.login) != int(self.cfg["login_guard"]):
            log(f"REFUSE: logged-in account {ai.login} != guard "
                f"{self.cfg['login_guard']}")
            return False
        if ai.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            log("REFUSE: not a DEMO account")
            return False
        if ai.margin_mode != mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
            log("REFUSE: account is NETTING — parity needs hedging")
            return False
        if not mt5.symbol_select(self.sym, True):
            log(f"REFUSE: cannot select symbol {self.sym}")
            return False
        si = mt5.symbol_info(self.sym)
        self.digits = int(si.digits)
        log(f"attached: {ti.path} login {ai.login} ({ai.server}) "
            f"{self.sym} digits={si.digits} point={si.point} "
            f"stops_level={si.trade_stops_level} dry={self.dry}")
        return True

    # ── exports ───────────────────────────────────────────────────────────
    def write_bars(self):
        mt5 = self.mt5
        rates = mt5.copy_rates_from_pos(self.sym, mt5.TIMEFRAME_M1, 1,
                                        int(self.cfg.get("bars_n", 3000)))
        if rates is None or not len(rates):
            return
        t = int(rates[-1]["time"])
        if t == self.last_bar_t:
            return
        self.last_bar_t = t
        lines = [BAR_HEADER]
        lines += [bar_line(r, self.digits) for r in rates]
        atomic_write(self.f_bars, "\n".join(lines) + "\n")

    def write_backfill(self, n: int):
        mt5 = self.mt5
        rates = mt5.copy_rates_from_pos(self.sym, mt5.TIMEFRAME_M1, 1, int(n))
        if rates is None:
            return 0
        lines = [BAR_HEADER] + [bar_line(r, self.digits) for r in rates]
        atomic_write(self.f_deep, "\n".join(lines) + "\n")
        return len(rates)

    def write_status(self) -> bool:
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.sym)
        ai = mt5.account_info()
        ti = mt5.terminal_info()
        si = mt5.symbol_info(self.sym)
        if not (tick and ai and si):
            return False
        poss = mt5.positions_get(symbol=self.sym) or ()
        ours = [p for p in poss if p.magic in self.cfg["magics"]]
        st = {
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.gmtime(int(tick.time))),
            "equity": ai.equity, "balance": ai.balance,
            "margin_free": ai.margin_free, "leverage": ai.leverage,
            "login": ai.login, "currency": ai.currency,
            "trade_allowed": bool(ti.trade_allowed),
            "positions": [{
                "ticket": p.ticket, "magic": p.magic,
                "type": 1 if p.type == mt5.POSITION_TYPE_BUY else -1,
                "volume": p.volume, "price_open": p.price_open,
                "sl": p.sl or None, "tp": p.tp or None,
                "time": time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.gmtime(int(p.time))),
            } for p in ours],
            "symbol": {
                "name": self.sym, "point": si.point, "digits": si.digits,
                "volume_step": si.volume_step, "volume_min": si.volume_min,
                "volume_max": si.volume_max,
                "stops_level": si.trade_stops_level,
                "freeze_level": si.trade_freeze_level,
                "tick_value": si.trade_tick_value,
                "tick_size": si.trade_tick_size,
                "spread": si.spread,
            },
        }
        atomic_write(self.f_status, json.dumps(st))
        return True

    def write_deals(self):
        mt5 = self.mt5
        now = time.time()
        if now - self.last_deals_t < 5.0:
            return
        self.last_deals_t = now
        frm = now - 7 * 86400
        deals = mt5.history_deals_get(frm, now + 86400)
        if deals is None:
            # half-dead session: never clobber the last good deals file —
            # the brain finalizes closed trades from it
            return
        rows = []
        for d in deals:
            if d.magic not in self.cfg["magics"] or d.symbol != self.sym:
                continue
            rows.append({
                "position": d.position_id,
                "time": time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.gmtime(int(d.time))),
                "entry": "in" if d.entry == mt5.DEAL_ENTRY_IN else "out",
                "price": d.price, "volume": d.volume, "profit": d.profit,
                "swap": d.swap, "commission": d.commission,
                "magic": d.magic, "reason": self._deal_reason(d),
            })
        buf = [",".join(DEAL_FIELDS)]
        for r in rows:
            buf.append(",".join(str(r[k]) for k in DEAL_FIELDS))
        atomic_write(self.f_deals, "\n".join(buf) + "\n")

    def _deal_reason(self, d) -> str:
        mt5 = self.mt5
        return {mt5.DEAL_REASON_SL: "sl", mt5.DEAL_REASON_TP: "tp"}.get(
            d.reason, str(d.comment or "").replace(",", ";"))

    # ── order execution ───────────────────────────────────────────────────
    def _report(self, o: dict, ok: bool, retcode="", ticket="", fill_price="",
                fill_volume="", sl="", tp="", error=""):
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.sym)
        srv = (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(tick.time)))
               if tick else "")
        append_csv(self.f_reports, REPORT_FIELDS, {
            "id": o.get("id"), "ok": int(ok), "retcode": retcode,
            "ticket": ticket, "fill_price": fill_price,
            "fill_volume": fill_volume, "sl": sl, "tp": tp,
            "srv_time": srv, "error": str(error).replace(",", ";")})
        self.done_ids.add(o.get("id"))

    def _fill_const(self, name: str):
        mt5 = self.mt5
        return {"IOC": mt5.ORDER_FILLING_IOC, "FOK": mt5.ORDER_FILLING_FOK,
                "RETURN": mt5.ORDER_FILLING_RETURN}[name]

    def _send_deal(self, req: dict):
        """order_send with reconnect-once + filling ladder on 10030."""
        mt5 = self.mt5
        for attempt in range(4):
            req["type_filling"] = self._fill_const(self.filling)
            res = mt5.order_send(req)
            if res is None:
                log(f"order_send None ({mt5.last_error()}) — reconnect")
                self.connect()
                continue
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                return res
            if res.retcode == 10030:                     # unsupported filling
                nxt = next_filling(self.filling)
                if nxt:
                    log(f"filling {self.filling} unsupported → {nxt}")
                    self.filling = nxt
                    continue
            if res.retcode in (10004, 10021):            # requote/price off
                tick = mt5.symbol_info_tick(self.sym)
                if tick:
                    req["price"] = (tick.ask if req["type"] ==
                                    mt5.ORDER_TYPE_BUY else tick.bid)
                continue
            return res
        return res

    def do_open(self, o: dict):
        mt5 = self.mt5
        if self.dry:
            self._report(o, ok=False, error="dry_run")
            return
        side = int(o["side"])
        tick = mt5.symbol_info_tick(self.sym)
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.sym,
            "volume": float(o["lots"]),
            "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if side > 0 else tick.bid,
            "deviation": int(self.cfg.get("deviation_points", 50)),
            "magic": int(o["magic"]),
            "comment": str(o.get("comment", ""))[:26],
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = self._send_deal(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            self._report(o, ok=False,
                         retcode=getattr(res, "retcode", ""),
                         error=getattr(res, "comment", "send failed"))
            return
        fill = float(res.price)
        vol = float(res.volume)
        ticket = int(res.order)               # hedging: position id == order
        sl = tp = 0.0
        if o.get("sl_dist"):
            sl = round(fill - side * float(o["sl_dist"]), self.digits)
        if o.get("tp_dist"):
            tp = round(fill + side * float(o["tp_dist"]), self.digits)
        if sl or tp:
            mod = {"action": mt5.TRADE_ACTION_SLTP, "symbol": self.sym,
                   "position": ticket, "sl": sl, "tp": tp}
            r2 = mt5.order_send(mod)
            if r2 is None or r2.retcode != mt5.TRADE_RETCODE_DONE:
                log(f"SLTP set failed ({getattr(r2, 'retcode', None)}) — "
                    f"retrying once")
                time.sleep(0.5)
                r2 = mt5.order_send(mod)
                if r2 is None or r2.retcode != mt5.TRADE_RETCODE_DONE:
                    # protective failure: refuse to hold an unprotected
                    # position the policy priced with a stop
                    self._close_ticket(ticket, vol, "sltp_failed")
                    self._report(o, ok=False, retcode=getattr(r2, "retcode", ""),
                                 error="SLTP set failed; position closed")
                    return
        self._report(o, ok=True, retcode=res.retcode, ticket=ticket,
                     fill_price=fill, fill_volume=vol,
                     sl=sl or "", tp=tp or "")
        log(f"OPEN ok id={o['id']} ticket={ticket} fill={fill} "
            f"vol={vol} sl={sl} tp={tp}")

    def _close_ticket(self, ticket: int, vol: float, comment: str):
        mt5 = self.mt5
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return None
        p = pos[0]
        tick = mt5.symbol_info_tick(self.sym)
        buy = p.type == mt5.POSITION_TYPE_BUY
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.sym,
               "volume": float(vol or p.volume), "position": ticket,
               "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY,
               "price": tick.bid if buy else tick.ask,
               "deviation": int(self.cfg.get("deviation_points", 50)),
               "magic": p.magic, "comment": comment[:26],
               "type_time": mt5.ORDER_TIME_GTC}
        return self._send_deal(req)

    def do_close(self, o: dict):
        mt5 = self.mt5
        if self.dry:
            self._report(o, ok=False, error="dry_run")
            return
        ticket = int(o["ticket"])
        if not mt5.positions_get(ticket=ticket):
            self._report(o, ok=False, error="position not found")
            return
        res = self._close_ticket(ticket, float(o.get("lots", 0)),
                                 str(o.get("comment", "close")))
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            self._report(o, ok=False, retcode=getattr(res, "retcode", ""),
                         error=getattr(res, "comment", "close failed"))
            return
        self._report(o, ok=True, retcode=res.retcode, ticket=ticket,
                     fill_price=float(res.price), fill_volume=float(res.volume))
        log(f"CLOSE ok id={o['id']} ticket={ticket} fill={res.price}")

    def do_flatten(self, o: dict):
        mt5 = self.mt5
        if self.dry:
            self._report(o, ok=False, error="dry_run")
            return
        poss = mt5.positions_get(symbol=self.sym) or ()
        n = 0
        for p in poss:
            if p.magic in self.cfg["magics"]:
                self._close_ticket(p.ticket, p.volume, "flatten")
                n += 1
        self._report(o, ok=True, error=f"closed {n}")

    def poll_orders(self):
        mt5 = self.mt5
        try:
            names = sorted(f for f in os.listdir(self.d_orders)
                           if f.endswith(".json"))
        except OSError:
            return
        for name in names:
            path = os.path.join(self.d_orders, name)
            try:
                o = json.load(open(path, encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue                              # mid-write; next poll
            tick = mt5.symbol_info_tick(self.sym)
            srv_now = float(tick.time) if tick else time.time()
            err = validate_order(o, self.cfg, self.done_ids, srv_now)
            if err:
                log(f"order {o.get('id')} rejected: {err}")
                self._report(o, ok=False, error=err)
            elif o["action"] == "OPEN":
                self.do_open(o)
            elif o["action"] == "CLOSE":
                self.do_close(o)
            elif o["action"] == "FLATTEN":
                self.do_flatten(o)
            elif o["action"] == "BACKFILL":
                n = self.write_backfill(int(o.get("n", 60000)))
                self._report(o, ok=True, error=f"backfill {n}")
            os.replace(path, os.path.join(self.d_orders, "done", name))

    # ── main ──────────────────────────────────────────────────────────────
    def run(self, once: bool = False):
        if not self.connect():
            sys.exit(2)
        poll = float(self.cfg.get("poll_ms", 750)) / 1000.0
        log("build 2026-07-03b: auto-reattach on silent-None API streak")
        dead = 0
        last_attach = 0.0
        while True:
            try:
                self.write_bars()
                dead = 0 if self.write_status() else dead + 1
                self.write_deals()
                self.poll_orders()
                if dead >= 40 and time.time() - last_attach > 60:
                    # after a server drop the old IPC session can return
                    # None forever while a fresh attach works fine
                    log(f"status calls None x{dead} — reattaching session")
                    last_attach = time.time()
                    try:
                        self.mt5.shutdown()
                    except Exception:
                        pass
                    if self.connect():
                        dead = 0
            except Exception as e:
                log(f"loop error: {e!r}")
                try:
                    self.connect()
                except Exception:
                    pass
            if once:
                return
            time.sleep(poll)


def main():
    i = sys.argv.index("--config")
    cfg = json.load(open(sys.argv[i + 1], encoding="utf-8"))
    Executor(cfg, dry=("--dry" in sys.argv)).run(once=("--once" in sys.argv))


if __name__ == "__main__":
    main()
