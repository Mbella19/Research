"""FileBus — the Mac side of the file protocol (D-032).

Directory layout (inside the Wine prefix, written by the executor):
    <data_dir>/bars.csv        last N completed M1 bars, MT5 export format
    <data_dir>/bars_deep.csv   one-shot backfill (BACKFILL order)
    <data_dir>/status.json     server time, account, positions, symbol spec
    <data_dir>/deals.csv       our-magic deal history (7d rolling)
    <data_dir>/reports.csv     order execution reports (append-only)
    <data_dir>/orders/*.json   brain → executor (atomic tmp+rename)
    <data_dir>/orders/done/    processed order files

Atomicity: every file the executor writes is tmp+os.replace, and every file
the brain writes likewise — readers can still hit a mid-swap moment on the
OTHER side's caches, so every read tolerates parse failures by returning the
last good value (bars/status) or skipping torn tail lines (csv logs).
"""
import csv
import io
import json
import os
from pathlib import Path

import pandas as pd

from ..data.loader import load_mt5_csv
from ..utils.log import get_logger

log = get_logger("live.bus")

REPORT_FIELDS = ["id", "ok", "retcode", "ticket", "fill_price", "fill_volume",
                 "sl", "tp", "srv_time", "error"]
DEAL_FIELDS = ["position", "time", "entry", "price", "volume", "profit",
               "swap", "commission", "magic", "reason"]


class FileBus:
    def __init__(self, data_dir: Path, login: int):
        self.dir = Path(data_dir)
        self.login = int(login)
        self.orders = self.dir / "orders"
        self.orders.mkdir(parents=True, exist_ok=True)
        (self.orders / "done").mkdir(exist_ok=True)
        self._bars_mtime = 0.0
        self._bars_cache: pd.DataFrame | None = None
        self._status_cache: dict | None = None

    # ── inbound ──────────────────────────────────────────────────────────
    def status(self) -> dict | None:
        p = self.dir / "status.json"
        try:
            self._status_cache = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            pass                                   # keep last good
        return self._status_cache

    def bars(self) -> pd.DataFrame | None:
        p = self.dir / "bars.csv"
        try:
            m = p.stat().st_mtime
        except OSError:
            return None
        if m == self._bars_mtime:
            return None                            # unchanged since last read
        try:
            df = load_mt5_csv(p)
        except Exception as e:                     # torn/partial: retry later
            log.warning(f"bars.csv unreadable ({e}) — retrying next poll")
            return None
        self._bars_mtime = m
        self._bars_cache = df
        return df

    def bars_deep(self) -> pd.DataFrame | None:
        p = self.dir / "bars_deep.csv"
        if not p.exists():
            return None
        try:
            return load_mt5_csv(p)
        except Exception:
            return None

    def _read_csv_rows(self, path: Path, fields: list[str]) -> list[dict]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        rows = []
        rdr = csv.reader(io.StringIO(text))
        for i, row in enumerate(rdr):
            if i == 0 and row and row[0] == fields[0]:
                continue                           # header
            if len(row) != len(fields):
                continue                           # torn tail line
            rows.append(dict(zip(fields, row)))
        return rows

    def reports(self, since_idx: int) -> list[dict]:
        out = []
        for i, r in enumerate(self._read_csv_rows(self.dir / "reports.csv",
                                                  REPORT_FIELDS)):
            if i < since_idx:
                continue
            r["_idx"] = i
            r["ok"] = str(r["ok"]).lower() in ("1", "true", "ok")
            for k in ("ticket",):
                r[k] = int(r[k]) if str(r[k]).strip() else None
            for k in ("fill_price", "fill_volume", "sl", "tp"):
                r[k] = float(r[k]) if str(r[k]).strip() else None
            out.append(r)
        return out

    def deals(self, ticket: int | None = None) -> list[dict]:
        rows = self._read_csv_rows(self.dir / "deals.csv", DEAL_FIELDS)
        out = []
        for r in rows:
            r["position"] = int(r["position"]) if str(r["position"]).strip() else 0
            for k in ("price", "volume", "profit", "swap", "commission"):
                r[k] = float(r[k]) if str(r[k]).strip() else 0.0
            if ticket is None or r["position"] == ticket:
                out.append(r)
        return out

    # ── outbound ─────────────────────────────────────────────────────────
    def send_order(self, order: dict) -> str:
        order = {**order, "login": self.login}
        oid = order["id"]
        tmp = self.orders / f"{oid}.json.tmp"
        final = self.orders / f"{oid}.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(order, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        return oid

    def request_backfill(self, n: int, seq_id: str) -> None:
        self.send_order({"id": seq_id, "action": "BACKFILL", "n": int(n),
                         "created_srv": ""})

    # ── health ───────────────────────────────────────────────────────────
    def status_age_s(self) -> float:
        p = self.dir / "status.json"
        try:
            import time
            return time.time() - p.stat().st_mtime
        except OSError:
            return float("inf")
