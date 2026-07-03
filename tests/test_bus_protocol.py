"""FileBus protocol robustness: atomic order writes, torn-file tolerance,
report/deal parsing, staleness."""
import json
import os

import numpy as np
import pandas as pd

from daytrader.live.bus import DEAL_FIELDS, REPORT_FIELDS, FileBus


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


BARS = ("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2026.07.03\t10:00:00\t24000.5\t24010.0\t23990.0\t24005.0\t150\t0\t12\n"
        "2026.07.03\t10:01:00\t24005.0\t24015.0\t24000.0\t24010.0\t120\t0\t11\n")


def test_bars_parse_and_mtime_dedupe(tmp_path):
    bus = FileBus(tmp_path, login=62130224)
    _write(tmp_path / "bars.csv", BARS)
    df = bus.bars()
    assert len(df) == 2 and df["open"].dtype == np.float32
    assert df["ts"].iloc[0] == pd.Timestamp("2026-07-03 10:00:00")
    assert df["spread_pts"].iloc[1] == np.float32(11)
    assert bus.bars() is None                    # unchanged mtime → no re-read


def test_torn_bars_returns_none_not_crash(tmp_path):
    bus = FileBus(tmp_path, login=1)
    _write(tmp_path / "bars.csv", BARS + "2026.07.03\t10:02")   # torn line
    df = bus.bars()
    assert df is None or len(df) >= 2            # tolerated either way


def test_status_keeps_last_good(tmp_path):
    bus = FileBus(tmp_path, login=1)
    _write(tmp_path / "status.json", json.dumps({"equity": 50000.0,
                                                 "server_time": "x"}))
    assert bus.status()["equity"] == 50000.0
    _write(tmp_path / "status.json", '{"equity": 5')             # torn
    assert bus.status()["equity"] == 50000.0     # last good retained


def test_send_order_atomic_and_login_injected(tmp_path):
    bus = FileBus(tmp_path, login=62130224)
    bus.send_order({"id": "v2-000001", "action": "OPEN", "side": 1,
                    "lots": 1.0, "magic": 622001, "created_srv": "t"})
    files = os.listdir(tmp_path / "orders")
    assert files == ["done", "v2-000001.json"] or \
           sorted(files) == ["done", "v2-000001.json"]
    o = json.loads((tmp_path / "orders" / "v2-000001.json").read_text())
    assert o["login"] == 62130224                # defense-in-depth guard


def test_reports_since_idx_and_torn_tail(tmp_path):
    bus = FileBus(tmp_path, login=1)
    rows = [",".join(REPORT_FIELDS),
            "v2-000001,1,10009,12345,24001.5,2.5,23970.0,24090.0,2026-07-03 10:00:05,",
            "v2-000002,0,10019,,,,,,2026-07-03 10:05:05,no;money",
            "v2-000003,1,10009"]                 # torn tail
    _write(tmp_path / "reports.csv", "\n".join(rows) + "\n")
    reps = bus.reports(0)
    assert len(reps) == 2
    assert reps[0]["ok"] is True and reps[0]["ticket"] == 12345
    assert reps[0]["fill_price"] == 24001.5
    assert reps[1]["ok"] is False and reps[1]["ticket"] is None
    assert bus.reports(1)[0]["id"] == "v2-000002"
    assert bus.reports(2) == []


def test_deals_filter_by_ticket(tmp_path):
    bus = FileBus(tmp_path, login=1)
    rows = [",".join(DEAL_FIELDS),
            "111,2026-07-03 10:00:00,in,24001.5,2.5,0.0,0.0,0.0,622001,",
            "111,2026-07-03 11:00:00,out,24050.0,2.5,121.25,0.0,0.0,622001,tp",
            "222,2026-07-03 23:00:00,in,24100.0,1.0,0.0,0.0,0.0,622002,"]
    _write(tmp_path / "deals.csv", "\n".join(rows) + "\n")
    d = bus.deals(ticket=111)
    assert len(d) == 2 and d[1]["entry"] == "out" and d[1]["profit"] == 121.25
    assert len(bus.deals()) == 3
