"""Executor pure helpers (no MetaTrader5 import): bar formatting must
round-trip load_mt5_csv bit-exactly; order validation matrix; filling
ladder. The executor module is loaded from bridge/ by path."""
import importlib.util
import time
from pathlib import Path

import numpy as np

from daytrader.data.loader import load_mt5_csv
from daytrader.utils import paths

spec = importlib.util.spec_from_file_location(
    "mt5_executor_daytrader",
    paths.PROJECT_ROOT / "bridge" / "mt5_executor_daytrader.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

CFG = {"login_guard": 62130224, "magics": [622001, 622002], "max_lots": 15.0}


def test_bar_line_roundtrips_loader(tmp_path):
    rates = [
        {"time": 1782033600, "open": 24000.5, "high": 24010.0, "low": 23990.5,
         "close": 24005.1, "tick_volume": 150, "spread": 12},
        {"time": 1782033660, "open": 24005.1, "high": 24015.9, "low": 24000.0,
         "close": 24010.4, "tick_volume": 121, "spread": 11},
    ]
    lines = [ex.BAR_HEADER] + [ex.bar_line(r, digits=1) for r in rates]
    p = tmp_path / "bars.csv"
    p.write_text("\n".join(lines) + "\n")
    df = load_mt5_csv(p)
    assert len(df) == 2
    assert df["open"].dtype == np.float32
    assert df["open"].iloc[0] == np.float32(24000.5)
    assert df["high"].iloc[1] == np.float32(24015.9)
    assert df["spread_pts"].iloc[0] == np.float32(12)
    assert int(df["tickvol"].iloc[0]) == 150
    # epoch → broker wall time via gmtime (MT5 epochs are server-tz)
    assert str(df["ts"].iloc[1] - df["ts"].iloc[0]) == "0 days 00:01:00"


def test_validate_order_matrix():
    import calendar
    now = calendar.timegm(time.strptime("2026-07-03 10:00:00",
                                        "%Y-%m-%d %H:%M:%S"))
    ok = {"id": "a1", "login": 62130224, "action": "OPEN", "magic": 622001,
          "side": 1, "lots": 2.0, "created_srv": "2026-07-03 09:59:30"}
    assert ex.validate_order(ok, CFG, set(), now) == ""
    assert "duplicate" in ex.validate_order(ok, CFG, {"a1"}, now)
    assert "login" in ex.validate_order({**ok, "login": 999}, CFG, set(), now)
    assert "magic" in ex.validate_order({**ok, "magic": 990012}, CFG, set(), now)
    assert "lots" in ex.validate_order({**ok, "lots": 99.0}, CFG, set(), now)
    assert "lots" in ex.validate_order({**ok, "lots": 0}, CFG, set(), now)
    assert "side" in ex.validate_order({**ok, "side": 2}, CFG, set(), now)
    assert "unknown action" in ex.validate_order({**ok, "action": "NUKE"},
                                                 CFG, set(), now)
    assert "stale" in ex.validate_order(
        {**ok, "created_srv": "2026-07-03 09:00:00"}, CFG, set(), now)
    assert "ticket" in ex.validate_order(
        {"id": "c1", "login": 62130224, "action": "CLOSE",
         "created_srv": "2026-07-03 09:59:59"}, CFG, set(), now)


def test_filling_ladder():
    assert ex.next_filling("IOC") == "FOK"
    assert ex.next_filling("FOK") == "RETURN"
    assert ex.next_filling("RETURN") is None


def test_atomic_write_no_partial(tmp_path):
    p = tmp_path / "x.json"
    ex.atomic_write(str(p), "A" * 100)
    assert p.read_text() == "A" * 100
    assert not (tmp_path / "x.json.tmp").exists()
