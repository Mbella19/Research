"""LiveHistory store: seed / append / revision handling, and the incremental
daily-context caches must always equal a from-scratch rebuild."""
import numpy as np
import pandas as pd

from daytrader.data.resample import resample_bars
from daytrader.features.daily_context import day_aggs
from daytrader.live.history import LiveHistory
from daytrader.portfolio.overnight import daily_frame


def _mk_1m(n_days=30, bars_per_day=40, seed=3, start="2026-01-05"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    rows, px = [], 500.0
    for d in days:
        for k in range(bars_per_day):
            ts = d + pd.Timedelta(minutes=60 + k)
            o = px
            c = px + rng.normal(0, 0.5)
            h, l = max(o, c) + 0.2, min(o, c) - 0.2
            rows.append((ts, o, h, l, c, 25, 10.0))
            px = c
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "tickvol", "spread_pts"])
    for col in ("open", "high", "low", "close", "spread_pts"):
        df[col] = df[col].astype(np.float32)
    return df


def _ctx_matches_scratch(h: LiveHistory):
    pd.testing.assert_frame_equal(
        h.feat_daily_ctx(), day_aggs(resample_bars(h.bars, 5)))
    pd.testing.assert_frame_equal(h.daily_ohlc(), daily_frame(h.bars))


def test_seed_and_ctx(tmp_path):
    df = _mk_1m()
    n = len(df) // 2
    h = LiveHistory("t", root=tmp_path)
    rep = h.seed([df.iloc[:n], df.iloc[n:]])
    assert rep["rows"] == len(df)
    assert h.bars["ts"].is_monotonic_increasing
    _ctx_matches_scratch(h)


def test_seed_overlap_keeps_later(tmp_path):
    df = _mk_1m()
    a, b = df.iloc[:700].copy(), df.iloc[650:].copy()
    b_orig_close = float(b["close"].iloc[0])
    a.loc[a.index[650:], "close"] = 99.0               # earlier frame is stale
    h = LiveHistory("t", root=tmp_path)
    h.seed([a, b])
    assert len(h.bars) == len(df)
    row = h.bars[h.bars["ts"] == b["ts"].iloc[0]]
    assert float(row["close"].iloc[0]) == b_orig_close


def test_append_new_and_identical_overlap(tmp_path):
    df = _mk_1m()
    h = LiveHistory("t", root=tmp_path)
    h.seed([df.iloc[:-20]])
    chunk = df.iloc[-60:]                              # 40 overlap + 20 new
    rep = h.append(chunk)
    assert rep == {"n_new": 20, "revised": 0, "gap_s": 60.0}
    assert len(h.bars) == len(df)
    _ctx_matches_scratch(h)


def test_append_revision_adopts_terminal(tmp_path):
    df = _mk_1m()
    h = LiveHistory("t", root=tmp_path)
    h.seed([df])
    h.feat_daily_ctx()                                 # warm the caches
    chunk = df.iloc[-50:].copy()
    ts_rev = chunk["ts"].iloc[5]
    chunk.loc[chunk.index[5], "close"] = np.float32(777.0)
    rep = h.append(chunk)
    assert rep["n_new"] == 0 and rep["revised"] == 1
    assert float(h.bars.loc[h.bars["ts"] == ts_rev, "close"].iloc[0]) == 777.0
    _ctx_matches_scratch(h)                            # dirty-day invalidation


def test_append_gap_reported_not_fatal(tmp_path):
    df = _mk_1m()
    h = LiveHistory("t", root=tmp_path)
    h.seed([df.iloc[:500]])
    rep = h.append(df.iloc[600:700])
    assert rep["n_new"] == 100 and rep["gap_s"] > 60
    assert h.bars["ts"].is_monotonic_increasing
    _ctx_matches_scratch(h)


def test_save_load_roundtrip(tmp_path):
    df = _mk_1m()
    h = LiveHistory("t", root=tmp_path)
    h.seed([df])
    h.save()
    h2 = LiveHistory("t", root=tmp_path)
    assert h2.load()
    pd.testing.assert_frame_equal(h.bars, h2.bars)
    _ctx_matches_scratch(h2)
