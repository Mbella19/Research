"""Port-correctness tests on handcrafted series (MS.txt + zigzag semantics)."""
import numpy as np
import pandas as pd

from daytrader.features.market_structure import run_state_machine
from daytrader.features.zigzag import PIV_RIGHT, compute as zz_compute


def test_ms_state_machine_lifecycle():
    n = 30
    o = np.full(n, 100.0)
    c = np.where(np.arange(n) % 2 == 0, 100.5, 99.5)  # alternate bull/bear candles
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)

    # bar 25: higher high → trend bullish, bear zone from last bearish candle (23)
    o[25], h[25], l[25], c[25] = 100.0, 106.0, 99.8, 105.0
    # bar 26: close below bear-zone bottom (99.0) → bearish CHoCH, bull zone from 25
    o[26], h[26], l[26], c[26] = 105.0, 105.5, 98.0, 98.5
    # bar 27: lower low but zone candle unchanged rule — bullish candle at new low
    o[27], h[27], l[27], c[27] = 98.0, 98.6, 97.0, 98.5
    # bar 28: fresh lower low → BOS bear, bull zone refreshes from candle 27
    o[28], h[28], l[28], c[28] = 98.4, 98.5, 96.5, 96.6
    # bar 29: bullish close above bull-zone top (98.6) → bullish CHoCH
    o[29], h[29], l[29], c[29] = 98.0, 99.5, 97.9, 99.0

    st = run_state_machine(o, h, l, c)
    assert st["ms_trend"][24] == 0
    assert st["ms_trend"][25] == 1
    assert st["ms_bear_top"][25] == 99.5 and st["ms_bear_bot"][25] == 99.0  # candle 23
    assert st["ms_choch_bear"][26] == 1 and st["ms_trend"][26] == -1
    assert st["ms_bull_top"][26] == 106.0 and st["ms_bull_bot"][26] == 105.0  # candle 25
    assert st["ms_bos_bear"][28] == 1
    assert st["ms_bull_top"][28] == 98.6 and st["ms_bull_bot"][28] == 98.5   # candle 27
    assert st["ms_choch_bull"][29] == 1 and st["ms_trend"][29] == 1
    # last bearish candle before 29 is 28 (c 96.6 < o 98.4) → zone = close→low of 28
    assert st["ms_bear_top"][29] == 96.6 and st["ms_bear_bot"][29] == 96.5


def _mk_df5(o, h, l, c):
    n = len(c)
    ts = pd.date_range("2025-01-06 01:00", periods=n, freq="5min")
    return pd.DataFrame({
        "ts": ts, "avail_ts": ts + pd.Timedelta(minutes=5),
        "day": ts.normalize(), "open": o, "high": h, "low": l, "close": c,
        "tickvol": np.full(n, 100), "spread_pts": np.full(n, 12.0),
    })


def test_zigzag_pivot_confirmation_delay():
    n = 40
    base_price = 100.0
    o = np.full(n, base_price)
    c = np.full(n, base_price)
    h = np.full(n, base_price + 0.2)
    l = np.full(n, base_price - 0.2)
    piv = 20
    h[piv] = 110.0  # clear pivot high
    c[piv] = 109.0
    o[piv] = 100.5

    df5 = _mk_df5(o, h, l, c)
    atr5 = pd.Series(np.full(n, 1.0))
    out = zz_compute(df5, atr5)

    conf = piv + PIV_RIGHT
    assert (out["zz_last_dir"].iloc[:conf] == 0).all(), "pivot used before confirmation!"
    assert out["zz_last_dir"].iloc[conf] == 1.0
    # distance measured against the pivot price 110 from the confirmation bar on
    assert np.isclose(out["zz_dist_pivot"].iloc[conf], (c[conf] - 110.0) / 1.0, atol=1e-4)
