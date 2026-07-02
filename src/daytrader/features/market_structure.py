"""`ms` feature group — causal port of MS.txt (4H / 30m / 5m market structure).

Faithful to the Pine script's state machine:
  • a higher high above the previous `HH_LOOKBACK` highs creates/updates the
    bear zone from the last bearish candle (close→low) and sets trend bullish;
  • a close below the bear-zone bottom flips bearish (CHoCH) and creates the
    bull reclaim zone from the last bullish candle (close→high);
  • while bearish, each fresh lower low refreshes the bull zone (BOS);
  • a bullish close above the bull-zone top flips bullish (CHoCH) and creates
    a fresh bear zone.

One deliberate difference vs TradingView: all HTF state is computed on
COMPLETED HTF bars and joined to decision bars by availability time — the
non-repainting equivalent of request.security(..., lookahead_off). The live
indicator repaints intrabar; we take the conservative variant.
"""
import numpy as np
import pandas as pd

HH_LOOKBACK = 20
LL_LOOKBACK = 20
ZONE_SEARCH = 30
MAX_LOOKBACK = 60  # state machine warmup on its own TF

_STATE_COLS = [
    "ms_trend", "ms_bear_top", "ms_bear_bot", "ms_bull_top", "ms_bull_bot",
    "ms_choch_bull", "ms_choch_bear", "ms_bos_bull", "ms_bos_bear",
    "ms_bars_since_flip", "ms_bars_since_zone",
]


def run_state_machine(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> dict:
    n = len(c)
    prev_hi = pd.Series(h).shift(1).rolling(HH_LOOKBACK).max().to_numpy()
    prev_lo = pd.Series(l).shift(1).rolling(LL_LOOKBACK).min().to_numpy()

    trend = np.zeros(n, dtype=np.int8)
    bear_top = np.full(n, np.nan, dtype=np.float64)
    bear_bot = np.full(n, np.nan, dtype=np.float64)
    bull_top = np.full(n, np.nan, dtype=np.float64)
    bull_bot = np.full(n, np.nan, dtype=np.float64)
    choch_bull = np.zeros(n, dtype=np.int8)
    choch_bear = np.zeros(n, dtype=np.int8)
    bos_bull = np.zeros(n, dtype=np.int8)
    bos_bear = np.zeros(n, dtype=np.int8)
    bars_since_flip = np.full(n, -1, dtype=np.int32)
    bars_since_zone = np.full(n, -1, dtype=np.int32)

    tr = 0
    bt = bb = ut = ub = np.nan          # bear top/bot, bull top/bot
    bear_zone_idx = bull_zone_idx = -1  # candle index the zone was built from
    last_bull = last_bear = -1          # last bullish / bearish candle < current
    last_flip_i = last_zone_i = -1

    for i in range(n):
        made_hh = np.isfinite(prev_hi[i]) and h[i] > prev_hi[i]
        made_ll = np.isfinite(prev_lo[i]) and l[i] < prev_lo[i]
        bearish_flip = tr == 1 and np.isfinite(bb) and c[i] < bb
        bullish_flip = tr == -1 and np.isfinite(ut) and c[i] > o[i] and c[i] > ut

        if bearish_flip:
            tr = -1
            choch_bear[i] = 1
            last_flip_i = i
            if last_bull >= 0 and i - last_bull <= ZONE_SEARCH:
                ut, ub, bull_zone_idx = h[last_bull], c[last_bull], last_bull
                last_zone_i = i
        elif bullish_flip:
            tr = 1
            choch_bull[i] = 1
            last_flip_i = i
            if last_bear >= 0 and i - last_bear <= ZONE_SEARCH:
                bt, bb, bear_zone_idx = c[last_bear], l[last_bear], last_bear
                last_zone_i = i
        elif tr >= 0 and made_hh:
            if last_bear >= 0 and i - last_bear <= ZONE_SEARCH and last_bear != bear_zone_idx:
                if tr == 1:
                    bos_bull[i] = 1  # continuation HH while already bullish
                tr = 1
                bt, bb, bear_zone_idx = c[last_bear], l[last_bear], last_bear
                last_zone_i = i
        elif tr == -1 and made_ll:
            if last_bull >= 0 and i - last_bull <= ZONE_SEARCH and last_bull != bull_zone_idx:
                bos_bear[i] = 1      # continuation LL while bearish
                ut, ub, bull_zone_idx = h[last_bull], c[last_bull], last_bull
                last_zone_i = i

        trend[i] = tr
        bear_top[i], bear_bot[i] = bt, bb
        bull_top[i], bull_bot[i] = ut, ub
        bars_since_flip[i] = i - last_flip_i if last_flip_i >= 0 else -1
        bars_since_zone[i] = i - last_zone_i if last_zone_i >= 0 else -1

        if c[i] > o[i]:
            last_bull = i
        elif c[i] < o[i]:
            last_bear = i

    return {
        "ms_trend": trend, "ms_bear_top": bear_top, "ms_bear_bot": bear_bot,
        "ms_bull_top": bull_top, "ms_bull_bot": bull_bot,
        "ms_choch_bull": choch_bull, "ms_choch_bear": choch_bear,
        "ms_bos_bull": bos_bull, "ms_bos_bear": bos_bear,
        "ms_bars_since_flip": bars_since_flip, "ms_bars_since_zone": bars_since_zone,
    }


def state_frame(bars: pd.DataFrame) -> pd.DataFrame:
    st = run_state_machine(
        bars["open"].to_numpy(np.float64), bars["high"].to_numpy(np.float64),
        bars["low"].to_numpy(np.float64), bars["close"].to_numpy(np.float64),
    )
    out = pd.DataFrame(st)
    out["avail_ts"] = bars["avail_ts"].to_numpy()
    return out


def _derive(out: pd.DataFrame, st: pd.DataFrame, close: pd.Series, a: pd.Series,
            suffix: str) -> None:
    """Turn joined raw state columns into normalized, NaN-safe features."""
    trend = st["ms_trend"].fillna(0)
    out[f"ms_trend_{suffix}"] = trend
    for zone, col in (("bear_top", "ms_bear_top"), ("bear_bot", "ms_bear_bot"),
                      ("bull_top", "ms_bull_top"), ("bull_bot", "ms_bull_bot")):
        d = ((close - st[col]) / a).clip(-20, 20)
        out[f"ms_dist_{zone}_{suffix}"] = d.fillna(0.0)
    out[f"ms_has_bear_{suffix}"] = st["ms_bear_top"].notna().astype("float32")
    out[f"ms_has_bull_{suffix}"] = st["ms_bull_top"].notna().astype("float32")
    out[f"ms_bear_width_{suffix}"] = ((st["ms_bear_top"] - st["ms_bear_bot"]) / a).clip(0, 20).fillna(0)
    out[f"ms_bull_width_{suffix}"] = ((st["ms_bull_top"] - st["ms_bull_bot"]) / a).clip(0, 20).fillna(0)
    out[f"ms_in_bear_{suffix}"] = (
        (close <= st["ms_bear_top"]) & (close >= st["ms_bear_bot"])
    ).astype("float32")
    out[f"ms_in_bull_{suffix}"] = (
        (close <= st["ms_bull_top"]) & (close >= st["ms_bull_bot"])
    ).astype("float32")
    bsf = st["ms_bars_since_flip"].astype("float64")
    out[f"ms_flip_age_{suffix}"] = np.log1p(bsf.where(bsf >= 0, 500).clip(0, 500))
    bsz = st["ms_bars_since_zone"].astype("float64")
    out[f"ms_zone_age_{suffix}"] = np.log1p(bsz.where(bsz >= 0, 500).clip(0, 500))
    for ev in ("choch_bull", "choch_bear", "bos_bull", "bos_bear"):
        out[f"ms_{ev}_{suffix}"] = st[f"ms_{ev}"].fillna(0).astype("float32")


def compute(df5: pd.DataFrame, htf: dict[int, pd.DataFrame], atr5: pd.Series) -> pd.DataFrame:
    """df5: decision bars; htf: {30: bars30, 240: bars240}; atr5 aligned to df5."""
    out = pd.DataFrame(index=df5.index)
    close = df5["close"].astype("float64")
    a = atr5.astype("float64").clip(lower=1e-9)

    # 5m TF: state at bar close, aligned 1:1
    st5 = state_frame(df5).drop(columns="avail_ts")
    st5.index = df5.index
    _derive(out, st5, close, a, "5m")

    # HTF: join last COMPLETED bar state by availability time
    close_time = df5["avail_ts"]  # decision-bar close
    joined = {}
    for minutes, name in ((30, "30m"), (240, "4h")):
        st = state_frame(htf[minutes])
        j = pd.merge_asof(
            pd.DataFrame({"close_time": close_time}).reset_index(),
            st.rename(columns={"avail_ts": "close_time"}).sort_values("close_time"),
            on="close_time", direction="backward",
        ).set_index("index")
        _derive(out, j, close, a, name)
        joined[name] = j

    # native MS.txt entry logic (non-repainting variant)
    t4, t30 = out["ms_trend_4h"], out["ms_trend_30m"]
    bear_bot_30 = joined["30m"]["ms_bear_bot"]
    bull_top_30 = joined["30m"]["ms_bull_top"]
    flip_bear_30 = joined["30m"]["ms_choch_bear"].fillna(0)
    flip_bull_30 = joined["30m"]["ms_choch_bull"].fillna(0)
    # a 30m flip "fires" on the first 5m bar where the new HTF state is visible
    js = joined["30m"]["ms_bars_since_flip"]
    new_state_30 = (js == 0) & (js.shift(1) != 0)

    sell_setup = (t4 == -1) & bear_bot_30.notna() & ((t30 == 1) | (flip_bear_30 * new_state_30 > 0))
    buy_setup = (t4 == 1) & bull_top_30.notna() & ((t30 == -1) | (flip_bull_30 * new_state_30 > 0))
    prev_close = close.shift(1)
    prev_bear_bot = bear_bot_30.shift(1)
    prev_bull_top = bull_top_30.shift(1)
    out["ms_sell_setup"] = sell_setup.astype("float32")
    out["ms_buy_setup"] = buy_setup.astype("float32")
    out["ms_sell_break"] = (
        sell_setup & (close < bear_bot_30) & (prev_close >= prev_bear_bot)
    ).astype("float32")
    out["ms_buy_break"] = (
        buy_setup & (close > bull_top_30) & (prev_close <= prev_bull_top)
    ).astype("float32")
    return out.astype("float32")
