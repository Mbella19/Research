"""Feature stack v2 (D-026 Phase B1): daily context, calendar, tape character.

Three groups the v1 model never saw, motivated by the P5 finding (yesterday's
day-character conditions today's breakout follow-through 4–5×):

  daily — lagged day-level regime description: yesterday's efficiency (the
          compression score) + its expanding percentile, realized-vol term
          structure, distances to daily SMAs / N-day extremes, drawdown state,
          maintenance-gap stats, up-day streak. Everything from completed
          days ≤ D−1 except the gap, which is fixed by day D's FIRST bar.
  cal   — calendar effects knowable in advance: day-of-month, turn-of-month,
          opex week, month phase.
  tape  — today-so-far session character, expanding within the day: session
          efficiency, 5m-return autocorrelation, range expansion vs yesterday,
          sign-persistence, relative activity.

All outputs are dimensionless/stationary (ratios, ATR units, percentiles),
clipped to sane ranges, float32. NaN = "not enough history yet" (LightGBM
treats it as missing natively).
"""
import numpy as np
import pandas as pd


def _expanding_pctile(s: pd.Series, min_periods: int = 126) -> pd.Series:
    a = s.to_numpy(np.float64)
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        if i + 1 >= min_periods and np.isfinite(a[i]):
            past = a[: i + 1]
            past = past[np.isfinite(past)]
            if len(past) >= min_periods:
                out[i] = (past <= a[i]).mean()
    return pd.Series(out, index=s.index)


def day_aggs(df5: pd.DataFrame) -> pd.DataFrame:
    """Per-day aggregates the `daily` group is derived from. Public so live
    deployments can build the SAME table from full history and inject it
    (`daily(df5, ctx=...)`) when df5 is only a recent window."""
    g = df5.groupby("day")
    return pd.DataFrame({
        "first_open": g["open"].first(),
        "last_close": g["close"].last(),
        "hi": g["high"].max(),
        "lo": g["low"].min(),
        "absmove": g["close"].apply(lambda c: float(np.abs(np.diff(c)).sum())),
    })


def daily(df5: pd.DataFrame, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    a = day_aggs(df5) if ctx is None else ctx
    if ctx is not None:
        missing = df5["day"].unique()
        missing = missing[~pd.Index(missing).isin(a.index)]
        if len(missing):
            raise ValueError(f"daily ctx missing {len(missing)} day(s), "
                             f"first: {missing[0]}")
    first_open = a["first_open"]
    last_close = a["last_close"]
    hi, lo = a["hi"], a["lo"]
    absmove = a["absmove"]

    tr = np.maximum(hi - lo, np.maximum((hi - last_close.shift(1)).abs(),
                                        (lo - last_close.shift(1)).abs()))
    atr_d = tr.rolling(14).mean()
    dln = np.log(last_close).diff()

    er = ((last_close - first_open).abs() / absmove.replace(0.0, np.nan)).clip(0, 1)
    rv5 = dln.rolling(5).std() * np.sqrt(252)
    rv20 = dln.rolling(20).std() * np.sqrt(252)
    rv60 = dln.rolling(60).std() * np.sqrt(252)

    upd = np.sign(dln)

    def _streak(x: pd.Series) -> pd.Series:
        s, out, run = x.to_numpy(), np.zeros(len(x)), 0.0
        for i, v in enumerate(s):
            run = run + v if v == np.sign(run) or run == 0 else v
            out[i] = run
        return pd.Series(out, index=x.index)

    d = pd.DataFrame({
        "d_er_yday": er,
        "d_er_pct": _expanding_pctile(er),
        "d_range_atr": ((hi - lo) / atr_d),
        "d_rv20": rv20,
        "d_rv_5_20": (rv5 / rv20),
        "d_rv_20_60": (rv20 / rv60),
        "d_vvol": (dln.abs().rolling(20).std() * np.sqrt(252) / rv20),
        "d_dist_sma20": ((last_close - last_close.rolling(20).mean()) / atr_d),
        "d_dist_sma50": ((last_close - last_close.rolling(50).mean()) / atr_d),
        "d_dist_sma200": ((last_close - last_close.rolling(200).mean()) / atr_d),
        "d_dist_hi20": ((last_close - hi.rolling(20).max()) / atr_d),
        "d_dist_lo20": ((last_close - lo.rolling(20).min()) / atr_d),
        "d_dist_hi55": ((last_close - hi.rolling(55).max()) / atr_d),
        "d_dist_lo55": ((last_close - lo.rolling(55).min()) / atr_d),
        "d_dd_depth": (last_close / last_close.expanding().max() - 1.0),
        "d_since_ath": np.log1p((last_close.expanding().max() != last_close)
                                .groupby((last_close.expanding().max()
                                          == last_close).cumsum()).cumsum()),
        "d_streak": _streak(upd).clip(-8, 8),
    }).shift(1)                                   # ← everything above: days ≤ D−1

    # gap features are fixed by day D's first bar (known during day D)
    gap = (first_open - last_close.shift(1)) / atr_d.shift(1)
    d["d_gap"] = gap.clip(-8, 8)
    d["d_gap_sum5"] = gap.rolling(5).sum().clip(-12, 12)

    day = df5["day"]
    out = pd.DataFrame({c: day.map(d[c]) for c in d.columns}, index=df5.index)
    for c in ("d_range_atr", "d_rv_5_20", "d_rv_20_60", "d_vvol"):
        out[c] = out[c].clip(0, 8)
    for c in ("d_dist_sma20", "d_dist_sma50", "d_dist_sma200",
              "d_dist_hi20", "d_dist_lo20", "d_dist_hi55", "d_dist_lo55"):
        out[c] = out[c].clip(-15, 15)
    return out.astype(np.float32)


def cal(df5: pd.DataFrame) -> pd.DataFrame:
    ts = df5["ts"]
    dom = ts.dt.day
    month_end = ts.dt.days_in_month
    # 3rd Friday: opex week = calendar days 15..21 containing/around it
    opex = ((dom >= 15) & (dom <= 21)).astype(np.float32)
    tom = ((dom >= month_end - 2) | (dom <= 4)).astype(np.float32)
    ang = 2 * np.pi * (ts.dt.month - 1) / 12.0
    return pd.DataFrame({
        "c_dom": (dom / month_end).astype(np.float32),
        "c_tom": tom,
        "c_opex": opex,
        "c_month_sin": np.sin(ang).astype(np.float32),
        "c_month_cos": np.cos(ang).astype(np.float32),
    }, index=df5.index)


def tape(df5: pd.DataFrame) -> pd.DataFrame:
    g = df5.groupby("day")
    c = df5["close"]
    ret = g["close"].diff()
    k = g.cumcount()

    first_open = g["open"].transform("first")
    cum_abs = ret.abs().groupby(df5["day"]).cumsum()
    sess_er = ((c - first_open).abs() / cum_abs.replace(0.0, np.nan)).clip(0, 1)
    sess_er = sess_er.where(k >= 6)

    # expanding within-day lag-1 autocorr of 5m returns (cheap running moments)
    r, r1 = ret, ret.groupby(df5["day"]).shift(1)
    pair = r.notna() & r1.notna()
    day = df5["day"]

    def _csum(x):
        return x.fillna(0.0).groupby(day).cumsum()

    n = _csum(pair.astype(float))
    sx, sy = _csum(r1.where(pair)), _csum(r.where(pair))
    sxx, syy = _csum((r1 ** 2).where(pair)), _csum((r ** 2).where(pair))
    sxy = _csum((r * r1).where(pair))
    cov = sxy / n - (sx / n) * (sy / n)
    den = np.sqrt((sxx / n - (sx / n) ** 2).clip(lower=0)
                  * (syy / n - (sy / n) ** 2).clip(lower=0))
    ac1 = (cov / den.replace(0.0, np.nan)).clip(-1, 1).where(n >= 12)

    hi_so_far = g["high"].cummax()
    lo_so_far = g["low"].cummin()
    yday_range = (g["high"].max() - g["low"].min()).shift(1)
    range_exp = ((hi_so_far - lo_so_far)
                 / df5["day"].map(yday_range).replace(0.0, np.nan)).clip(0, 6)

    agree = (np.sign(ret) == np.sign(ret.groupby(day).shift(1))).astype(float)
    persist = agree.rolling(24, min_periods=12).mean().where(k >= 12)

    yday_tv = df5["day"].map(g["tickvol"].mean().shift(1))
    tv_rel = ((df5["tickvol"].groupby(day).cumsum() / (k + 1))
              / yday_tv.replace(0.0, np.nan)).clip(0, 8)

    return pd.DataFrame({
        "t_sess_er": sess_er, "t_ac1": ac1, "t_range_exp": range_exp,
        "t_persist": persist, "t_tv_rel": tv_rel,
    }, index=df5.index).astype(np.float32)
