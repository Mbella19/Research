"""THE anti-lookahead gate.

Every feature value at decision time t must be bit-identical whether computed
on the full history or on data truncated at t. Any mismatch = a feature that
peeks into the future = disqualifying bug.
"""
import numpy as np
import pandas as pd
import pytest

from daytrader.data.loader import load_bars
from daytrader.features.registry import GROUPS, build_features_from_1m, feature_cols

N_PROBES = 16


@pytest.fixture(scope="module")
def df1m():
    df = load_bars("real_validation")
    days = pd.Series(df["ts"].dt.normalize().unique())
    return df[df["ts"] < days.iloc[70]].reset_index(drop=True)


@pytest.fixture(scope="module")
def full(df1m):
    return build_features_from_1m(df1m, list(GROUPS))


def test_no_lookahead(df1m, full):
    rng = np.random.default_rng(7)
    idx = rng.choice(np.arange(len(full) // 2, len(full) - 1), size=N_PROBES, replace=False)
    cols = feature_cols(full)
    for i in idx:
        t_close = full["avail_ts"].iloc[i]
        trunc = df1m[df1m["ts"] + pd.Timedelta(minutes=1) <= t_close].reset_index(drop=True)
        rebuilt = build_features_from_1m(trunc, list(GROUPS))
        assert rebuilt["ts"].iloc[-1] == full["ts"].iloc[i], "row alignment broke"
        a = full.iloc[i][cols].to_numpy(np.float64)
        b = rebuilt.iloc[-1][cols].to_numpy(np.float64)
        bad = ~np.isclose(a, b, atol=1e-5, rtol=1e-4, equal_nan=True)
        assert not bad.any(), (
            f"LOOKAHEAD at {t_close}: "
            f"{[(c, float(x), float(y)) for c, x, y in zip(np.array(cols)[bad], a[bad], b[bad])]}"
        )
