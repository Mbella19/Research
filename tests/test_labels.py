"""Label correctness: invariants + slow-reference cross-check on real data."""
import numpy as np
import pandas as pd
import pytest

from daytrader.config import experiment, instrument
from daytrader.data.loader import load_bars
from daytrader.labels.triple_barrier import build_labels


@pytest.fixture(scope="module")
def labels():
    return build_labels("real_validation")


def test_invariants(labels):
    el = labels["eligible"]
    assert el.sum() > 10_000
    both = (labels.loc[el, "y_long"] == 1) & (labels.loc[el, "y_short"] == 1)
    assert both.sum() == 0, "long and short cannot both win with TP > SL"
    assert (labels.loc[el, "t1_long"] >= labels.loc[el, "ts"]).all()
    for side in ("long", "short"):
        rate = labels.loc[el, f"y_{side}"].mean()
        assert 0.05 < rate < 0.5, f"{side} base rate implausible: {rate}"
        w = labels.loc[el, f"w_uniq_{side}"]
        assert (w > 0).all() and (w <= 1.0 + 1e-6).all()


def test_slow_reference_agreement(labels):
    """Re-derive 150 random labels with a dumb per-bar loop; must match exactly."""
    ex = experiment()
    lab = ex["labels"]
    sess = instrument()["session"]
    W = lab["horizon_bars"] * ex["timeframes"]["decision_minutes"]
    fh, fm = sess["force_flat_at"].split(":")
    flat_mod = int(fh) * 60 + int(fm)

    df1 = load_bars("real_validation")
    ts1 = df1["ts"].to_numpy()
    day1 = df1["ts"].dt.normalize().to_numpy()
    mod1 = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute).to_numpy()
    o1 = df1["open"].to_numpy(float)
    h1 = df1["high"].to_numpy(float)
    l1 = df1["low"].to_numpy(float)

    el_idx = np.flatnonzero(labels["eligible"].to_numpy())
    rng = np.random.default_rng(3)
    for i in rng.choice(el_idx, 150, replace=False):
        row = labels.iloc[i]
        t_close = row["ts"] + pd.Timedelta(minutes=ex["timeframes"]["decision_minutes"])
        e = int(np.searchsorted(ts1, t_close.to_numpy(), side="left"))
        entry = o1[e]
        assert np.isclose(entry, row["entry_open"], atol=1e-3)
        atr_ = float(row["atr_abs"])
        tp_l, sl_l = entry + lab["tp_atr"] * atr_, entry - lab["sl_atr"] * atr_
        tp_s, sl_s = entry - lab["tp_atr"] * atr_, entry + lab["sl_atr"] * atr_
        yl = ys = 0
        done_l = done_s = False
        for k in range(W):
            j = e + k
            if j >= len(ts1) or day1[j] != day1[e] or mod1[j] >= flat_mod:
                break
            if not done_l:
                if l1[j] <= sl_l:          # SL priority on ambiguity
                    done_l = True
                elif h1[j] >= tp_l:
                    yl, done_l = 1, True
            if not done_s:
                if h1[j] >= sl_s:
                    done_s = True
                elif l1[j] <= tp_s:
                    ys, done_s = 1, True
            if done_l and done_s:
                break
        assert yl == int(row["y_long"]), f"long mismatch at row {i} ({row['ts']})"
        assert ys == int(row["y_short"]), f"short mismatch at row {i} ({row['ts']})"
