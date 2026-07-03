"""Live feature path acceptance (D-032, post-addendum): the live decision
build must be EXACTLY the training/backtest build — same function, full
history — and fit the latency budget. The refuted window mode is probed as
a diagnostic (documented divergence on ms_* columns; not asserted).

Slow (real data). Run explicitly:  pytest -m slow -k suffix
"""
from pathlib import Path

import pandas as pd
import pytest

from daytrader.data.loader import load_mt5_csv
from daytrader.features.registry import build_features_from_1m
from daytrader.live.features_live import FeatureEngine
from daytrader.live.history import LiveHistory, stored_real_frames

NEW_CSV = Path.home() / "Downloads" / "new data.csv"

POLICIES = {
    "v2": ["base", "time", "ms", "zz"],
    "v3": ["base", "time", "ms", "zz", "daily", "cal", "tape"],
}


@pytest.fixture(scope="module")
def hist(tmp_path_factory):
    frames = stored_real_frames()
    assert len(frames) == 3, "expected training+validation+locked_oos seeds"
    if NEW_CSV.exists():
        frames.append(load_mt5_csv(NEW_CSV))
    h = LiveHistory("paritytest", root=tmp_path_factory.mktemp("live"))
    h.seed(frames)
    return h


@pytest.mark.slow
@pytest.mark.parametrize("name", ["v2", "v3"])
def test_live_path_is_training_path(hist, name):
    groups = POLICIES[name]
    eng = FeatureEngine(hist, groups)          # mode="full" (live default)
    live = eng.decision_frame()
    print(f"\n[{name}] full decision build: {len(live):,} rows "
          f"in {eng.last_build_s:.1f}s")
    assert eng.last_build_s < 60.0, "decision build must fit the live budget"
    ref = build_features_from_1m(hist.bars, groups)
    pd.testing.assert_frame_equal(live, ref)


@pytest.mark.slow
def test_window_mode_divergence_documented(hist):
    """The refuted fast path: record (don't assert) its divergence so any
    future improvement is measured against the same stick."""
    eng = FeatureEngine(hist, POLICIES["v2"], mode="window")
    rep = eng.pin_check(n_tail=300)
    print(f"\n[window-mode diagnostic] ok={rep['ok']} "
          f"worst|Δ|={rep['worst_abs_diff']:.3g} "
          f"bad_cols={[c for c, _ in rep['bad_cols']][:6]}")
    assert isinstance(rep["ok"], bool)
