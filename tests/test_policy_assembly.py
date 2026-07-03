"""PolicyRuntime assembly: frozen dicts resolve exactly as pre-registered
(D-032), sizing formulas reproduce the engine, and the gate path is
rowwise-pure (tail slice == full frame)."""
import numpy as np
import pandas as pd
import pytest

from daytrader.config import clear_overrides
from daytrader.live.policy import MAGICS, V2_DECISION, PolicyRuntime
from daytrader.utils import paths
from daytrader.utils.hashio import load_json


@pytest.fixture(autouse=True)
def _clean_overrides():
    yield
    clear_overrides()


def test_v2_assembly():
    p = PolicyRuntime.load("v2")
    assert p.decision == V2_DECISION
    assert p.sleeves["w2"] == 0.87 and p.risk_per_trade == 0.005
    assert p.s2p.stop_atr is None and p.s2p.derisk is False
    assert p.s2p.window == "usC" and p.s2p.gate == "sma50"
    assert (p.magic_s1, p.magic_s2) == (622001, 622002)
    assert p.labels["tp_atr"] == 3.0 and p.horizon_1m_bars() == 240


def test_v3_assembly():
    p = PolicyRuntime.load("v3")
    fz = load_json(paths.MODELS_DIR / "FINAL_FROZEN_V3.json")
    assert p.decision == fz["decision"]
    assert p.decision["min_ev_atr"] == 0.10
    assert p.sleeves["w2"] == 1.17
    assert p.s2p.stop_atr == 5.0 and p.s2p.derisk is True
    assert (p.magic_s1, p.magic_s2) == (623001, 623002)
    assert set(MAGICS["v2"]).isdisjoint(MAGICS["v3"])


def _rand_feature_frame(p: PolicyRuntime, n=64, seed=11):
    rng = np.random.default_rng(seed)
    meta = load_json(p.art_path / "meta.json")
    ts = pd.date_range("2026-03-02 10:00", periods=n, freq="5min")
    df = pd.DataFrame({c: rng.normal(0, 1, n).astype(np.float32)
                       for c in meta["feature_names"]})
    df["ts"], df["avail_ts"] = ts, ts + pd.Timedelta(minutes=5)
    df["day"] = df["ts"].dt.normalize()
    df["close"] = np.float32(24000.0)
    df["_atr_points"] = np.float32(300.0)          # atr_abs = 30.0 px
    if "spread_now" in df.columns:
        df["spread_now"] = np.float32(12.0)
    else:
        df["spread_now"] = np.float32(12.0)
    return df


@pytest.mark.parametrize("name", ["v2", "v3"])
def test_gate_path_rowwise_pure(name):
    p = PolicyRuntime.load(name)
    feat = _rand_feature_frame(p)
    full = p.s1_signals(feat)
    tail = p.s1_signals(feat, tail_rows=10)
    ref = full.tail(10).reset_index(drop=True)
    pd.testing.assert_frame_equal(tail, ref)
    assert set(full["side"].unique()) <= {-1, 0, 1}


def test_s1_lots_engine_formula():
    p = PolicyRuntime.load("v3")
    eq, atr, px = 50_000.0, 30.0, 24_000.0
    lots, riskd = p.s1_lots(eq, atr, px)
    assert riskd == 250.0
    # engine.py:209-216: 250/(1.5*30*1.0)=5.555… → floor to 5.55
    assert lots == pytest.approx(5.55)
    # leverage cap binds: tiny ATR → huge lots, cap = eq*20/px = 41.66 → 41.66→41.66
    lots2, _ = p.s1_lots(eq, 0.1, px)
    assert lots2 == pytest.approx(np.floor((eq * 20 / px) / 0.01) * 0.01)
    # sizing floor: microscopic equity → lots < 0.01 → skip (0.0)
    lots3, _ = p.s1_lots(10.0, atr, px)
    assert lots3 == 0.0


def test_s1_levels_anchored_to_fill_and_rounded():
    p = PolicyRuntime.load("v3")
    sl, tp = p.s1_levels(24000.33, +1, 30.0)
    assert sl == pytest.approx(round(24000.33 - 45.0, 1))
    assert tp == pytest.approx(round(24000.33 + 90.0, 1))
    sl_s, tp_s = p.s1_levels(24000.33, -1, 30.0)
    assert sl_s == pytest.approx(round(24000.33 + 45.0, 1))
    assert tp_s == pytest.approx(round(24000.33 - 90.0, 1))


def test_s2_sizing_and_stop():
    v3 = PolicyRuntime.load("v3")
    # live.py:105-107: floor(expo*w2*eq/close/step)*step
    lots = v3.s2_lots(50_000.0, 24_000.0, expo=0.8)
    assert lots == pytest.approx(np.floor((0.8 * 1.17 * 50_000 / 24_000) / 0.01) * 0.01)
    assert v3.s2_stop_level(24_000.0, 250.0) == pytest.approx(24_000 - 1250.0)
    assert v3.s2_stop_level(24_000.0, float("nan")) is None
    v2 = PolicyRuntime.load("v2")
    assert v2.s2_stop_level(24_000.0, 250.0) is None   # v2: NO stop (parity)


def test_s2_state_gate_lag():
    v2 = PolicyRuntime.load("v2")
    n = 60
    idx = pd.bdate_range("2026-01-05", periods=n)
    up = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": np.linspace(100, 160, n)}, index=idx)
    st = v2.s2_state(up)
    assert st["gate"] is True and st["expo"] == 1.0    # v2: no derisk
    down = up.assign(close=np.linspace(160, 100, n))
    assert v2.s2_state(down)["gate"] is False
