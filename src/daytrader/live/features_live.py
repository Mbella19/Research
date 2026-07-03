"""Live feature path: FULL-history rebuild per decision (D-032 addendum).

The pre-registered pinned-window fast path was REFUTED by measurement
(2026-07-03): the 4h market-structure state machine is path-dependent
without bound — an active supply/demand zone can have formed months before
any window start, so `ms_dist_*_4h` / `ms_*_width_4h` / ages diverged by up
to 20 ATR on 0.5–5.4% of rows (14/424 v2 side flips on the last 120 days).
No window length fixes a structurally unbounded dependence.

The measurement that killed it also made it unnecessary: a full 6.5-year
build takes ~12s (v2 groups) / ~18s (v3 groups) on this machine — well
inside the 90s decision budget. Live therefore rebuilds from FULL history
every decision: bit-identical to the training/backtest path by construction
(same `build_features_from_1m`, same input frame).

The window mode + daily_ctx injection machinery is kept (tested, additive)
for diagnostics and as a future multi-instrument fallback, but is NOT the
live path.
"""
import time

import numpy as np
import pandas as pd

from ..features.registry import build_features_from_1m
from ..utils.log import get_logger
from .history import LiveHistory

log = get_logger("live.features")

PIN_TRADING_DAYS = 120


class FeatureEngine:
    def __init__(self, hist: LiveHistory, groups: list[str],
                 mode: str = "full", pin_trading_days: int = PIN_TRADING_DAYS):
        assert mode in ("full", "window")
        self.hist = hist
        self.groups = list(groups)
        self.mode = mode
        self.pin_days = int(pin_trading_days)
        self.pin_start: pd.Timestamp | None = None
        self.last_build_s: float | None = None
        if mode == "window":
            self.repin()

    # ── the live path ────────────────────────────────────────────────────
    def decision_frame(self) -> pd.DataFrame:
        """Feature frame for the next decision. mode='full' (live default):
        training-identical build over the entire history."""
        t0 = time.time()
        if self.mode == "full":
            out = build_features_from_1m(self.hist.bars, self.groups)
        else:
            out = build_features_from_1m(self.window_1m(), self.groups,
                                         daily_ctx=self._ctx())
        self.last_build_s = time.time() - t0
        return out

    def full_frame(self) -> pd.DataFrame:
        return build_features_from_1m(self.hist.bars, self.groups)

    # ── window mode (diagnostics / future fallback only) ─────────────────
    def repin(self) -> pd.Timestamp:
        days = pd.DatetimeIndex(
            pd.unique(self.hist.bars["ts"].dt.normalize()))
        self.pin_start = (days[-self.pin_days]
                          if len(days) >= self.pin_days else days[0])
        log.info(f"feature window pinned at {self.pin_start.date()} "
                 f"({min(self.pin_days, len(days))} trading days)")
        return self.pin_start

    def window_1m(self) -> pd.DataFrame:
        b = self.hist.bars
        i = int(np.searchsorted(b["ts"].to_numpy(),
                                np.datetime64(self.pin_start)))
        return b.iloc[i:].reset_index(drop=True)

    def _ctx(self):
        return self.hist.feat_daily_ctx() if "daily" in self.groups else None

    def pin_check(self, n_tail: int = 300, atol: float = 1e-5,
                  rtol: float = 1e-4) -> dict:
        """Window-mode diagnostic: last n_tail suffix rows vs full build.
        (Known to FAIL on ms_* columns — kept as the measuring stick.)"""
        if self.pin_start is None:
            self.repin()
        a = build_features_from_1m(self.window_1m(), self.groups,
                                   daily_ctx=self._ctx())
        a = a.tail(n_tail).reset_index(drop=True)
        b = self.full_frame()
        b = b[b["ts"].isin(a["ts"])].reset_index(drop=True)
        cols = [c for c in a.columns
                if c not in ("ts", "avail_ts", "day") and a[c].dtype.kind == "f"]
        worst, bad = 0.0, []
        for c in cols:
            x, y = a[c].to_numpy(np.float64), b[c].to_numpy(np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            if (np.isfinite(x) != np.isfinite(y)).any():
                bad.append((c, "nan-pattern"))
                continue
            if m.any():
                d = float(np.max(np.abs(x[m] - y[m])))
                worst = max(worst, d)
                if not np.allclose(x[m], y[m], atol=atol, rtol=rtol):
                    bad.append((c, d))
        return {"ok": not bad, "worst_abs_diff": worst, "bad_cols": bad}
