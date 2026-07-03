"""Live bar store per account: full 1m history + full-history daily contexts.

Seeded once from every stored real source (locked OOS joins under the same
flag-or-env policy as book.py `_real_daily_context`) plus any fresh exports
and a broker backfill; then appended live from the executor's bars.csv.

Truth rules (D-032):
- appends must overlap-or-abut; overlapping rows must MATCH — mismatches are
  counted as `revised_bars` and the terminal's version wins (exports are the
  corrected history; decisions already made stay as-decided in the logs);
- the two daily context tables (`daily_ohlc` for the S2 gate/ATR/derisk,
  `feat_daily_ctx` for the feature `daily` group) are always computed from
  the FULL history, incrementally refreshed from the first dirty day.
"""
import gc
import os

import numpy as np
import pandas as pd

from ..data.resample import resample_bars
from ..features.daily_context import day_aggs
from ..portfolio.overnight import daily_frame
from ..utils import paths
from ..utils.log import get_logger

log = get_logger("live.history")

BAR_COLS = ["ts", "open", "high", "low", "close", "tickvol", "spread_pts"]
CMP_COLS = ["open", "high", "low", "close"]  # revision check (tickvol/spread
                                             # may legitimately be refined)


def stored_real_frames() -> list[pd.DataFrame]:
    """All stored real 1m sources in chronological order. Locked OOS is
    included only when consumed (flag) or explicitly unlocked — the same
    policy as book.py:_real_daily_context / run_fullhistory."""
    oos_ok = ((paths.RUNS_DIR / "OOS_EXECUTED.flag").exists()
              or os.environ.get("DAYTRADER_UNLOCK_OOS") == "1")
    sources = ["real_training", "real_validation"] + (
        ["real_locked_oos"] if oos_ok else [])
    out = []
    for s in sources:
        p = paths.PARQUET_DIR / f"{s}.parquet"
        if p.exists():
            out.append(pd.read_parquet(p, columns=BAR_COLS))
    return out


class LiveHistory:
    """Full-history 1m bar store for one live account."""

    def __init__(self, name: str, root=None):
        self.name = name
        self.dir = (root or (paths.DATA_DIR / "live")) / name
        self.path = self.dir / "bars_1m.parquet"
        self.bars: pd.DataFrame | None = None
        self._aggs: pd.DataFrame | None = None      # feature daily ctx cache
        self._dohlc: pd.DataFrame | None = None     # S2 daily OHLC cache
        self._dirty_from: pd.Timestamp | None = None

    # ── persistence ──────────────────────────────────────────────────────
    def load(self) -> bool:
        if self.path.exists():
            self.bars = pd.read_parquet(self.path)
            self._invalidate(self.bars["ts"].iloc[0])
            return True
        return False

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.parquet")
        self.bars.to_parquet(tmp, index=False)
        os.replace(tmp, self.path)

    # ── construction ─────────────────────────────────────────────────────
    def seed(self, frames: list[pd.DataFrame]) -> dict:
        """Concatenate chronologically ordered frames into the store.
        Frames may overlap; overlaps are deduped keeping the LATER frame
        (later exports are the corrected history)."""
        parts = []
        for f in frames:
            f = f[BAR_COLS].sort_values("ts")
            parts.append(f)
        allb = pd.concat(parts, ignore_index=True)
        allb = allb.drop_duplicates(subset="ts", keep="last").sort_values("ts")
        allb = allb.reset_index(drop=True)
        d = allb["ts"].diff().dt.total_seconds()
        if (d <= 0).any():
            raise ValueError("seed produced non-increasing timestamps")
        self.bars = allb
        self._invalidate(allb["ts"].iloc[0])
        rep = {
            "rows": int(len(allb)),
            "start": str(allb["ts"].iloc[0]),
            "end": str(allb["ts"].iloc[-1]),
            "days": int(allb["ts"].dt.normalize().nunique()),
        }
        log.info(f"{self.name}: seeded {rep['rows']:,} bars "
                 f"{rep['start']} → {rep['end']} ({rep['days']} days)")
        gc.collect()
        return rep

    def append(self, df_new: pd.DataFrame) -> dict:
        """Merge a fresh (small) bar frame from the executor. Returns
        {'n_new', 'revised', 'gap_s'}."""
        if self.bars is None:
            raise RuntimeError("history not seeded")
        df_new = (df_new[BAR_COLS].drop_duplicates(subset="ts", keep="last")
                  .sort_values("ts").reset_index(drop=True))
        last_ts = self.bars["ts"].iloc[-1]
        new_mask = df_new["ts"] > last_ts
        revised = 0
        dirty: pd.Timestamp | None = None

        overlap = df_new[~new_mask]
        if len(overlap):
            lo_ts = overlap["ts"].iloc[0]
            i0 = int(np.searchsorted(self.bars["ts"].to_numpy(),
                                     np.datetime64(lo_ts)))
            tail = self.bars.iloc[i0:]
            m = tail.merge(overlap, on="ts", suffixes=("", "_n"), how="inner")
            if len(m):
                diff = np.zeros(len(m), dtype=bool)
                for c in CMP_COLS:
                    diff |= (m[c].to_numpy() != (m[c + "_n"]).to_numpy())
                revised = int(diff.sum())
                if revised:
                    # terminal wins: replace the revised rows in place
                    rts = set(m.loc[diff, "ts"])
                    idx = self.bars.index[self.bars["ts"].isin(rts)]
                    repl = overlap.set_index("ts")
                    for c in BAR_COLS[1:]:
                        self.bars.loc[idx, c] = (
                            self.bars.loc[idx, "ts"].map(repl[c]).values)
                    dirty = min(rts).normalize()

        new_rows = df_new[new_mask]
        gap_s = 0.0
        if len(new_rows):
            gap_s = float((new_rows["ts"].iloc[0] - last_ts).total_seconds())
            self.bars = pd.concat([self.bars, new_rows], ignore_index=True)
            d0 = new_rows["ts"].iloc[0].normalize()
            dirty = d0 if dirty is None else min(dirty, d0)

        if dirty is not None:
            self._invalidate(dirty)
        return {"n_new": int(len(new_rows)), "revised": revised, "gap_s": gap_s}

    # ── daily context tables (full history, incremental) ────────────────
    def _invalidate(self, from_day: pd.Timestamp) -> None:
        d = pd.Timestamp(from_day).normalize()
        self._dirty_from = d if self._dirty_from is None else min(self._dirty_from, d)

    def _slice_from_day(self, day: pd.Timestamp) -> pd.DataFrame:
        i = int(np.searchsorted(self.bars["ts"].to_numpy(), np.datetime64(day)))
        return self.bars.iloc[i:]

    def _refresh(self) -> None:
        if self._dirty_from is None:
            return
        if self._aggs is None or self._dohlc is None or \
                self._dirty_from <= self._aggs.index[0]:
            df5 = resample_bars(self.bars, 5)
            self._aggs = day_aggs(df5)
            self._dohlc = daily_frame(self.bars)
        else:
            # recompute from the first dirty day (bins are day-anchored, so a
            # day-boundary slice reproduces the same 5m bars exactly)
            start = min(self._dirty_from, self._aggs.index[-1])
            part = self._slice_from_day(start)
            a_new = day_aggs(resample_bars(part, 5))
            d_new = daily_frame(part)
            self._aggs = pd.concat([self._aggs[self._aggs.index < start], a_new])
            self._dohlc = pd.concat([self._dohlc[self._dohlc.index < start], d_new])
        self._dirty_from = None

    def feat_daily_ctx(self) -> pd.DataFrame:
        """Full-history per-day aggregates for daily_context.daily(ctx=...).
        Includes today's (partial) row — causally safe: the daily block is
        shift(1)-lagged and the gap feature is fixed by today's first bar."""
        self._refresh()
        return self._aggs

    def daily_ohlc(self) -> pd.DataFrame:
        """Full-history daily OHLC for the S2 gate/ATR/derisk (context_daily
        pattern). Includes today's partial row (harmless: gate is lagged)."""
        self._refresh()
        return self._dohlc
