"""`zz` feature group — causal port of "All Chart Patterns — Detector".

Zigzag pivots (pivLen=8 left, pivRight=2 confirmation bars) feed a pattern
engine with the Pine script's exact priority chain and lifecycle:
FORMING → AWAIT (breakout close) → REACHED (target touch) / FAILED
(invalidation close or expiry, incl. apex expiry for converging patterns).

Causality contract: a pivot at bar p is only usable from bar p + PIV_RIGHT
(its confirmation bar), matching `ta.pivothigh(high, 8, 2)` semantics; all
status transitions are evaluated on confirmed bar closes, matching the
script's `barstate.isconfirmed` gate. Ties (equal highs) reject a pivot —
documented deviation for determinism; Pine tie-handling differs per feed.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PIV_LEN = 8
PIV_RIGHT = 2
TOL = 0.25
MAX_WIDTH = 200
MAX_PAT = 20
MAX_ZZ = 60
MAX_LOOKBACK = 260  # pattern width cap + pivot window

ST_FORMING, ST_AWAIT, ST_REACHED, ST_FAILED = 0, 1, 2, 3

PATTERN_CODES = {
    "Triple Top": 1, "Triple Bottom": 2, "Head and Shoulders": 3,
    "Inv. Head and Shoulders": 4, "Rectangle": 5, "Double Top": 6,
    "Double Bottom": 7, "Bullish Flag": 8, "Bearish Flag": 9,
    "Bullish Pennant": 10, "Bearish Pennant": 11, "Rising Wedge": 12,
    "Falling Wedge": 13, "Triangle": 14,
}


def _val_at(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    return y2 if x2 == x1 else y1 + (y2 - y1) * (x - x1) / (x2 - x1)


@dataclass
class Pattern:
    name: str
    dir: int
    two_lines: bool
    xs: list
    ys: list
    nx1: int; ny1: float; nx2: int; ny2: float
    mx1: int = 0; my1: float = np.nan; mx2: int = 0; my2: float = np.nan
    tgt_size: float = 0.0
    inv_level: float = np.nan
    expiry_bar: int = 0
    status: int = ST_FORMING
    born: int = 0
    bk_bar: int = -1
    bk_price: float = np.nan
    target: float = np.nan
    done_bar: int = -1

    def neck_at(self, x: float) -> float:
        return _val_at(self.nx1, self.ny1, self.nx2, self.ny2, x)

    def top_at(self, x: float) -> float:
        return _val_at(self.mx1, self.my1, self.mx2, self.my2, x)


def confirmed_pivots(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays over CONFIRMATION bars i: pivot at p=i-PIV_RIGHT confirmed at i."""
    h = pd.Series(high)
    l = pd.Series(low)
    left_max = h.shift(1).rolling(PIV_LEN, min_periods=PIV_LEN).max()
    # skipna=False: a pivot needs ALL right-side bars to exist (no early confirmation)
    right_max = pd.concat([h.shift(-k) for k in range(1, PIV_RIGHT + 1)], axis=1).max(axis=1, skipna=False)
    is_ph = (h > left_max) & (h > right_max)
    left_min = l.shift(1).rolling(PIV_LEN, min_periods=PIV_LEN).min()
    right_min = pd.concat([l.shift(-k) for k in range(1, PIV_RIGHT + 1)], axis=1).min(axis=1, skipna=False)
    is_pl = (l < left_min) & (l < right_min)
    ph_conf = np.zeros(len(h), dtype=bool)
    pl_conf = np.zeros(len(h), dtype=bool)
    ph_conf[PIV_RIGHT:] = is_ph.to_numpy()[:-PIV_RIGHT]
    pl_conf[PIV_RIGHT:] = is_pl.to_numpy()[:-PIV_RIGHT]
    return ph_conf, pl_conf


class _Engine:
    def __init__(self) -> None:
        self.zzP: list[float] = []
        self.zzB: list[int] = []
        self.zzD: list[int] = []
        self.pats: list[Pattern] = []

    # zigzag builder: alternation kept, same-direction extremes replaced
    def add_pivot(self, d: int, price: float, b: int) -> bool:
        if not self.zzD:
            self.zzP.append(price); self.zzB.append(b); self.zzD.append(d)
            return True
        if self.zzD[-1] == d:
            if (d > 0 and price > self.zzP[-1]) or (d < 0 and price < self.zzP[-1]):
                self.zzP[-1] = price; self.zzB[-1] = b
                return True
            return False
        self.zzP.append(price); self.zzB.append(b); self.zzD.append(d)
        if len(self.zzP) > MAX_ZZ:
            self.zzP.pop(0); self.zzB.pop(0); self.zzD.pop(0)
        return True

    def _new_pattern(self, i: int, name: str, dir: int, two: bool, k: int,
                     nx1, ny1, nx2, ny2, mx1=0, my1=np.nan, mx2=0, my2=np.nan,
                     tgt_size=0.0, inv_level=np.nan) -> bool:
        xs = self.zzB[-k:]
        ys = self.zzP[-k:]
        w = xs[-1] - xs[0]
        if MAX_WIDTH and w > MAX_WIDTH:
            return False
        p = Pattern(name, dir, two, xs, ys, nx1, ny1, nx2, ny2, mx1, my1, mx2, my2,
                    tgt_size, inv_level, born=i)
        p.expiry_bar = xs[-1] + max(20, 2 * w)
        if two:
            sU = (my2 - my1) / max(1, mx2 - mx1)
            sL = (ny2 - ny1) / max(1, nx2 - nx1)
            if abs(sU - sL) > 1e-10:
                apex = ((ny1 - sL * nx1) - (my1 - sU * mx1)) / (sU - sL)
                if np.isfinite(apex) and apex > xs[-1]:
                    p.expiry_bar = min(p.expiry_bar, int(apex))
        # a new pattern supersedes overlapping still-forming ones
        self.pats = [q for q in self.pats
                     if not (q.status == ST_FORMING and q.xs[-1] >= xs[0])]
        self.pats.append(p)
        while len(self.pats) > MAX_PAT:
            self.pats.pop(0)
        return True

    # pattern detection — faithful priority chain, one pattern per zigzag event
    def detect(self, i: int) -> bool:
        n = len(self.zzP)
        if n < 4:
            return False
        P = lambda k: self.zzP[-1 - k]
        B = lambda k: self.zzB[-1 - k]
        d0 = self.zzD[-1]
        p0, p1, p2, p3 = P(0), P(1), P(2), P(3)
        b0, b1, b2, b3 = B(0), B(1), B(2), B(3)
        p4 = P(4) if n >= 5 else np.nan
        b4 = B(4) if n >= 5 else 0
        p5 = P(5) if n >= 6 else np.nan

        # Triple Top / Bottom
        if d0 == 1 and n >= 6:
            neck = min(p3, p1); hi = max(p0, p2, p4); lo = min(p0, p2, p4)
            h = hi - neck
            if h > 0 and hi - lo <= TOL * h and lo > neck and p5 < neck:
                return self._new_pattern(i, "Triple Top", -1, False, 5,
                                         b4, neck, b0, neck, tgt_size=h, inv_level=hi)
        if d0 == -1 and n >= 6:
            neck = max(p3, p1); lo = min(p0, p2, p4); hi = max(p0, p2, p4)
            h = neck - lo
            if h > 0 and hi - lo <= TOL * h and hi < neck and p5 > neck:
                return self._new_pattern(i, "Triple Bottom", 1, False, 5,
                                         b4, neck, b0, neck, tgt_size=h, inv_level=lo)
        # Head and Shoulders / inverted
        if d0 == 1 and n >= 6:
            head, ls, rs = p2, p4, p0
            neck_min = min(p3, p1); h = head - neck_min
            if (h > 0 and head - max(ls, rs) > TOL * h and abs(ls - rs) <= 1.5 * TOL * h
                    and min(ls, rs) > max(p3, p1) and abs(p3 - p1) <= 0.5 * h and p5 < neck_min):
                tgt = head - _val_at(b3, p3, b1, p1, b2)
                return self._new_pattern(i, "Head and Shoulders", -1, False, 5,
                                         b3, p3, b1, p1, tgt_size=tgt, inv_level=head)
        if d0 == -1 and n >= 6:
            head, ls, rs = p2, p4, p0
            neck_max = max(p3, p1); h = neck_max - head
            if (h > 0 and min(ls, rs) - head > TOL * h and abs(ls - rs) <= 1.5 * TOL * h
                    and max(ls, rs) < min(p3, p1) and abs(p3 - p1) <= 0.5 * h and p5 > neck_max):
                tgt = _val_at(b3, p3, b1, p1, b2) - head
                return self._new_pattern(i, "Inv. Head and Shoulders", 1, False, 5,
                                         b3, p3, b1, p1, tgt_size=tgt, inv_level=head)
        # Rectangle
        if n >= 5:
            if d0 == 1:
                hi_max, hi_min = max(p4, p2, p0), min(p4, p2, p0)
                lo_max, lo_min = max(p3, p1), min(p3, p1)
            else:
                hi_max, hi_min = max(p3, p1), min(p3, p1)
                lo_max, lo_min = max(p4, p2, p0), min(p4, p2, p0)
            top, bot = (hi_max + hi_min) / 2, (lo_max + lo_min) / 2
            h = top - bot
            if h > 0 and hi_min > lo_max and hi_max - hi_min <= TOL * h and lo_max - lo_min <= TOL * h:
                return self._new_pattern(i, "Rectangle", 0, True, 5,
                                         b4, bot, b0, bot, b4, top, b0, top,
                                         tgt_size=h, inv_level=np.nan)
        # Double Top / Bottom
        if d0 == 1:
            h = max(p2, p0) - p1
            if h > 0 and abs(p0 - p2) <= TOL * h and p3 < p1:
                return self._new_pattern(i, "Double Top", -1, False, 3,
                                         b2, p1, b0, p1, tgt_size=h, inv_level=max(p2, p0))
        if d0 == -1:
            h = p1 - min(p2, p0)
            if h > 0 and abs(p0 - p2) <= TOL * h and p3 > p1:
                return self._new_pattern(i, "Double Bottom", 1, False, 3,
                                         b2, p1, b0, p1, tgt_size=h, inv_level=min(p2, p0))
        # Flags
        if d0 == -1 and n >= 5:
            pole = p3 - p4
            hC = ((p3 - p2) + (p1 - p0)) / 2
            wC = b0 - b3
            sH = (p1 - p3) / max(1, b1 - b3)
            sL = (p0 - p2) / max(1, b0 - b2)
            if (pole > 0 and hC > 0 and pole >= 2.0 * hC and p1 < p3 and p0 < p2
                    and p3 - min(p0, p2) <= 0.6 * pole
                    and abs(sH - sL) * wC <= 2 * TOL * hC and b3 - b4 <= 2 * wC):
                return self._new_pattern(i, "Bullish Flag", 1, True, 5,
                                         b2, p2, b0, p0, b3, p3, b1, p1,
                                         tgt_size=pole, inv_level=p4 + 0.3 * pole)
        if d0 == 1 and n >= 5:
            pole = p4 - p3
            hC = ((p2 - p3) + (p0 - p1)) / 2
            wC = b0 - b3
            sH = (p0 - p2) / max(1, b0 - b2)
            sL = (p1 - p3) / max(1, b1 - b3)
            if (pole > 0 and hC > 0 and pole >= 2.0 * hC and p1 > p3 and p0 > p2
                    and max(p0, p2) - p3 <= 0.6 * pole
                    and abs(sH - sL) * wC <= 2 * TOL * hC and b3 - b4 <= 2 * wC):
                return self._new_pattern(i, "Bearish Flag", -1, True, 5,
                                         b3, p3, b1, p1, b2, p2, b0, p0,
                                         tgt_size=pole, inv_level=p4 - 0.3 * pole)
        # Pennants
        if d0 == -1 and n >= 5:
            pole = p3 - p4
            hC = p3 - p2
            if pole > 0 and hC > 0 and pole >= 2.0 * hC and p1 < p3 and p0 > p2 and b3 - b4 <= 2 * (b0 - b3):
                return self._new_pattern(i, "Bullish Pennant", 1, True, 5,
                                         b2, p2, b0, p0, b3, p3, b1, p1,
                                         tgt_size=pole, inv_level=p4 + 0.3 * pole)
        if d0 == 1 and n >= 5:
            pole = p4 - p3
            hC = p2 - p3
            if pole > 0 and hC > 0 and pole >= 2.0 * hC and p1 > p3 and p0 < p2 and b3 - b4 <= 2 * (b0 - b3):
                return self._new_pattern(i, "Bearish Pennant", -1, True, 5,
                                         b3, p3, b1, p1, b2, p2, b0, p0,
                                         tgt_size=pole, inv_level=p4 - 0.3 * pole)
        # Wedges & Triangle
        if n >= 5:
            if d0 == 1:
                ux1, uy1, ux2, uy2 = b4, p4, b0, p0
                lx1, ly1, lx2, ly2 = b3, p3, b1, p1
            else:
                ux1, uy1, ux2, uy2 = b3, p3, b1, p1
                lx1, ly1, lx2, ly2 = b4, p4, b0, p0
            fit_err = abs(p2 - _val_at(b4, p4, b0, p0, b2))
            hS = _val_at(ux1, uy1, ux2, uy2, b4) - _val_at(lx1, ly1, lx2, ly2, b4)
            hE = _val_at(ux1, uy1, ux2, uy2, b0) - _val_at(lx1, ly1, lx2, ly2, b0)
            w = b0 - b4
            if hS > 0 and hE > 0 and hE <= 0.85 * hS and fit_err <= TOL * hS:
                sUn = (uy2 - uy1) / max(1, ux2 - ux1) * w / hS
                sLn = (ly2 - ly1) / max(1, lx2 - lx1) * w / hS
                ys5 = self.zzP[-5:]
                if sUn > 0.2 and sLn > 0.2:
                    return self._new_pattern(i, "Rising Wedge", -1, True, 5,
                                             lx1, ly1, lx2, ly2, ux1, uy1, ux2, uy2,
                                             tgt_size=hS, inv_level=max(ys5))
                if sUn < -0.2 and sLn < -0.2:
                    return self._new_pattern(i, "Falling Wedge", 1, True, 5,
                                             lx1, ly1, lx2, ly2, ux1, uy1, ux2, uy2,
                                             tgt_size=hS, inv_level=min(ys5))
                if sUn <= 0.2 and sLn >= -0.2:
                    return self._new_pattern(i, "Triangle", 0, True, 5,
                                             lx1, ly1, lx2, ly2, ux1, uy1, ux2, uy2,
                                             tgt_size=hS, inv_level=np.nan)
        return False

    def set_await(self, p: Pattern, dir_bk: int, i: int) -> None:
        p.dir = dir_bk
        p.status = ST_AWAIT
        p.bk_bar = i
        line_p = p.top_at(i) if (dir_bk > 0 and p.two_lines) else p.neck_at(i)
        p.bk_price = line_p
        p.target = line_p + (p.tgt_size if dir_bk > 0 else -p.tgt_size)
        if not np.isfinite(p.inv_level):
            p.inv_level = min(p.ys) if dir_bk > 0 else max(p.ys)

    def step_status(self, i: int, close: float, high: float, low: float) -> dict:
        ev = {"break_up": 0, "break_dn": 0, "reached": 0, "failed": 0}
        for p in self.pats:
            if p.status == ST_FORMING:
                upL = p.top_at(i) if p.two_lines else p.neck_at(i)
                dnL = p.neck_at(i)
                fail_now = (np.isfinite(p.inv_level) and p.dir != 0
                            and (close > p.inv_level if p.dir < 0 else close < p.inv_level))
                broke_up = (not fail_now) and p.dir >= 0 and close > upL
                broke_dn = (not fail_now) and p.dir <= 0 and close < dnL
                if broke_up or broke_dn:
                    self.set_await(p, 1 if broke_up else -1, i)
                    ev["break_up" if broke_up else "break_dn"] = 1
                elif fail_now or i > p.expiry_bar:
                    p.status = ST_FAILED
                    p.done_bar = i
                    ev["failed"] = 1
            elif p.status == ST_AWAIT:
                hit = high >= p.target if p.dir > 0 else low <= p.target
                if hit:
                    p.status = ST_REACHED
                    p.done_bar = i
                    ev["reached"] = 1
                elif np.isfinite(p.inv_level) and (
                        close < p.inv_level if p.dir > 0 else close > p.inv_level):
                    p.status = ST_FAILED
                    p.done_bar = i
                    ev["failed"] = 1
        return ev


def compute(df5: pd.DataFrame, atr5: pd.Series) -> pd.DataFrame:
    o = df5["open"].to_numpy(np.float64)
    h = df5["high"].to_numpy(np.float64)
    l = df5["low"].to_numpy(np.float64)
    c = df5["close"].to_numpy(np.float64)
    a = atr5.to_numpy(np.float64).clip(1e-9)
    n = len(c)
    ph_conf, pl_conf = confirmed_pivots(h, l)

    eng = _Engine()
    cols = {k: np.zeros(n, dtype=np.float32) for k in [
        "zz_last_dir", "zz_pivot_age", "zz_dist_pivot", "zz_leg_ret", "zz_leg_ratio",
        "zz_hh", "zz_hl", "pat_forming", "pat_awaiting", "pat_type", "pat_dir",
        "pat_age", "pat_height_atr", "pat_dist_neck", "pat_dist_target",
        "pat_break_up", "pat_break_dn", "pat_reached", "pat_failed"]}
    last_conf_bar = -1
    ev_age = {"break_up": 999, "break_dn": 999, "reached": 999, "failed": 999}

    for i in range(n):
        # pivot(s) confirmed at this bar refer to bar p = i - PIV_RIGHT
        zz_event = False
        p = i - PIV_RIGHT
        if ph_conf[i] and pl_conf[i]:
            if eng.zzD and eng.zzD[-1] == 1:
                zz_event |= eng.add_pivot(-1, l[p], p)
                zz_event |= eng.add_pivot(1, h[p], p)
            else:
                zz_event |= eng.add_pivot(1, h[p], p)
                zz_event |= eng.add_pivot(-1, l[p], p)
        elif ph_conf[i]:
            zz_event = eng.add_pivot(1, h[p], p)
        elif pl_conf[i]:
            zz_event = eng.add_pivot(-1, l[p], p)
        if zz_event:
            last_conf_bar = i
            eng.detect(i)

        ev = eng.step_status(i, c[i], h[i], l[i])
        for k in ev_age:
            ev_age[k] = 0 if ev[k] else min(ev_age[k] + 1, 999)

        # zigzag features
        if eng.zzD:
            cols["zz_last_dir"][i] = eng.zzD[-1]
            cols["zz_pivot_age"][i] = np.log1p(min(i - last_conf_bar, 500))
            cols["zz_dist_pivot"][i] = np.clip((c[i] - eng.zzP[-1]) / a[i], -20, 20)
            if len(eng.zzP) >= 2:
                leg = eng.zzP[-1] - eng.zzP[-2]
                cols["zz_leg_ret"][i] = np.clip(leg / a[i], -30, 30)
            if len(eng.zzP) >= 3:
                leg1 = abs(eng.zzP[-1] - eng.zzP[-2])
                leg2 = abs(eng.zzP[-2] - eng.zzP[-3])
                cols["zz_leg_ratio"][i] = np.clip(np.log(leg1 / max(leg2, 1e-9)), -3, 3)
            if len(eng.zzP) >= 4:
                # compare last two same-direction pivots: higher-high / higher-low state
                d_last = eng.zzD[-1]
                same = eng.zzP[-1] - eng.zzP[-3]
                other = eng.zzP[-2] - eng.zzP[-4]
                if d_last == 1:
                    cols["zz_hh"][i] = np.sign(same)
                    cols["zz_hl"][i] = np.sign(other)
                else:
                    cols["zz_hl"][i] = np.sign(same)
                    cols["zz_hh"][i] = np.sign(other)

        # pattern features: most relevant = latest AWAIT else latest FORMING
        best = None
        for q in reversed(eng.pats):
            if q.status == ST_AWAIT:
                best = q
                break
        if best is None:
            for q in reversed(eng.pats):
                if q.status == ST_FORMING:
                    best = q
                    break
        cols["pat_forming"][i] = float(any(q.status == ST_FORMING for q in eng.pats))
        cols["pat_awaiting"][i] = float(any(q.status == ST_AWAIT for q in eng.pats))
        if best is not None:
            cols["pat_type"][i] = PATTERN_CODES[best.name]
            cols["pat_dir"][i] = best.dir
            cols["pat_age"][i] = np.log1p(min(i - best.born, 500))
            cols["pat_height_atr"][i] = np.clip(best.tgt_size / a[i], 0, 30)
            cols["pat_dist_neck"][i] = np.clip((c[i] - best.neck_at(i)) / a[i], -20, 20)
            if best.status == ST_AWAIT and np.isfinite(best.target):
                cols["pat_dist_target"][i] = np.clip((best.target - c[i]) / a[i], -30, 30)
        for k, col in (("break_up", "pat_break_up"), ("break_dn", "pat_break_dn"),
                       ("reached", "pat_reached"), ("failed", "pat_failed")):
            age = ev_age[k]
            cols[col][i] = max(0.0, 1.0 - age / 3.0) if age <= 3 else 0.0

    out = pd.DataFrame(cols, index=df5.index)
    return out.astype("float32")
