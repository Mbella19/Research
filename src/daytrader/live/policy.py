"""PolicyRuntime — one frozen policy's complete live decision surface.

Everything a decision needs is resolved HERE, once, at load time, from the
frozen artifacts (D-032): v3 verbatim from FINAL_FROZEN_V3.json; v2 from
FINAL_FROZEN_V2.json sleeves + the pre-registered assembled decision dict
(that file predates the decision block; the dict is exactly what FWD-01 ran).
`override_experiment` is applied so that every config-reading helper on the
inference path (drift constant, labels, timeframes) agrees with the policy —
one PolicyRuntime per PROCESS, never two policies in one process.

Formula sources (parity contract):
  s1_lots/levels  → backtest/engine.py:209-216, 171-182
  gate            → eval/validate.predict_gbt_from + backtest/signals.
                    from_probabilities + models/dataset.cost_atr/drift_atr
  S2 state/sizing → portfolio/overnight.gate_series/derisk_exposure/
                    _daily_atr + eval/live.py:105-107
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.signals import from_probabilities
from ..config import instrument, override_experiment
from ..eval.validate import predict_gbt_from
from ..models.dataset import cost_atr, drift_atr
from ..portfolio.book import _s2_params
from ..portfolio.overnight import (EXIT_MOD, S2Params, _daily_atr,
                                   derisk_exposure, gate_series)
from ..utils import paths
from ..utils.hashio import load_json
from ..utils.log import get_logger

log = get_logger("live.policy")

# D-032: assembled v2 decision (FINAL_FROZEN_V2.json has no decision block;
# these are v1's frozen gate values — identical to the FWD-01 override).
V2_DECISION = {
    "s1_artifact": "lgbm_final",
    "min_ev_atr": 0.20,
    "drift_mu_daily": 0.000636,
    "gate_cost_profile": "cfd_stressed",
    "allowed_sides": "both",
    "prob_floor": 0.40,
}
V2_LABELS = {"atr_period": 14, "tp_atr": 3.0, "sl_atr": 1.5,
             "horizon_bars": 48, "side": "both"}

GROUPS = {
    "v2": ["base", "time", "ms", "zz"],                       # 113 features
    "v3": ["base", "time", "ms", "zz", "daily", "cal", "tape"],  # 142
}
N_FEATURES = {"v2": 113, "v3": 142}

MAGICS = {"v2": (622001, 622002), "v3": (623001, 623002)}      # (S1, S2)

S2_ENTRY_MOD = 23 * 60          # overnight.py:145 — first 1m bar ≥ 23:00
S2_EXIT_MOD = EXIT_MOD          # 990 = 16:30


def _minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


@dataclass
class PolicyRuntime:
    name: str
    decision: dict
    sleeves: dict
    labels: dict
    groups: list[str]
    magic_s1: int
    magic_s2: int
    s2p: S2Params = field(default=None)
    # instrument/session constants
    point: float = 0.1
    digits: int = 1
    vpu: float = 1.0
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 50.0
    leverage_cap: float = 20.0
    risk_per_trade: float = 0.005
    no_entry_mod: int = 1365
    flat_mod: int = 1410

    @classmethod
    def load(cls, name: str) -> "PolicyRuntime":
        if name == "v3":
            fz = load_json(paths.MODELS_DIR / "FINAL_FROZEN_V3.json")
            decision, sleeves, labels = fz["decision"], fz["sleeves"], fz["labels"]
        elif name == "v2":
            fz = load_json(paths.MODELS_DIR / "FINAL_FROZEN_V2.json")
            decision, sleeves, labels = dict(V2_DECISION), fz["sleeves"], dict(V2_LABELS)
        else:
            raise ValueError(f"unknown policy {name!r}")

        # process-wide config agreement (one policy per process!)
        override_experiment(decision=dict(decision), sleeves=dict(sleeves),
                            labels=dict(labels),
                            features={"groups": list(GROUPS[name])})

        ins = instrument()
        from ..config import experiment
        bt = experiment()["backtest"]
        sess = ins["session"]
        p = cls(
            name=name, decision=decision, sleeves=sleeves, labels=labels,
            groups=list(GROUPS[name]),
            magic_s1=MAGICS[name][0], magic_s2=MAGICS[name][1],
            point=float(ins["point_size"]), digits=int(ins["digits"]),
            vpu=float(ins["value_per_price_unit_per_lot"]),
            lot_step=float(ins["lot_step"]), min_lot=float(ins["min_lot"]),
            max_lot=float(ins["max_lot"]),
            leverage_cap=float(bt["leverage_cap"]),
            risk_per_trade=float(sleeves.get("risk1", bt["risk_per_trade"])),
            no_entry_mod=_minutes(sess["no_entry_after"]),
            flat_mod=_minutes(sess["force_flat_at"]),
        )
        p.s2p = _s2_params(sleeves)
        art = paths.MODELS_DIR / decision["s1_artifact"]
        meta = load_json(art / "meta.json")
        n = len(meta["feature_names"])
        if n != N_FEATURES[name]:
            raise RuntimeError(f"{name}: artifact {decision['s1_artifact']} has "
                               f"{n} features, expected {N_FEATURES[name]}")
        log.info(f"policy {name}: art={decision['s1_artifact']} ({n} feats) "
                 f"gate={decision['min_ev_atr']} w2={sleeves['w2']} "
                 f"s2={p.s2p} magics=({p.magic_s1},{p.magic_s2})")
        return p

    # ── S1 ────────────────────────────────────────────────────────────────
    @property
    def art_path(self):
        return paths.MODELS_DIR / self.decision["s1_artifact"]

    def s1_signals(self, feat: pd.DataFrame,
                   tail_rows: int | None = None) -> pd.DataFrame:
        """Gate decisions for (the tail of) a feature frame — the exact
        backtest signal path. Rowwise-pure, so a tail slice is exact."""
        f = feat if tail_rows is None else feat.tail(tail_rows).reset_index(drop=True)
        p_l, p_s = predict_gbt_from(self.art_path, f)
        ca = cost_atr(f, profile=self.decision["gate_cost_profile"])
        return from_probabilities(
            f, np.nan_to_num(p_l), np.nan_to_num(p_s),
            self.labels["tp_atr"], self.labels["sl_atr"], ca,
            self.decision["min_ev_atr"], self.decision["prob_floor"],
            allowed_sides=self.decision.get("allowed_sides", "both"),
            drift_atr=drift_atr(f))

    def s1_lots(self, equity: float, atr_abs: float,
                entry_px_est: float) -> tuple[float, float]:
        """engine.py:209-216 verbatim. Returns (lots, risk_dollars);
        lots == 0.0 means sizing-floor skip (does NOT count as a trade)."""
        risk_price = self.labels["sl_atr"] * atr_abs
        risk_dollars = equity * self.risk_per_trade
        lots = risk_dollars / (risk_price * self.vpu)
        lev_cap = equity * self.leverage_cap / (entry_px_est * self.vpu)
        lots = min(lots, lev_cap, self.max_lot)
        lots = float(np.floor(lots / self.lot_step) * self.lot_step)
        if lots < self.min_lot:
            return 0.0, risk_dollars
        return lots, risk_dollars

    def s1_levels(self, fill_px: float, side: int,
                  atr_abs: float) -> tuple[float, float]:
        """SL/TP anchored to the ACTUAL fill (engine anchors at its modeled
        fill, engine.py:171-182), broker-rounded."""
        sl = fill_px - side * self.labels["sl_atr"] * atr_abs
        tp = fill_px + side * self.labels["tp_atr"] * atr_abs
        return round(sl, self.digits), round(tp, self.digits)

    def horizon_1m_bars(self) -> int:
        return int(self.labels["horizon_bars"]) * 5   # 48 × 5m = 240 1m bars

    # ── S2 ────────────────────────────────────────────────────────────────
    def s2_state(self, daily_ohlc: pd.DataFrame) -> dict:
        """Today's gate/expo/dailyATR from the FULL daily history (today's
        partial row present and harmless — every series is ≤ D−1 lagged)."""
        gate = bool(gate_series(daily_ohlc, self.s2p).iloc[-1])
        expo = (float(derisk_exposure(daily_ohlc).iloc[-1])
                if self.s2p.derisk else 1.0)
        datr = float(_daily_atr(daily_ohlc).iloc[-1])
        return {"gate": gate, "expo": expo, "datr": datr}

    def s2_lots(self, equity: float, close: float, expo: float) -> float:
        """live.py:105-107 verbatim: notional sizing, floored to lot_step."""
        w2 = float(self.sleeves["w2"])
        return float(np.floor((expo * w2 * equity / (close * self.vpu))
                              / self.lot_step) * self.lot_step)

    def s2_stop_level(self, fill_px: float, datr: float) -> float | None:
        if not self.s2p.stop_atr or not np.isfinite(datr):
            return None
        return round(fill_px - self.s2p.stop_atr * datr, self.digits)
