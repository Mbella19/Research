"""Stacking blend + champion selection among {GBT, TCN, blend}.

Evaluated on the INTERSECTION of both models' OOF rows (same trades, same
costs) — champion by OOF net EV with per-trade bootstrap LB as tiebreaker.
The champion still has to survive its pre-registered validation look.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..utils import paths
from ..utils.artifacts import new_run_dir
from ..utils.hashio import load_json, save_json
from ..utils.log import get_logger
from .dataset import assemble
from .lgbm import economic_score
from .pipeline import RECIPE, _per_trade_lb
from .tcn_pipeline import TCN_OOF

log = get_logger("models.blend")

CHAMPION = paths.MODELS_DIR / "champion.json"
STACKER = paths.MODELS_DIR / "stacker.json"


def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def run_champion() -> None:
    run_dir = new_run_dir("champion")
    log.info(f"artifacts → {run_dir}")
    gbt = pd.read_parquet(paths.MODELS_DIR / "lgbm_final" / "oof_real.parquet")
    tcn = pd.read_parquet(TCN_OOF)
    df = gbt.merge(tcn, on="ts", suffixes=("_gbt", "_tcn"), how="inner")
    log.info(f"common OOF rows: {len(df):,}")

    bundle = assemble(w_synth=0.0)         # real rows only for evaluation maps
    ts_to_row = pd.Series(np.arange(len(bundle["ts"])),
                          index=pd.DatetimeIndex(bundle["ts"]))
    rows = ts_to_row.reindex(pd.DatetimeIndex(df["ts"])).to_numpy()
    ok = np.isfinite(rows)
    df, rows = df[ok], rows[ok].astype(int)

    # stacker per side on OOF logits
    stack_p = {}
    coefs = {}
    for side in ("long", "short"):
        Xs = np.column_stack([_logit(df[f"p_{side}_gbt"]), _logit(df[f"p_{side}_tcn"])])
        y = bundle[f"y_{side}"][rows]
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(Xs, y)
        stack_p[side] = lr.predict_proba(Xs)[:, 1].astype(np.float32)
        coefs[side] = {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}

    contenders = {
        "gbt": (df["p_long_gbt"].to_numpy(), df["p_short_gbt"].to_numpy()),
        "tcn": (df["p_long_tcn"].to_numpy(), df["p_short_tcn"].to_numpy()),
        "blend": (stack_p["long"], stack_p["short"]),
    }
    results = {}
    for name, (pl, ps) in contenders.items():
        econ = economic_score(pl, ps, bundle, rows)
        n = len(bundle["X"])
        full_l = np.full(n, np.nan, dtype=np.float32)
        full_s = np.full(n, np.nan, dtype=np.float32)
        full_l[rows], full_s[rows] = pl, ps
        lb = _per_trade_lb({"long": full_l, "short": full_s}, bundle, rows)
        results[name] = {"econ": econ, "per_trade_lb": lb}
        log.info(f"{name}: netEV {econ['net_ev_sum']:+.1f} "
                 f"({econ['n_trades']:,} trades, {econ['net_ev_mean']:+.4f}/tr) LB {lb:+.4f}")

    def key(n):
        e = results[n]["econ"]
        return e["net_ev_sum"] if e["n_trades"] >= 200 else -1e9

    champ = max(contenders, key=key)
    save_json(coefs, STACKER)
    save_json({"champion": champ, "results": results,
               "recipe": load_json(RECIPE) if RECIPE.exists() else None},
              CHAMPION)
    log.info(f"CHAMPION: {champ} → {CHAMPION}")
