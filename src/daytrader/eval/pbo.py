"""Probability of Backtest Overfitting via CSCV (Bailey et al.).

Input: score matrix M[config, time_block]. Split blocks into IS/OOS halves
across combinations; PBO = fraction of splits where the best-IS config lands
in the bottom half OOS.
"""
from itertools import combinations

import numpy as np


def pbo_cscv(M: np.ndarray, max_combos: int = 126, seed: int = 0) -> dict:
    n_cfg, n_blk = M.shape
    if n_cfg < 3 or n_blk < 4 or n_blk % 2 != 0:
        return {"pbo": np.nan, "n_combos": 0}
    combos = list(combinations(range(n_blk), n_blk // 2))
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    below = 0
    logits = []
    for c in combos:
        is_idx = np.array(c)
        oos_idx = np.setdiff1d(np.arange(n_blk), is_idx)
        is_score = M[:, is_idx].sum(axis=1)
        oos_score = M[:, oos_idx].sum(axis=1)
        star = int(np.argmax(is_score))
        # relative OOS rank of the IS winner
        rank = (oos_score < oos_score[star]).sum() / (n_cfg - 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
        if rank < 0.5:
            below += 1
    return {"pbo": below / len(combos), "n_combos": len(combos),
            "mean_logit": float(np.mean(logits))}
