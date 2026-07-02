"""Performance metrics, statistical significance, benchmark comparisons."""
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def compute_metrics(trades: pd.DataFrame, daily: pd.DataFrame, equity0: float) -> dict:
    m: dict = {"n_trades": int(len(trades))}
    if len(trades):
        r = trades["R"].to_numpy()
        wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
        m.update(
            profit_factor=float(wins / losses) if losses > 0 else np.inf,
            win_rate=float((trades["pnl"] > 0).mean()),
            expectancy_R=float(r.mean()),
            avg_win_R=float(r[r > 0].mean()) if (r > 0).any() else 0.0,
            avg_loss_R=float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
            total_pnl=float(trades["pnl"].sum()),
            commission_total=float(trades["commission"].sum()),
        )
        by_day = trades.groupby(trades["exit_ts"].dt.normalize())["pnl"].sum()
        by_month = trades.groupby(trades["exit_ts"].dt.to_period("M"))["pnl"].sum()
        total = trades["pnl"].sum()
        if total > 0:
            m["max_day_share"] = float(by_day.max() / total)
            m["max_month_share"] = float(by_month.max() / total)
        m["months_positive"] = int((by_month > 0).sum())
        m["months_total"] = int(len(by_month))
        top5 = trades["pnl"].nlargest(5).sum()
        m["pnl_without_top5"] = float(total - top5)
        m["trades_per_day"] = float(len(trades) / max(len(daily), 1))

    eq = daily["equity"]
    ret = daily["ret"]
    years = max(len(daily) / TRADING_DAYS, 1e-9)
    m["total_return_pct"] = float((eq.iloc[-1] / equity0 - 1) * 100)
    m["cagr_pct"] = float(((eq.iloc[-1] / equity0) ** (1 / years) - 1) * 100)
    dd = 1 - eq / eq.cummax()
    m["max_dd_pct"] = float(dd.max() * 100)
    sd = ret.std()
    m["sharpe"] = float(ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    downside = ret[ret < 0].std()
    m["sortino"] = float(ret.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
    m["calmar"] = float((m["cagr_pct"] / 100) / max(dd.max(), 1e-9))
    return m


def buy_and_hold(df1m: pd.DataFrame, equity0: float) -> tuple[pd.DataFrame, dict]:
    """Fully-invested B&H marked at daily last closes (the bar the AI must beat)."""
    ts = pd.DatetimeIndex(df1m["ts"])
    daily_close = df1m["close"].groupby(ts.normalize()).last()
    eq = equity0 * daily_close / daily_close.iloc[0]
    daily = pd.DataFrame({"equity": eq})
    daily["ret"] = daily["equity"].pct_change().fillna(0.0)
    return daily, compute_metrics(pd.DataFrame(), daily, equity0)


def bootstrap_lb(values: np.ndarray, q: float = 0.05, n_boot: int = 4000,
                 seed: int = 0) -> float:
    """Lower confidence bound of the mean by iid bootstrap."""
    if len(values) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, q))


def probabilistic_sharpe(ret: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & López de Prado): P(true SR > benchmark) given skew/kurtosis."""
    r = ret.to_numpy()
    n = len(r)
    if n < 30 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()                      # per-period SR
    g3 = pd.Series(r).skew()
    g4 = pd.Series(r).kurt() + 3
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe(ret: pd.Series, n_trials: int, trial_sr_var: float | None = None) -> float:
    """DSR: PSR against the expected max SR of n_trials random strategies."""
    r = ret.to_numpy()
    n = len(r)
    if n < 30 or r.std() == 0 or n_trials < 1:
        return np.nan
    sr_var = trial_sr_var if trial_sr_var is not None else 1.0 / (n - 1)
    emc = 0.5772156649
    maxZ = ((1 - emc) * norm.ppf(1 - 1.0 / n_trials)
            + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))) if n_trials > 1 else 0.0
    sr_star = np.sqrt(sr_var) * maxZ
    return probabilistic_sharpe(ret, sr_star)
