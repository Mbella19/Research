"""Session profiling + synthetic realism report (stylized facts vs real).

Profiles real TRAINING/VALIDATION and every synthetic universe; the locked
OOS set is never profiled. The realism verdicts cap the synthetic pooling
weight that later stages are allowed to consider (synth can veto, never select).
"""
import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..config import synth_sources
from ..data.loader import load_bars
from ..utils.artifacts import new_run_dir
from ..utils.hashio import save_json
from ..utils.log import get_logger
from .plotstyle import CRITICAL, GOOD, MUTED, SERIES, SYNTH_FAMILY, WARNING, apply_style

log = get_logger("eval.profile")

ABS_ACF_LAGS = [1, 2, 3, 6, 12, 24, 48, 96]
RET_ACF_LAGS = list(range(1, 11))


def _acf(x: np.ndarray, lags: list[int]) -> dict[int, float]:
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = float((x * x).sum())
    return {k: float((x[:-k] * x[k:]).sum() / denom) for k in lags}


def _source_stats(source: str) -> dict:
    df = load_bars(source)
    ts = pd.DatetimeIndex(df["ts"])
    day = ts.normalize().to_numpy()
    close64 = df["close"].to_numpy(np.float64)

    # 1m log returns within the same day (overnight/maintenance gaps excluded)
    r1 = np.diff(np.log(close64), prepend=np.nan)
    r1[day != np.roll(day, 1)] = np.nan

    hours = ts.hour.to_numpy()
    prof = (
        pd.DataFrame({"h": hours, "absr1": np.abs(r1) * 1e4,
                      "tickvol": df["tickvol"].to_numpy(np.float64),
                      "spread": df["spread_pts"].to_numpy(np.float64)})
        .groupby("h").mean()
    )

    # 5m bars for stylized facts
    c5 = pd.Series(close64, index=ts).resample("5min").last().dropna()
    r5 = np.diff(np.log(c5.to_numpy(np.float64)), prepend=np.nan)
    days5 = c5.index.normalize().to_numpy()
    r5[days5 != np.roll(days5, 1)] = np.nan
    r5 = r5[np.isfinite(r5)]

    z5 = (r5 - r5.mean()) / r5.std()
    rng = np.random.default_rng(0)
    sample = rng.choice(z5, size=min(150_000, len(z5)), replace=False)

    daily_close = pd.Series(close64, index=ts).groupby(ts.normalize()).last()

    stats = {
        "source": source,
        "n_5m": int(len(r5)),
        "std_5m_bps": float(r5.std() * 1e4),
        "skew": float(pd.Series(r5).skew()),
        "ex_kurtosis": float(pd.Series(r5).kurt()),
        "total_return_pct": float((close64[-1] / close64[0] - 1) * 100),
        "acf_r": _acf(r5, RET_ACF_LAGS),
        "acf_absr": _acf(np.abs(r5), ABS_ACF_LAGS),
        "p999_over_p50_absr": float(
            np.quantile(np.abs(r5), 0.999) / np.quantile(np.abs(r5), 0.5)
        ),
    }
    out = {"stats": stats, "prof": prof, "sample_z5": sample, "daily_close": daily_close}
    del df, ts, close64, r1, c5, r5, z5
    gc.collect()
    return out


def _realism_verdict(real: dict, synth: dict) -> dict:
    """Compare one synth universe against real training. PASS/WARN/FAIL per axis."""
    ks = float(ks_2samp(real["sample_z5"], synth["sample_z5"]).statistic)
    kurt_ratio = synth["stats"]["ex_kurtosis"] / max(real["stats"]["ex_kurtosis"], 1e-9)
    a_r, a_s = real["stats"]["acf_absr"], synth["stats"]["acf_absr"]
    acf1_ratio = a_s[1] / max(a_r[1], 1e-9)
    acf12_ratio = a_s[12] / max(a_r[12], 1e-9)
    pr, ps = real["prof"]["absr1"], synth["prof"]["absr1"]
    common = pr.index.intersection(ps.index)
    intraday_corr = float(np.corrcoef(pr[common], ps[common])[0, 1])

    def grade(val, ok, warn):
        return "PASS" if ok(val) else ("WARN" if warn(val) else "FAIL")

    axes = {
        "ks_z5": (round(ks, 4), grade(ks, lambda v: v < 0.05, lambda v: v < 0.12)),
        "kurt_ratio": (round(kurt_ratio, 2),
                       grade(kurt_ratio, lambda v: 0.4 <= v <= 2.5, lambda v: 0.2 <= v <= 4.0)),
        "acf_absr_lag1_ratio": (round(acf1_ratio, 2),
                                grade(acf1_ratio, lambda v: 0.5 <= v <= 2.0, lambda v: 0.3 <= v <= 3.0)),
        "acf_absr_lag12_ratio": (round(acf12_ratio, 2),
                                 grade(acf12_ratio, lambda v: 0.4 <= v <= 2.5, lambda v: 0.2 <= v <= 4.0)),
        "intraday_corr": (round(intraday_corr, 3),
                          grade(intraday_corr, lambda v: v >= 0.8, lambda v: v >= 0.5)),
    }
    grades = [g for _, g in axes.values()]
    overall = "FAIL" if "FAIL" in grades else ("WARN" if "WARN" in grades else "PASS")
    return {"axes": axes, "overall": overall}


def run_profile() -> None:
    apply_style()
    run_dir = new_run_dir("profile")
    log.info(f"artifacts → {run_dir}")

    sources = ["real_training", "real_validation"] + list(synth_sources().keys())
    results: dict[str, dict] = {}
    for s in sources:
        log.info(f"profiling {s} …")
        results[s] = _source_stats(s)

    real = results["real_training"]
    synth_names = [s for s in sources if s.startswith("synth")]
    verdicts = {s: _realism_verdict(real, results[s]) for s in synth_names}

    # suggested synth weight cap from worst overall verdict
    overall = [v["overall"] for v in verdicts.values()]
    w_cap = 1.0 if all(o == "PASS" for o in overall) else (0.5 if "FAIL" not in overall else 0.25)

    # ── plot 1: close overviews ──────────────────────────────────────
    fig, axes_ = plt.subplots(2, 4, figsize=(14, 6))
    for ax, s in zip(axes_.flat, sources):
        dc = results[s]["daily_close"]
        col = SERIES[0] if s == "real_training" else SERIES[1] if s == "real_validation" else SYNTH_FAMILY
        ax.plot(dc.index, dc.values, color=col, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(f"{s}  ({results[s]['stats']['total_return_pct']:+.0f}%)")
        ax.tick_params(axis="x", labelrotation=45)
    for ax in axes_.flat[len(sources):]:
        ax.axis("off")
    fig.suptitle("Daily closes (log scale)")
    fig.savefig(run_dir / "plots" / "overview_closes.png")
    plt.close(fig)

    # ── plot 2: intraday profiles ────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("absr1", "mean |1m return| (bps)"), ("tickvol", "mean tick volume"),
              ("spread", "mean spread (points)")]
    for ax, (colname, title) in zip(axs, panels):
        for s in synth_names:
            p = results[s]["prof"][colname]
            ax.plot(p.index, p.values, color=SYNTH_FAMILY, linewidth=0.9)
        pv = results["real_validation"]["prof"][colname]
        pt = results["real_training"]["prof"][colname]
        ax.plot(pv.index, pv.values, color=SERIES[1], linewidth=1.8)
        ax.plot(pt.index, pt.values, color=SERIES[0], linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("hour (broker time)")
        if colname == "absr1":
            ax.annotate("real train", xy=(pt.index[-4], pt.iloc[-4]), color=SERIES[0],
                        fontsize=8, xytext=(0, 8), textcoords="offset points")
            ax.annotate("real valid", xy=(pv.index[-8], pv.iloc[-8]), color=SERIES[1],
                        fontsize=8, xytext=(0, -12), textcoords="offset points")
            ax.annotate("synth U1–U5", xy=(6, results["synth_u1"]["prof"][colname].loc[6]),
                        color=MUTED, fontsize=8)
    fig.suptitle("Intraday profiles — real vs synthetic")
    fig.savefig(run_dir / "plots" / "intraday_profiles.png")
    plt.close(fig)

    # ── plot 3: stylized facts ───────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    for s in synth_names:
        a = results[s]["stats"]["acf_absr"]
        ax.plot(list(a.keys()), list(a.values()), color=SYNTH_FAMILY, linewidth=0.9)
    a = real["stats"]["acf_absr"]
    ax.plot(list(a.keys()), list(a.values()), color=SERIES[0], linewidth=2.2)
    av = results["real_validation"]["stats"]["acf_absr"]
    ax.plot(list(av.keys()), list(av.values()), color=SERIES[1], linewidth=1.8)
    ax.set_title("ACF of |5m returns| (vol clustering)")
    ax.set_xlabel("lag (5m bars)")

    ax = axs[0, 1]
    bins = np.linspace(-8, 8, 161)
    hist_r, _ = np.histogram(real["sample_z5"], bins=bins, density=True)
    pooled = np.concatenate([results[s]["sample_z5"] for s in synth_names])
    hist_s, _ = np.histogram(pooled, bins=bins, density=True)
    mid = (bins[:-1] + bins[1:]) / 2
    ax.semilogy(mid, hist_r, color=SERIES[0], linewidth=2.0, label="real train")
    ax.semilogy(mid, hist_s, color=SYNTH_FAMILY, linewidth=1.6, label="synth pooled")
    ax.semilogy(mid, np.exp(-mid**2 / 2) / np.sqrt(2 * np.pi), color=MUTED,
                linewidth=1.0, linestyle="--", label="N(0,1)")
    ax.set_title("standardized 5m return density")
    ax.legend()

    ax = axs[1, 0]
    names = sources
    kurts = [results[s]["stats"]["ex_kurtosis"] for s in names]
    cols = [SERIES[0] if s == "real_training" else SERIES[1] if s == "real_validation"
            else SYNTH_FAMILY for s in names]
    ax.bar(range(len(names)), kurts, color=cols, width=0.62)
    ax.set_xticks(range(len(names)), [n.replace("real_", "r.").replace("synth_", "s.")
                                      for n in names], rotation=30)
    ax.set_title("excess kurtosis of 5m returns")

    ax = axs[1, 1]
    for s in synth_names:
        a = results[s]["stats"]["acf_r"]
        ax.plot(list(a.keys()), list(a.values()), color=SYNTH_FAMILY, linewidth=0.9)
    a = real["stats"]["acf_r"]
    ax.plot(list(a.keys()), list(a.values()), color=SERIES[0], linewidth=2.2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("ACF of signed 5m returns")
    ax.set_xlabel("lag (5m bars)")
    fig.suptitle("Stylized facts — real vs synthetic")
    fig.savefig(run_dir / "plots" / "stylized_facts.png")
    plt.close(fig)

    # ── report ───────────────────────────────────────────────────────
    lines = ["# Data profile & synthetic realism report", "",
             "## Source stats (5m returns)", "",
             "| source | n_5m | std (bps) | skew | ex.kurt | acf|r| L1 | acf|r| L12 | p99.9/p50 | total ret |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in sources:
        st = results[s]["stats"]
        lines.append(
            f"| {s} | {st['n_5m']:,} | {st['std_5m_bps']:.1f} | {st['skew']:.2f} "
            f"| {st['ex_kurtosis']:.1f} | {st['acf_absr'][1]:.3f} | {st['acf_absr'][12]:.3f} "
            f"| {st['p999_over_p50_absr']:.1f} | {st['total_return_pct']:+.0f}% |")
    lines += ["", "## Realism verdicts vs real_training", "",
              "| universe | ks_z5 | kurt_ratio | acf1_ratio | acf12_ratio | intraday_corr | overall |",
              "|---|---|---|---|---|---|---|"]
    for s, v in verdicts.items():
        ax_ = v["axes"]
        row = " | ".join(f"{val} {g}" for val, g in ax_.values())
        lines.append(f"| {s} | {row} | **{v['overall']}** |")
    lines += ["", f"**Suggested synthetic pooling weight cap: w ≤ {w_cap}**",
              "", "Synth may veto recipes but never select them (see decisions ledger D-002)."]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save_json({"verdicts": {s: {k: v for k, v in vv["axes"].items()} for s, vv in verdicts.items()},
               "w_cap": w_cap,
               "stats": {s: results[s]["stats"] for s in sources}},
              run_dir / "profile.json")
    log.info(f"report → {run_dir / 'report.md'} | suggested synth w cap = {w_cap}")
