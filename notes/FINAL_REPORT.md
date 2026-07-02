# Final Report v3 — NAS100 Two-Sleeve Portfolio AI (rebuilt S1 + S2 risk package)

**Date:** 2026-07-02 · **Status: FROZEN (v3), DEPLOYABLE.** `models/FINAL_FROZEN_V3.json`.
v1/v2 artifacts untouched and auditable. Every decision, look, and refutation:
`notes/decisions.md` D-021…D-031, VLOOK-01…22 (cumulative DSR accounting).

## What v3 is

- **S1′ — intraday ML (rebuilt):** LightGBM champion (63 leaves / depth 9) on
  **142 lookahead-certified features** — v1's 113 plus three new causal blocks:
  *daily context* (yesterday's compression score, vol term structure, daily
  levels, gap stats, streaks), *calendar* (turn-of-month, opex week), *tape
  character* (session efficiency-so-far, autocorrelation, range expansion).
  Triple-barrier tp3/sl1.5/H48, drift-priced symmetric EV gate at 0.10, both
  sides, futures-stressed execution, 0.5% risk/trade, flat by EOD.
  **54% of its trades are SHORT (+0.089R short-side OOF)** — bear capability
  learned from regime features, not forced by rules.
- **S2 — gated overnight drift + risk package:** long 23:00→next 16:30 while
  yesterday's close > SMA50 (lagged), now with a **5×ATR catastrophe stop**
  (0 fires in 6.5y — pure tail insurance) and **one-sided vol de-risk**
  (de-levers in wild vol, never levers up; Sharpe 0.97→1.00, DD −15.4→−14.6).
- **Portfolio:** vol-parity budgets (w2 = 1.17), correlation ≈ 0.03.

## Out-of-sample record of the exact frozen policy

| set | Sharpe | return | maxDD | B&H | verdict |
|---|---|---|---|---|---|
| Training era, honest OOF (5.4y) | 1.15 | +134.9% (+16.6%/yr) | **−13.6%** | +140% / −35.5% | ~B&H return at 38% of its DD |
| **Validation (Jun–Dec 2025)** | **2.51** | **+18.6%** | **−7.5%** | +18.4% / −8.0% | **beats B&H, 7/8 criteria** |
| Tertiary (Jan–Jun 2026, 3rd view) | 1.14 | +7.0% | −5.1% | +15.6% / −12.4% | trails B&H; weak spot, see risks |

**Full 6.5y stitched: +198.2% (18.0%/yr), Sharpe 1.25, maxDD −13.6%** (v2:
+161%, −18.3%). Year-by-year: **2020 +35.7 · 2021 −1.2 · 2022 +25.9 · 2023
+1.1 · 2024 +36.4 · 2025 +19.8 · 2026H1 +7.0** — one negative year (−1.2%),
worst month −5.0%. v2's −7.3% 2022 became **+25.9%** (the model shorts bears
now); the "cooked in a downtrend" scenario is where v3 now earns most.

## v3 pre-registered bar — honest scoring (5 of 7)

PASS: DD ≤15 (−13.6) · worst year ≥ −3 (−1.2) · months+ ≥60 (60) · shorts
≥20% non-neg (54%, +0.089R) · validation Sharpe ≥1.2 & beats B&H (2.51, +18.6
vs +18.4). **MISS: OOF Sharpe 1.5 (achieved 1.15)** — single-instrument
ceiling; **MISS: 2021+2023 ≥ +6% (achieved −0.1%)** — v3 turned the flat
years from "slightly negative" to "flat"; after real costs the 2021-style
compression regime still yields ~nothing. Both misses reported, not hidden.

## What was tried and refuted in v3 (all pre-registered, all ledgered)

Regime-switched drift (whipsaws turns) · bear-overnight short sleeve (one-
event 2022 pattern, t=1.37) · tight S2 stop (still refuted; 5× insurance
adopted) · symmetric ivol sizing (vol-loving alpha) · **geometry tp2/sl1/H24**
(netEV −18.5) · **EV-regression head** (EV↔R corr 0.02, negative at every
gate) · **meta-label sizing** (primary already consumes the regime features)
· **TCN deep challenger, 2nd attempt, now with all 142 features** (netEV −854
on 9,530 trades — deep nets remain refuted at this data volume) · **v1+v3a
logit blend** (destroys both edges). The chop-breakout P5 finding entered
through features (d_er_pct etc.), which is where it proved to belong.

## Honest limitations (unchanged in kind, updated in detail)

1. **The 2026H1 tertiary is v3's weak window** (+7.0% vs B&H +15.6%; S1′
   Sharpe 0.49 there vs v1's 2.01). The rebuilt S1 shifted edge into
   bear/chop regimes and gave back some melt-up/selloff-rebound alpha. The
   window is thrice-seen and thin (67 S1 trades) — the paper trade decides.
2. Sharpe 2.5 (validation) is a regime-favorable number; ~1.15–1.25 is the
   through-the-cycle number. 2021/2023-style years remain ≈ flat.
3. No virgin test window exists; validation was the decisive gate and the
   **2–3 month MNQ paper trade is the only true cold OOS.** Go-live rules
   unchanged (S1 expectancy LB, S2 halt flag, per-sleeve divergence in
   `daytrader forward`).
4. Absolute B&H outperformance "by a lot" is sizing-dependent: at the frozen
   ~14% vol the 6.5y return trails B&H by 30pp at 1/2.6 the drawdown. The
   risk knob scales linearly; choose within DD tolerance BEFORE paper trade.

## Deployment

`daytrader signal --csv <MT5 export>` → S1′ order (entry cutoff enforced) +
S2 gate state with de-risk-scaled size and catastrophe stop level.
`daytrader forward --csv` → per-sleeve divergence + halt flags.
Retrain S1′ quarterly or on divergence; S2 rule needs no retraining.
New instrument: swap CSVs, edit instrument.yaml, rerun EVERYTHING including
the S2 grid and the v3 search — nothing transfers on faith.
