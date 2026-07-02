# Findings Log (append-only)

Running notebook of empirical findings, anomalies, and their resolutions.

## F-007 · 2026-07-02 · VALIDATION VERDICT: no deployable edge (VLOOK-01..03)
- Champion (GBT, tp3/sl1.5/H48, ev>0.20 raw-p gate): 72 validation trades,
  PF 0.83, expectancy −0.116R, −4.2% vs B&H +18.4%. Criteria 1/15.
- Threshold sweep: NO gate level viable (best: exact break-even +0.002R at ev>0.05,
  n=193). Failure uniform, not threshold-sensitive.
- Root cause (visible in training OOF blocks): the edge is concentrated in
  high-vol eras (2020 COVID blocks +92.8/+74.7, 2022 bear +63.9); the calm
  2024→2025-05 blocks were already flat/negative (+9.1/−9.0/−4.1). The signal
  is a high-volatility regime artifact that does not exist in the validation
  regime. Aggregate OOF netEV (+228) masked this concentration.
- Per pre-registered honesty clause: verdict recorded, validation not tortured
  further (ablation/sizing looks unspent — they cannot cure era-dependence),
  LOCKED OOS **preserved unburned** for a future qualifying recipe.

## F-006 · 2026-07-02 · Artifact-transfer defect found & fixed (VLOOK-01→02)
- VLOOK-01 gated 0 trades: isotonic-calibrated outputs were gated, but CV
  economics had been measured on RAW fold-model probabilities (policy mismatch);
  additionally the single refit model had tighter output spread than fold models.
- Fix: gate path = raw booster output quantile-mapped onto the OOF distribution
  (fit on training rows only); isotonic retained for reporting. Deployed policy
  now matches the validated one. Lesson encoded in models/pipeline.py.

## F-005 · 2026-07-02 · Synth pooling gate: real-only wins decisively (runs/lgbm_synth_*)
- Real-only, 5-fold purged OOF at tp3/sl1.5/H48, ev>0.20 gate: +0.188/tr on 1,220 trades,
  bootstrap LB95 +0.082 > 0 → D-002 gate PASSED. AUC 0.536/0.532.
- Pooling w∈{0.25,0.5,1.0}: AUC preserved (~0.535) but gated trades collapse to 66/37/34 —
  synthetic training compresses probability spread (conviction destroyed, discrimination kept).
- w_synth = 0. Synthetic stays as robustness screen/veto only. Empirically confirms the
  "stress test, not alpha truth" review stance (D-002).

## F-001 · 2026-07-02 · Data reconnaissance
- MT5 1m bars, broker time (UTC+2/+3), Mon–Fri 01:00–23:59, maintenance gap 23:59→01:00.
- Spread: 10–15 pts (1.0–1.5 index pts); widest overnight (15), tightest US hours (10), spike to 40 max.
- Spread ≈ 19% of mean 1m range → edge must live at multi-bar horizons.
- Activity (tickvol) peaks 16:00–18:00 broker (US open ≈16:30).
- VOL column all zeros → use TICKVOL.
- Synth universes: U1 +5.4×, U2 +3.1×, U3 −37% (bear), U4 +68%, U5 +3.0× over 6y — good regime spread.

## F-002 · 2026-07-02 · Ingest + profile (runs/ingest_20260702_034137, runs/profile_20260702_034400)
- All 8 sources CLEAN: 0 dupes, 0 OHLC violations, 0 off-session bars.
- Synth realism excellent on all 5 axes (KS ≤ 0.021, |r|-ACF ratios 0.8–1.0, intraday-profile corr ≥ 0.994,
  kurtosis in family range, near-zero signed-return ACF like real). Suggested pooling cap w ≤ 1.0.
- REGIME NOTE: validation (Jun–Dec 2025) is much calmer than training (5m vol 5.7 vs 9.5 bps,
  ex-kurt 18 vs 65). Training holds COVID + 2022 bear. Thresholds must not assume high vol.

## F-004 · 2026-07-02 · Rule benchmarks on TRAIN net of 1.25× costs (runs/benchmark_real_training_20260702_040557)
- MS.txt native entries: 1058 trades, PF 0.50, expectancy −0.43R, −90% equity, 2/41 months positive.
- Momentum breakout: 1026 trades, PF 0.52, expectancy −0.45R.
- B&H same period: +140%, maxDD 35.5%.
- Conclusion: naive TA entries are strongly anti-predictive at 2.0/1.2·ATR barriers under stressed costs.
  The indicators' value (if any) is as CONTEXT features, not raw entries. The ML filter must create the
  entire edge; beating B&H risk-adjusted is the bar. Cost realism ≈ 0.1–0.15R per round trip.

## F-003 · 2026-07-02 · Synth↔real leakage audit (runs/leakage_20260702_034539)
- 0/60 exact 2h return-windows from any universe found in TRAIN or VALIDATION (blake2b on rounded returns).
- Day-path max-corr vs validation: synth max 0.67 < null (train-vs-valid) max 0.72 → indistinguishable from null.
- Verdict: synthetic is genuinely generated (no copy-paste bootstrap), safe to pool per protocol gates;
  no synth-induced CV fold-leak risk. Locked OOS not touched (postdates synth creation; assumption logged).

## 2026-07-02 · S2 gate Monte Carlo null (circular rotation, training, zero looks)
All 1,339 circular rotations of the real SMA50 gate against the real
overnight-return vector (same exposure, same on/off clustering, alignment
randomized): true Sharpe 0.974 = 97.7th pct of null (mean 0.313, p=0.023);
true maxDD −15.4% vs null mean −25.6% — 99.6% of random alignments draw
deeper (p≈0.004). The gate's value is genuine regime timing, dominantly via
drawdown quarantine, not just being long 66% of a drifting market.
