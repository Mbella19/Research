# Data profile & synthetic realism report

## Source stats (5m returns)

| source | n_5m | std (bps) | skew | ex.kurt | acf|r| L1 | acf|r| L12 | p99.9/p50 | total ret |
|---|---|---|---|---|---|---|---|---|
| real_training | 380,200 | 9.5 | -0.16 | 64.9 | 0.410 | 0.333 | 25.8 | +143% |
| real_validation | 41,054 | 5.7 | -0.23 | 18.1 | 0.384 | 0.256 | 19.7 | +20% |
| synth_u1 | 421,317 | 8.2 | 0.09 | 83.1 | 0.399 | 0.295 | 31.6 | +435% |
| synth_u2 | 423,042 | 8.4 | -0.19 | 48.0 | 0.387 | 0.283 | 27.9 | +207% |
| synth_u3 | 422,279 | 9.3 | 0.92 | 127.6 | 0.402 | 0.295 | 32.5 | -37% |
| synth_u4 | 422,379 | 8.6 | 0.18 | 80.6 | 0.396 | 0.299 | 30.1 | +68% |
| synth_u5 | 423,086 | 7.7 | 0.13 | 53.4 | 0.385 | 0.268 | 27.3 | +201% |

## Realism verdicts vs real_training

| universe | ks_z5 | kurt_ratio | acf1_ratio | acf12_ratio | intraday_corr | overall |
|---|---|---|---|---|---|---|
| synth_u1 | 0.0191 PASS | 1.28 PASS | 0.97 PASS | 0.89 PASS | 0.995 PASS | **PASS** |
| synth_u2 | 0.0073 PASS | 0.74 PASS | 0.94 PASS | 0.85 PASS | 0.995 PASS | **PASS** |
| synth_u3 | 0.0207 PASS | 1.97 PASS | 0.98 PASS | 0.89 PASS | 0.994 PASS | **PASS** |
| synth_u4 | 0.0151 PASS | 1.24 PASS | 0.97 PASS | 0.9 PASS | 0.996 PASS | **PASS** |
| synth_u5 | 0.0099 PASS | 0.82 PASS | 0.94 PASS | 0.8 PASS | 0.996 PASS | **PASS** |

**Suggested synthetic pooling weight cap: w ≤ 1.0**

Synth may veto recipes but never select them (see decisions ledger D-002).