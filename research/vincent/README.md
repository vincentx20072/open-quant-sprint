# Track 10: does market co-movement change when simple signals work?

This is Vincent's contribution to Open Quant Sprint 01, Track 10 ([Issue #16](https://github.com/valiyevoktay-cmd/open-quant-sprint/issues/16)). Both parts were reviewed and merged into the sprint repository:
- Part 1 (the absorption ratio): [PR #38](https://github.com/valiyevoktay-cmd/open-quant-sprint/pull/38)
- Part 2 (the measurement): [PR #51](https://github.com/valiyevoktay-cmd/open-quant-sprint/pull/51). The pre-registration was posted in this PR before any result was computed.

## Question

Some hours the 20 coins in the panel move almost as one asset. Other hours they move more independently. The **absorption ratio** (Kritzman, Li, Page and Rigobon, "Principal Components as a Measure of Systemic Risk", *Journal of Portfolio Management*, 2011) measures this: the share of total return variance explained by the top `k` principal components of a rolling covariance matrix.

A cross-sectional signal ranks coins against each other, so it can only use the *coin-specific* part of each coin's move. **Does the rank IC of simple cross-sectional signals differ between high- and low-absorption hours?**

**Hypothesis (pre-registered).** When one factor dominates there is less symbol-specific movement to rank, so the IC should be closer to zero in high-absorption hours.

## The absorption ratio

`absorption_ratio(df, window=168, k=1)` in [`src/contrib/vincent.py`](../../src/contrib/vincent.py):
1. builds hourly log returns as an hours × symbols matrix on a full hourly grid;
2. at each hour t, takes the sample covariance of the last `window` returns up to and including t;
3. returns the sum of its `k` largest eigenvalues divided by the sum of all eigenvalues.

The value is the same for all 20 symbols. It is NaN while the window contains a missing return, and it is strictly causal.

| | Paper | Here |
|---|---|---|
| Data | daily stock and industry returns | hourly returns, 20 Binance USD-M perpetuals |
| Window | 500 days | 168 hours (1 week) |
| Weighting | exponential, half-life 250 days | equal weights |
| Eigenvectors | about 1/5 of the number of assets | k = 1 (primary) and k = 4 (= 20/5) |

## Method (pre-registered)

All of these choices were posted in PR #51 before the first run on the full panel.

| Choice | Value |
|---|---|
| Data | `data/train_1h.parquet`: 20 symbols, 14,871 hours (2025-01-01 to 2026-09-12 UTC), no missing hours |
| Signals | **MOM**: the raw 24h momentum signal of `strategies/template.py` (`pct_change(24) / parkinson_volatility(24)`, before z-score, EWM and deadband)<br>**REV**: minus the last hour's return<br>**FLOW**: the weights of `strategies/baseline_flow.py` |
| Forward return | next hour's log return on the full hourly grid, computed on the full panel before any split (log and simple returns rank identically, so the IC is unchanged) |
| Rank IC | per hour, the Spearman correlation across symbols; hours with fewer than 10 valid symbols dropped |
| Regimes | AR terciles with cutoffs fixed on the first 40 % of hours (to 2025-09-05 19:00 UTC); comparison on the remaining 60 % (8,923 hours) |
| Statistic | mean IC(high AR) − mean IC(low AR) |
| Interval | 95 % percentile interval and two-sided p-value from a block bootstrap over calendar days (10,000 draws, seed 0); 7-day blocks as a sensitivity check |
| Multiple testing | Holm over 6 tests (3 signals × k ∈ {1, 4}) |
| Primary comparison | MOM, k = 1 |
| Strategy rule | write a strategy only if MOM or REV at k = 1 has Holm p < 0.05 |

## Results

The cutoffs come from the first 40 %:
- k = 1: low ≤ 0.7628, high > 0.8069
- k = 4: low ≤ 0.8752, high > 0.9028

| Signal | k | Mean IC, high AR | Mean IC, low AR | Diff (high − low) | 95 % CI (1-day blocks) | p | p (Holm) | 95 % CI (7-day blocks) | Hours high / low |
|---|---|---|---|---|---|---|---|---|---|
| **MOM (primary)** | 1 | −0.0263 | −0.0085 | −0.0177 | [−0.0318, −0.0037] | 0.012 | 0.072 | [−0.0315, −0.0017] | 2066 / 5724 |
| REV | 1 | +0.0294 | +0.0289 | +0.0005 | [−0.0145, +0.0154] | 0.947 | 0.947 | [−0.0136, +0.0163] | 2066 / 5724 |
| FLOW | 1 | −0.0158 | −0.0030 | −0.0128 | [−0.0259, −0.0002] | 0.052 | 0.259 | [−0.0241, −0.0013] | 2066 / 5724 |
| MOM | 4 | −0.0212 | −0.0093 | −0.0120 | [−0.0261, +0.0024] | 0.099 | 0.396 | [−0.0288, +0.0047] | 2435 / 4466 |
| REV | 4 | +0.0219 | +0.0285 | −0.0066 | [−0.0216, +0.0081] | 0.382 | 0.763 | [−0.0233, +0.0105] | 2435 / 4466 |
| FLOW | 4 | −0.0127 | −0.0026 | −0.0100 | [−0.0224, +0.0024] | 0.117 | 0.396 | [−0.0210, +0.0015] | 2435 / 4466 |

Mean IC over the whole test period:
- MOM −0.0151
- REV +0.0297
- FLOW −0.0060

Mean IC in the middle third (k = 1):
- MOM −0.0278
- REV +0.0347
- FLOW −0.0031

**Strategy.** The best Holm-adjusted p for MOM or REV at k = 1 is 0.072, so under the pre-registered rule no strategy was written.

## Findings

This track tested whether three signals (momentum, reversal and flow) were able to predict next-hour returns differently depending on how tightly the 20 symbols move together. This was measured by the absorption ratio, which is the variance share of the first principal component. I expected the rank IC to be closer to zero when one factor dominates because there is less symbol-specific movement to rank.

However, no comparison is significant after the Holm correction. One-hour reversal has the highest mean IC (+0.030) and no regime dependence. The primary comparison, 24-hour momentum at k = 1, comes closest. Its IC is negative (−0.015), so recent 24-hour outperformers tend to underperform over the next hour. This is stronger in high- than in low-absorption hours (−0.026 versus −0.009). The 95% interval for the difference excludes zero under 1-day and 7-day blocks, but the Holm-adjusted p-value is 0.072. The flow weights show the same pattern more weakly and both differences shrink at k = 4.

In conclusion, the absorption ratio does not significantly change these signals' IC, with the closest result being against my hypothesis. A plausible reason is that when the symbols move together, recent 24-hour winners get pulled back toward the group more strongly.

## Assumptions

1. "First 40 %" means by hour count: 5,948 of 14,871 hours. The cutoffs ignore the first 168 hours, where the AR is NaN, and are computed separately for each k.
2. MOM is the raw template signal, not the template's final weights. The Issue left this open, so it was stated in the pre-registration.
3. Log instead of simple returns. This changes no rank IC, because ranks are unchanged.
4. Spearman ties use average ranks.
5. Bootstrap blocks are UTC calendar days. The p-value is two-sided, from the bootstrap distribution centred on the estimate.
6. Holm includes the primary comparison. That is the strict convention: on its own, the primary comparison's p would be 0.012.
7. IC measures predictive accuracy, not profit after fees.

## Strengths and limitations

**Strengths.**
- The analysis was pre-registered publicly before any result.
- The cutoffs and the comparison use separate periods.
- The block bootstrap respects the dependence between neighbouring hours.
- The multiple-testing correction is applied.

**Limitations.**
- The effects are small (|IC| ≈ 0.01–0.03).
- The momentum result is borderline.
- The test period has few high-AR hours (2,066 vs 5,724), so the market was less coupled than in the cutoff period.
- There is one dataset and one horizon.
- The "pulled back toward the group" explanation is untested.

**Follow-ups.**
- Pre-register "momentum's negative IC is stronger in high-AR hours" and test it on new data.
- Check other horizons.
- Try the paper's exponential weighting and its "standardized shift" of the absorption ratio.

## Code and reproduction

Functions in [`src/contrib/vincent.py`](../../src/contrib/vincent.py), merged into the sprint repository:

| Function | Purpose |
|---|---|
| `to_wide`, `hourly_log_returns` | hours × symbols tables on a full hourly grid, shared by Part 1 and Part 2 |
| `absorption_ratio` | Part 1 |
| `forward_returns` | next hour's return at each hour (NaN if that hour is missing) |
| `rank_ic_by_hour` | per-hour Spearman IC across symbols |
| `regime_cutoffs`, `label_regimes` | terciles fixed on the first 40 %, and low/mid/high labels |
| `day_block_bootstrap` | the high − low difference with a day-block bootstrap interval and p-value |
| `holm` | Holm-adjusted p-values |

Tests are in [`tests/contrib/test_vincent.py`](../../tests/contrib/test_vincent.py) (11 tests).

The analysis script [`part2_ic_by_absorption.py`](part2_ic_by_absorption.py) writes [`results/part2_results.csv`](results/part2_results.csv). To reproduce, from the repository root (needs numpy, pandas and pyarrow):

```
python research/vincent/part2_ic_by_absorption.py
```

It runs in a few seconds and reproduces the table above exactly.

## Glossary

- **Absorption ratio:** the share of variance explained by the top k principal components. High means the coins move together.
- **Rank IC:** the Spearman correlation between a signal and next-hour returns across symbols. +1 is a perfect ranking, 0 is no information, and −1 is perfectly reversed.
- **Regime:** a third of the hours sorted by absorption ratio, with cutoffs fixed in advance.
- **Block bootstrap:** resampling whole days with replacement, so that neighbouring hours stay together.
- **Holm correction:** adjusts p-values for running several tests, keeping the overall false-positive rate at 5 %.
- **Pre-registration:** publishing the analysis plan before seeing the results.
