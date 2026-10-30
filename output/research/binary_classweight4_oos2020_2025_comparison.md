# XGBoost Binary Class Weight 4:1 — OOS 2020–2025

## Protocol

- Each year uses 47 training months, one purge month, and the prior year for validation.
- Factor v4; no refit; Top15/Hold30; alpha 1.0; inverse-volatility weighting.
- Class weights affect training loss only; validation and OOS metrics remain unweighted.

## Six-year stitched results

| Variant | CAGR | Sharpe | Sortino | MDD | Mean turnover | Best Sharpe years |
|---|---:|---:|---:|---:|---:|---:|
| unweighted | 7.53% | 0.701 | 0.942 | -17.70% | 5.78 | 3 |
| classweight4_fixedparams | 6.33% | 0.588 | 0.787 | -18.65% | 5.86 | 0 |
| classweight4_optuna | 6.00% | 0.565 | 0.743 | -19.91% | 5.67 | 3 |

## Conclusion

- Highest stitched Sharpe: **unweighted** (0.701).
- Fixed 4:1 weighting changed stitched CAGR by -1.20% and Sharpe by -0.114 versus unweighted.
- Weighted Optuna changed stitched CAGR by -1.53% and Sharpe by -0.136 versus unweighted.
- The fixed-parameter arm isolates class weighting; the Optuna arm also includes parameter reselection.
- Continuous Rank IC and Top15 excess return remain the checks against a portfolio-only result.
- 2025-08-01 contains only 323 local price rows, so July-end execution has incomplete coverage.

## Annual audit

### 2020
- unweighted: CAGR 0.65%, Sharpe 0.116, MDD -14.78%, turnover 8.40.
- classweight4_fixedparams: CAGR -1.19%, Sharpe -0.008, MDD -18.10%, turnover 9.32.
- classweight4_optuna: CAGR 2.53%, Sharpe 0.244, MDD -17.53%, turnover 9.53.

### 2021
- unweighted: CAGR 15.66%, Sharpe 1.628, MDD -6.65%, turnover 4.16.
- classweight4_fixedparams: CAGR 10.09%, Sharpe 1.143, MDD -6.61%, turnover 3.45.
- classweight4_optuna: CAGR 8.19%, Sharpe 0.897, MDD -7.38%, turnover 3.83.

### 2022
- unweighted: CAGR 1.30%, Sharpe 0.184, MDD -7.90%, turnover 4.00.
- classweight4_fixedparams: CAGR 0.91%, Sharpe 0.139, MDD -9.71%, turnover 4.46.
- classweight4_optuna: CAGR -5.01%, Sharpe -0.462, MDD -13.32%, turnover 4.22.

### 2023
- unweighted: CAGR 12.55%, Sharpe 1.354, MDD -7.92%, turnover 5.64.
- classweight4_fixedparams: CAGR 17.54%, Sharpe 1.707, MDD -6.17%, turnover 5.53.
- classweight4_optuna: CAGR 16.91%, Sharpe 1.798, MDD -6.25%, turnover 5.09.

### 2024
- unweighted: CAGR 5.21%, Sharpe 0.480, MDD -10.52%, turnover 6.72.
- classweight4_fixedparams: CAGR 1.22%, Sharpe 0.164, MDD -9.79%, turnover 6.75.
- classweight4_optuna: CAGR 2.35%, Sharpe 0.285, MDD -6.72%, turnover 5.11.

### 2025
- unweighted: CAGR 10.50%, Sharpe 0.850, MDD -16.03%, turnover 5.74.
- classweight4_fixedparams: CAGR 10.23%, Sharpe 0.791, MDD -17.70%, turnover 5.66.
- classweight4_optuna: CAGR 11.88%, Sharpe 0.857, MDD -19.91%, turnover 6.24.
