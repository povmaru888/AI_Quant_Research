# beta_60d 消融研究（2020～2025 OOS）

## 實驗設計

- Baseline：完整 factor v4 XGBoost Pseudo-Huber。
- Fixed params：移除 `beta_60d`，沿用各年度 baseline 最佳超參數。
- Re-Optuna：移除 `beta_60d`，各年度重新執行 50 次 Optuna。
- 三組使用相同 panels、標籤、訓練／purge／驗證／OOS 切分、Top15／Hold30、反波動權重及交易成本。
- 底層 panel 版本仍為 `factor_v4`；研究變體記錄為 `factor_v4_ablation_no_beta60d`。

## 六年度平均

| 組別 | Validation IC | OOS IC | Top15 Excess | Top-Bottom | CAGR | Sharpe | MDD | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.066 | 0.033 | 1.22% | 1.39% | 13.08% | 1.036 | -11.08% | 9.699 |
| 移除 Beta／固定參數 | 0.050 | 0.042 | 0.69% | 0.26% | 8.80% | 0.720 | -12.98% | 10.267 |
| 移除 Beta／重新 Optuna | 0.063 | 0.039 | 0.04% | 0.18% | 6.86% | 0.530 | -12.53% | 10.178 |

## 配對年度結果

| 年度 | Baseline IC | 固定參數 IC | Re-Optuna IC | Baseline Sharpe | 固定參數 Sharpe | Re-Optuna Sharpe |
|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 0.012 | 0.027 | 0.038 | 0.970 | 0.578 | 1.565 |
| 2021 | 0.028 | 0.042 | 0.030 | 0.912 | 0.289 | -0.037 |
| 2022 | 0.016 | 0.029 | 0.026 | -0.171 | -0.378 | -0.013 |
| 2023 | 0.032 | 0.055 | 0.059 | 1.333 | 1.562 | 0.634 |
| 2024 | 0.083 | 0.072 | 0.071 | 1.793 | 0.782 | 1.017 |
| 2025 | 0.024 | 0.025 | 0.008 | 1.381 | 1.490 | 0.016 |

## 判讀

- 固定原參數移除 Beta：平均 OOS IC 差異 `+0.0090`，平均 Sharpe 差異 `-0.316`。
- 重新 Optuna 後移除 Beta：平均 OOS IC 差異 `+0.0062`，平均 Sharpe 差異 `-0.506`。
- 固定參數組 OOS IC 優於 baseline：5/6 年；Sharpe 優於 baseline：2/6 年。
- Re-Optuna 組 OOS IC 優於 baseline：4/6 年；Sharpe 優於 baseline：2/6 年。
- 移除 Beta 後的平均 Top15 重疊率：固定參數 49.81%；Re-Optuna 38.24%。
- 平均 Top15 Excess 差異：固定參數 -0.53%；Re-Optuna -1.18%。

## 72 個月配對 Bootstrap

| 組別 | 指標 | 平均差異 | 95% 區間 | 優於 Baseline 月份比例 |
|---|---|---:|---:|---:|
| 移除 Beta／固定參數 | Continuous Rank IC | 0.009 | [-0.001, 0.019] | 50.00% |
| 移除 Beta／固定參數 | Top15 Excess | -0.53% | [-1.42%, 0.34%] | 56.94% |
| 移除 Beta／固定參數 | Top-Bottom Spread | -1.13% | [-2.49%, 0.17%] | 44.44% |
| 移除 Beta／重新 Optuna | Continuous Rank IC | 0.006 | [-0.006, 0.019] | 54.17% |
| 移除 Beta／重新 Optuna | Top15 Excess | -1.18% | [-2.26%, -0.13%] | 43.06% |
| 移除 Beta／重新 Optuna | Top-Bottom Spread | -1.20% | [-2.59%, 0.14%] | 41.67% |

## 結論

`beta_60d` 並非穩定、不可替代的重要因子。移除後平均 OOS Rank IC 反而小幅上升，固定參數組有 5/6 年 IC 較高；但 bootstrap 區間涵蓋 0，也不足以認定移除 Beta 能穩定改善排序。

經濟結果的六年平均偏向保留 Beta，但年度結果有好有壞。固定參數消融後 CAGR、Sharpe、Sortino 各有 2/6 年改善，MDD 也有 2/6 年改善；重新 Optuna 組的 CAGR、Sharpe、Sortino 同樣各有 2/6 年改善。

Top15 Excess 與 Top-Bottom Spread 在兩組消融實驗都只有 1/6 年改善，顯示 Beta 可能對頭部選股有條件式價值；不過固定參數組的月度差異信賴區間仍涵蓋 0，證據強度有限。

較保守的結論是：`beta_60d` 的增量重要性低於 SHAP 排名所呈現，效果具有市場狀態依賴。它可以暫時保留在目前主模型中，但不能單憑 SHAP 宣稱是核心 Alpha；更適合進一步研究它與波動率、動能及市場 regime 的交互作用。

績效指標存在年度市場差異；正式判斷以配對的 OOS IC、Top15 Excess 與跨年度方向一致性為主，不以單一年份 CAGR 決定。
