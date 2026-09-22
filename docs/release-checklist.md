# Release 驗收清單（P5-08）

> 候選版本：________　資料截止日：________　簽核人／日期：________
> 每條勾選時請填證據（run_id 或檔名）；任一 `[ ]` 不得放行。

## 1. 資料

- [ ] `audit_data_quality` 回 `passed=True`（證據 run：________）
- [ ] Point-in-Time：無 `financial_pit_violation` blocker（財報只用 available_date 前資料）
- [ ] Survivorship Bias：下市股歷史保留（universe 以當期成分為準，非現存成分回填）
- [ ] `data_lag_days` 為 0（價格已同步至截止日）

## 2. 研究

- [ ] 最終 OOS 成本前報告（`run_summary.json` metrics，證據：________）
- [ ] 最終 OOS 成本後報告（成本三件套後，證據：________）
- [ ] 敏感度分析確認（四檔 slippage：0.0005／0.001／0.002／0.003 結果並列）
- [ ] 可重現性：`feature_version`＋`model_version`＋`parameter_version`＋`random_state` 全記錄於 run

## 3. 回測

- [ ] 次日開盤執行（signal 日收盤 → execution 日開盤，無當日收盤偷跑）
- [ ] 成本建模：手續費 0.1425%／賣出稅 0.3%／單邊滑價如實計入 `total_cost`
- [ ] Top 15＋rank buffer（hold_rank_threshold=30）生效，單檔上限 10%

## 4. Dashboard

- [ ] 五頁可開（總覽／投組／模型／風險／研究比較），選單只列 succeeded run
- [ ] SHAP 免責標註在場（特徵重要性≠因子收益歸因）
- [ ] 無資料時為空狀態而非 stack trace

## 5. 安全

- [ ] `validate_runtime_secrets` 通過（啟用的 integration 皆有 secret）
- [ ] CI 雙工作流綠（`ci.yml`：lint＋測試；`security.yml`：audit＋掃描）
- [ ] 無 secret 入庫（secret scan 乾淨、`.env`／`*.db` 未提交）

## 6. 文件

- [ ] Runbook（`docs/runbook.md`）與排程文件（`docs/windows-task-scheduler.md`）為最新
- [ ] 本清單已簽核（簽核人／日期如頁首）
