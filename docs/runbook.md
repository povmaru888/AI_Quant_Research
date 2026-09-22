# 營運 Runbook（P5-07）

> 對象：每日 09:00 例行檢查與 job 失敗時的值班者。
> 指令一律不含 secret；token 只活在環境變數（見 `.env.example`）。
> 健康狀態以 `get_system_health` 為單一入口，稽核以 `audit_data_quality` 為準。

## 0. 每日 09:00 例行

1. 看健康狀態：`latest_success` 是否為昨日／今日、`data_lag_days` 是否為 0。
2. `latest_failure` 若有新失敗，依第 1–5 節對號入座。
3. `feature_coverage` < 0.9 → 按第 1 節（資料失敗）處理。

## 1. 資料失敗（缺價格、FinMind 超時、lag 飆高）

- 觀察：`data_lag_days` > 0、`audit_data_quality` 報 `missing_prices`、
  結構化日誌 `event=fetch` 連續 `transport failure`。
- 判斷：FinMind 限流／維護（看 HTTP 429／5xx），或 yfinance 備援亦空。
- 處置：
  - 重跑日常更新（冪等 upsert，可安全重跑）：
    `python -m src.jobs.daily_update --as-of YYYY-MM-DD`
  - 仍失敗 → **停止月度訊號**（勿在半套資料上跑），等次日再查。
  - 連敗 3 日 → 回報並檢查 `price_start_date` 與網路 egress。

## 2. PIT 違規（financial_pit_violation blocker）

- 觀察：`audit_data_quality` 報 `financial_pit_violation`，`assert_no_blockers`
  直接擋下訊號 job（ValueError: data quality blockers）。
- 判斷：上游財報 `available_date` 早於 `announcement_date`（時序不可能），
  屬來源髒資料，非模型問題。
- 處置：
  - **停止訊號 job**（已被程式阻擋，無需手動停）。
  - 修正來源後重跑 `daily_update`，再跑 `audit_data_quality` 確認 `passed`，
    方可跑 `python -m src.jobs.monthly_rebalance --signal-date YYYY-MM-DD`。

## 3. 模型失敗（run failed：空 universe、單類別、無標籤）

- 觀察：`latest_failure.error` 為 `empty universe`／`single-class`／`no labels`。
- 判斷：多為資料端問題（全下市、標籤窗口不夠），先查 `universe_count` 與
  `data_end_date` 是否合理。
- 處置：
  - 資料問題 → 回第 1 節補資料後重跑研究。
  - 非資料問題（參數改壞）→ **停止重跑**，回報並附 `parameter_version`。

## 4. 協方差不足（covariance missing）

- 觀察：`latest_failure.error` 含 `covariance`；P2-12 風控已把該月降為
  保守曝險（訊號仍產出但權重縮水）。
- 判斷：`returns` 窗口不足 60 日（新上市股過多或缺 bar）。
- 處置：
  - 補齊價格窗口後重跑 `monthly_rebalance`（`replace_orders` 原子取代，
    不會殘留半批訂單）。
  - 若為常態（小宇宙），接受降曝險結果並記錄於 release checklist。

## 5. 報表失敗（export_reports ValueError、PDF 缺字）

- 觀察：`export_reports` 報 `not completed`（run 非 succeeded）或
  `missing columns`（store 欄位漂移）。
- 判斷：前者等 run 成功再匯；後者為 Phase 5 store 實作與 P4-02 欄位契約不一致。
- 處置：
  - 前者 → 不處理，等 run 完成。
  - 後者 → **停止發布**，修 store 對齊 `HOLDING_COLUMNS` 後重匯。
  - PDF 中文缺字為已知限制（P4-08），不算失敗，英文鍵正常即可放行。

## 附錄：手動重跑指令（皆無 secret）

```text
python -m src.jobs.daily_update --as-of YYYY-MM-DD
python -m src.jobs.monthly_rebalance --signal-date YYYY-MM-DD --execution-date YYYY-MM-DD
python -m pytest tests/test_data_quality.py tests/test_health.py -v
```
