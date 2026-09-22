# Windows Task Scheduler 設定（日常更新＋月度換倉）

> 佔位符：`<REPO>`＝專案根目錄、`<VENV>`＝虛擬環境目錄。
> Token 只經環境變數注入，任何 `schtasks` 指令與 log 檔都不得出現 secret。

## 1. 前置檢查

1. `<VENV>\Scripts\python.exe --version` 可執行。
2. `<REPO>\.env` 存在，僅含變數名（參照 `.env.example`）：
   `FINMIND_TOKEN`、`TELEGRAM_BOT_TOKEN`、`GEMINI_API_KEY`。
3. `<REPO>\config.yaml` 就緒；`<REPO>\logs\` 目錄存在。
4. Dry run（不連網，只驗參數解析）：
   `<VENV>\Scripts\python.exe -m src.jobs.daily_update --help`
   `<VENV>\Scripts\python.exe -m src.jobs.monthly_rebalance --help`

## 2. Task 1：每日 09:00 資料更新

```bat
schtasks /Create /TN "QuantDailyUpdate" /SC DAILY /ST 09:00 ^
  /TR "\"<VENV>\Scripts\python.exe\" -m src.jobs.daily_update --config \"<REPO>\config.yaml\" >> \"<REPO>\logs\daily_%DATE%.log\" 2>&1" ^
  /ST 09:00 /F
```

- 工作目錄（Start in）：`<REPO>`。
- 成功 exit code `0`；`1` 為執行失敗；`2` 為參數或設定錯誤。

## 3. Task 2：月末收盤後換倉（每月最後日曆日 18:00 觸發）

Scheduler 無「月末交易日」觸發器，採每月最後日曆日 18:00 觸發，
`--signal-date` 由操作者填入當月最後交易日（job 內 `verify_month_end` 把關）：

```bat
schtasks /Create /TN "QuantMonthlyRebalance" /SC MONTHLY /MO LASTDAY /ST 18:00 ^
  /TR "\"<VENV>\Scripts\python.exe\" -m src.jobs.monthly_rebalance --signal-date YYYY-MM-DD --config \"<REPO>\config.yaml\" >> \"<REPO>\logs\rebalance_%DATE%.log\" 2>&1" ^
  /F
```

查詢當月最後交易日（有價日曆在 DB，操作前確認）：

```bat
<VENV>\Scripts\python.exe -c "from database import session_scope; print('check prices table')"
```

## 4. 失敗檢查與手動重跑

| exit code | 含義 | 處置 |
| --- | --- | --- |
| 0 | 成功 | 核對 log 內 `run_id=` 行 |
| 1 | 執行失敗 | 看 log 末行；查 `pipeline_runs` 該 run 的 `error_message`；修因後重跑 |
| 2 | 參數或設定錯誤 | 檢查日期格式與 `--config` 路徑 |

手動重跑（冪等，重跑不產生重複資料）：

```bat
<VENV>\Scripts\python.exe -m src.jobs.daily_update --as-of YYYY-MM-DD --config "<REPO>\config.yaml"
<VENV>\Scripts\python.exe -m src.jobs.monthly_rebalance --signal-date YYYY-MM-DD --config "<REPO>\config.yaml"
```

## 5. 測試環境驗證紀錄

- 依本文件在測試環境執行 dry run（`--help` 兩 job 皆回 0）。
- 自動驗證：`tests/test_scheduler_docs.py` 掃描本文件（任務名、`09:00`、log 目錄存在性、無 token 賦值）。
