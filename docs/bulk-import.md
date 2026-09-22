# 單次大量匯入規格（Bulk Import，接免費額度耗盡後的資料缺口）

> 狀態：規格先行，實作（`tools/import_bulk.py`）等檔案交付後再寫，
> 避免對沒見過的格式寫死解析器。匯入一律走既有 repository upsert，
> 冪等、可重跑；匯入後跑稽核＋health 確認。

## 檔案清單（UTF-8 CSV，除 shares 外）

| 檔名 | 來源表 | 必填欄位 |
|------|--------|----------|
| `stocks.csv` | stocks | `stock_id`；建議 `stock_name,market,listed_date,delisted_date,industry` |
| `prices.csv` | prices | `stock_id,trade_date,open,high,low,close,volume,traded_value`；建議 `source`（缺省填 `bulk`） |
| `financials.csv` | financials | `stock_id,report_period,announcement_date,available_date`；建議 `revenue,net_income,equity,assets,operating_income,operating_cash_flow`，缺省 `source=bulk` |
| `institutional.csv` | institutional | `stock_id,trade_date`；建議 `foreign_net_buy,trust_net_buy,margin_balance,short_balance,float_shares`，缺省 `source=bulk` |
| `taiex.csv` | prices（`stock_id=TAIEX`） | 同 prices；加權指數日線（市場 beta＋MA60  regime 用） |
| `shares.json` | database/shares.json | `{stock_id: {"shares": float\|null, "market_cap": float\|null, "as_of": "YYYY-MM-DD"}}`（ETF 填 market_cap 即可） |

## 欄位規則（違反會整批拒收，由 to_records  fail loud）

- 日期一律 `YYYY-MM-DD`；`report_period` 格式 `YYYYQN`（如 `2024Q1`）。
- PIT 不可違反：`available_date >= announcement_date`（DDL CHECK）。
- OHLC 必須 `> 0` 且 `low <= open,close <= high`（DDL＋稽核會擋；
  實測 FinMind 原始列偶有開盤價差一檔超出區間的 source noise，
  提供前請先清，否則稽核 blocker 會擋訊號）。
- 價格 `volume,traded_value >= 0`。
- 未知欄位直接報錯（不靜默丟棄），缺鍵欄位直接報錯。

## 匯入順序（FK 要求父表先行）

1. `stocks.csv` → 2. `prices.csv`＋`taiex.csv` → 3. `financials.csv` →
   4. `institutional.csv` → 5. `shares.json` 放到 `database/shares.json`。

## 匯入後驗證（離線可跑）

```text
python -u -c "… audit_data_quality('YYYY-MM-DD', session) …"
```

- `missing_prices` blocker 只應剩真正無資料的邊緣商品；
  主力 universe（4 位數代碼普通股＋ETF）應全數有 bar。
- `invalid_prices` 應為 0（先清 source noise）。
- `feature_coverage` 在跑出第一個月 rebalance 前為 blocker 屬正常。
