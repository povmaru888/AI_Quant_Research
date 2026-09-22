# 台股多因子 XGBoost 輪動與波動率風控量化交易系統 SDD

| 欄位 | 內容 |
| --- | --- |
| 文件版本 | v1.0 |
| 文件類型 | Software Design Description |
| 對應需求 | 台股多因子量化交易系統 PRD v1.0 |
| 系統類型 | Long-only 台股量化研究與投組管理系統 |
| 核心技術 | Python、SQLite、FinMind、XGBoost、Optuna、VectorBT、Streamlit |
| 開發範圍 | 14 天 MVP |

## 1. 目的與範圍

本文件將 PRD 定義為可實作、可測試的軟體設計。系統在每月月底使用當時可得的台股資料建立多因子特徵，以 XGBClassifier 預測未來 20 個交易日進入高報酬五分位的機率，並將排名轉為受成本與風險約束的 Long-only Top 15 投組。

系統產出研究與模擬結果，不包含券商下單、個股放空、融券管理或投資建議。所有交易訊號與權重由可重現的模型和規則決定；LLM 僅能產生摘要與解釋。

### 1.1 MVP 交付

- FinMind 與備援價格資料的擷取、驗證與 SQLite 儲存。
- 歷史可交易宇宙與 Point-in-Time 財務、籌碼資料快照。
- 30 項因子、標籤、基準模型、XGBClassifier 與 Optuna。
- Rolling Walk-forward、Purge 與串接後 OOS 績效。
- Top 15、Top 30 排名緩衝、反波動權重、組合波動率目標與 MA60 濾網。
- 含成本的 VectorBT 回測、Streamlit Dashboard、CSV 與 PDF 報表。

### 1.2 延後或排除功能

| 類別 | 項目 |
| --- | --- |
| 延後 | Telegram、Gemini 摘要、Backtrader 對照、TXF Beta Hedge。 |
| 排除 | 當沖、直接放空、券商 API、真實下單、LSTM 收盤價預測。 |

## 2. 設計原則

1. **Point-in-Time 優先**：非價格資料只能在 available date 小於或等於 rebalance date 時使用。
2. **時間切割優先**：同月所有股票必須位於同一個 Train、Validation 或 Test 集合。
3. **訊號與成交分離**：月底收盤生成訊號，次一交易日開盤成交。
4. **投組層級風控**：反波動只給相對權重；總曝險由組合協方差與波動率目標決定。
5. **逐筆成本建模**：手續費、交易稅和滑價依訂單記錄計算。
6. **可重現性**：資料截止日、特徵、模型、參數、套件版本、隨機種子與 run log 都必須保存。
7. **Test 隔離**：Test 不得參與 Optuna、Top N、濾網與任何模型選擇。

## 3. 整體架構

~~~mermaid
flowchart LR
    F[FinMind] --> E[ETL 與資料驗證]
    Y[yfinance 備援] --> E
    E --> D[(SQLite)]
    D --> U[歷史宇宙篩選]
    U --> P[Point in Time 快照]
    P --> FE[特徵引擎]
    FE --> WF[Walk forward 訓練]
    WF --> M[XGBoost 預測]
    M --> S[排名與訊號]
    S --> PC[投組建構]
    PC --> R[風控與曝險]
    R --> B[VectorBT 回測]
    B --> O[OOS 指標與報表]
    O --> UI[Streamlit]
~~~

### 3.1 分層責任

| 層級 | 模組 | 責任 |
| --- | --- | --- |
| 資料擷取 | etl、Adapter | 擷取、標準化、資料品質檢查與 upsert。 |
| 資料庫 | database、SQLite | 保存原始資料、特徵、預測、訊號、訂單與 run log。 |
| 研究 | universe、features、labels | 建立歷史可交易宇宙、Point-in-Time 快照、特徵與標籤。 |
| 模型 | model、optimization、walk forward | 訓練、調參、驗證與產出 OOS 預測。 |
| 投組 | portfolio、risk | 排名緩衝、權重、曝險、波動率和市場濾網。 |
| 回測 | backtest、metrics | 次日開盤成交、成本模擬、績效與敏感度分析。 |
| 呈現 | app、report | Dashboard、CSV、PDF 和可選摘要。 |

## 4. 專案結構

~~~text
taiwan-quant-xgb/
├── README.md
├── requirements.txt
├── config.yaml
├── app.py
├── database/
│   └── schema.sql
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_walk_forward.ipynb
├── src/
│   ├── etl.py
│   ├── database.py
│   ├── universe.py
│   ├── features.py
│   ├── labels.py
│   ├── model.py
│   ├── optimization.py
│   ├── walk_forward.py
│   ├── portfolio.py
│   ├── risk.py
│   ├── backtest.py
│   ├── metrics.py
│   └── report.py
├── tests/
│   ├── test_data_quality.py
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_portfolio.py
│   └── test_execution.py
└── reports/
    ├── performance.csv
    ├── factor_ic.csv
    ├── run_summary.json
    └── report.pdf
~~~

## 5. 執行環境與設定

### 5.1 套件

requirements.txt 至少固定以下套件的相容版本：pandas、numpy、scikit-learn、xgboost、optuna、vectorbt、SQLAlchemy、requests、PyYAML、plotly、streamlit、shap 與 pytest。

### 5.2 設定檔

config.yaml 是策略與研究假設的唯一入口；模組內不得硬編碼策略參數。

~~~yaml
project:
  name: taiwan-quant-xgb
  timezone: Asia/Taipei
  random_state: 42

data:
  database_url: sqlite:///database/quant.db
  finmind_token_env: FINMIND_TOKEN
  price_start_date: "2015-01-01"
  fallback_source: yfinance

universe:
  min_market_cap_twd: 5000000000
  min_avg_traded_value_20d_twd: 20000000
  min_price_twd: 10
  excluded_flags: [KY, DISPOSAL, FULL_CASH_SETTLEMENT]

features:
  winsor_lower_quantile: 0.01
  winsor_upper_quantile: 0.99
  correlation_threshold: 0.85
  volatility_window: 60
  feature_version: factor_v1

label:
  horizon_trading_days: 20
  top_quantile: 0.20

validation:
  train_years: 4
  validation_years: 1
  test_years: 1
  purge_trading_days: 20
  optuna_trials: 50

portfolio:
  top_n: 15
  hold_rank_threshold: 30
  max_individual_weight: 0.10
  target_annual_volatility: 0.15
  max_equity_exposure: 1.00
  taiex_ma_window: 60
  reduced_exposure_below_ma: 0.50

execution:
  signal_time: month_end_close
  execution_time: next_trading_day_open
  broker_fee_rate: 0.001425
  sell_tax_rate: 0.003
  slippage_rate_per_side: 0.001
~~~

### 5.3 設計預設

下列項目為開發必須固定、但 PRD 未明訂的技術預設：

- 日期格式為 YYYY-MM-DD，交易日以台灣市場實際有價格資料的日期序列認定。
- stock id 使用文字型別，保留前導零與可能的代號字元。
- ETL 採 idempotent upsert；重跑不得產生重複資料。
- 參數變動必須同步更新 config、測試與 run log。

## 6. 資料模型

### 6.1 關聯

~~~mermaid
erDiagram
    STOCKS ||--o{ PRICES : has
    STOCKS ||--o{ FINANCIALS : reports
    STOCKS ||--o{ INSTITUTIONAL : has
    STOCKS ||--o{ FEATURES : has
    STOCKS ||--o{ PREDICTIONS : has
    STOCKS ||--o{ SIGNALS : has
    STOCKS ||--o{ POSITIONS : has
    STOCKS ||--o{ ORDERS : trades
    PIPELINE_RUNS ||--o{ PREDICTIONS : produces
    PIPELINE_RUNS ||--o{ ORDERS : simulates
~~~

### 6.2 SQLite Schema

以下 schema 是 MVP 的資料庫基線。金額一律為新台幣，數值資料使用 REAL。

~~~sql
PRAGMA foreign_keys = ON;

CREATE TABLE stocks (
    stock_id TEXT PRIMARY KEY,
    stock_name TEXT,
    market TEXT NOT NULL,
    listed_date TEXT,
    delisted_date TEXT,
    industry TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE prices (
    trade_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    open REAL NOT NULL CHECK (open > 0),
    high REAL NOT NULL CHECK (high > 0),
    low REAL NOT NULL CHECK (low > 0),
    close REAL NOT NULL CHECK (close > 0),
    volume REAL NOT NULL CHECK (volume >= 0),
    traded_value REAL NOT NULL CHECK (traded_value >= 0),
    source TEXT NOT NULL,
    PRIMARY KEY (trade_date, stock_id),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE financials (
    stock_id TEXT NOT NULL,
    report_period TEXT NOT NULL,
    announcement_date TEXT NOT NULL,
    available_date TEXT NOT NULL,
    revenue REAL,
    net_income REAL,
    equity REAL,
    assets REAL,
    operating_income REAL,
    operating_cash_flow REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (stock_id, report_period, available_date),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
    CHECK (available_date >= announcement_date)
);

CREATE TABLE institutional (
    trade_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    foreign_net_buy REAL,
    trust_net_buy REAL,
    margin_balance REAL,
    short_balance REAL,
    float_shares REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (trade_date, stock_id),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    run_time TEXT NOT NULL,
    data_end_date TEXT NOT NULL,
    model_version TEXT,
    feature_version TEXT NOT NULL,
    parameter_version TEXT NOT NULL,
    universe_count INTEGER,
    position_count INTEGER,
    status TEXT NOT NULL CHECK (status IN ('started', 'succeeded', 'failed')),
    error_message TEXT
);

CREATE TABLE features (
    rebalance_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    momentum_20d REAL,
    momentum_60d REAL,
    momentum_120d REAL,
    momentum_20d_ex_5d REAL,
    ma20_ma60_gap REAL,
    rsi14 REAL,
    price_ma20_gap REAL,
    volume_ma20_ma60 REAL,
    earnings_yield REAL,
    book_to_market REAL,
    sales_yield REAL,
    dividend_yield REAL,
    roe REAL,
    roa REAL,
    revenue_yoy REAL,
    operating_income_qoq REAL,
    accrual_assets REAL,
    volatility_60d REAL,
    beta_60d REAL,
    max_drawdown_120d REAL,
    turnover_60d REAL,
    foreign_net_buy_float REAL,
    trust_net_buy_float REAL,
    margin_balance_change REAL,
    close_60d_high REAL,
    log_market_cap REAL,
    amihud_illiquidity REAL,
    short_margin_ratio REAL,
    operating_margin REAL,
    revenue_mom REAL,
    missing_flag INTEGER NOT NULL DEFAULT 0 CHECK (missing_flag IN (0, 1)),
    PRIMARY KEY (rebalance_date, stock_id, feature_version),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE predictions (
    prediction_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction_probability REAL NOT NULL CHECK (
        prediction_probability >= 0 AND prediction_probability <= 1
    ),
    rank INTEGER NOT NULL CHECK (rank > 0),
    PRIMARY KEY (prediction_date, stock_id, model_version),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);

CREATE TABLE signals (
    signal_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    signal TEXT NOT NULL CHECK (signal IN ('BUY', 'HOLD', 'SELL', 'NONE')),
    rank INTEGER NOT NULL,
    target_weight REAL NOT NULL CHECK (target_weight >= 0 AND target_weight <= 1),
    PRIMARY KEY (signal_date, stock_id),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);

CREATE TABLE positions (
    position_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    shares REAL NOT NULL CHECK (shares >= 0),
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),
    market_value REAL NOT NULL CHECK (market_value >= 0),
    PRIMARY KEY (position_date, stock_id),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    execution_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL CHECK (quantity > 0),
    open_price REAL NOT NULL CHECK (open_price > 0),
    executed_price REAL NOT NULL CHECK (executed_price > 0),
    notional REAL NOT NULL CHECK (notional > 0),
    broker_fee REAL NOT NULL CHECK (broker_fee >= 0),
    transaction_tax REAL NOT NULL CHECK (transaction_tax >= 0),
    slippage_cost REAL NOT NULL CHECK (slippage_cost >= 0),
    total_cost REAL NOT NULL CHECK (total_cost >= 0),
    FOREIGN KEY (stock_id) REFERENCES stocks(stock_id),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id),
    CHECK (execution_date > signal_date)
);

CREATE TABLE portfolio_daily (
    trade_date TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    nav REAL NOT NULL CHECK (nav >= 0),
    equity_exposure REAL NOT NULL CHECK (equity_exposure >= 0 AND equity_exposure <= 1),
    forecast_volatility REAL,
    realized_volatility REAL,
    drawdown REAL,
    taiex_close REAL,
    taiex_ma60 REAL,
    market_regime TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);

CREATE INDEX idx_prices_stock_date ON prices (stock_id, trade_date);
CREATE INDEX idx_financials_available ON financials (stock_id, available_date);
CREATE INDEX idx_features_date_version ON features (rebalance_date, feature_version);
CREATE INDEX idx_predictions_date_rank ON predictions (prediction_date, rank);
CREATE INDEX idx_orders_execution_date ON orders (execution_date);
~~~

### 6.3 Point-in-Time Join

每個股票與 rebalance date 只選擇當時最新可用的財務資料。下列查詢可作為批次邏輯的參考；實作應以 window function 或 pandas merge asof 避免 N+1 查詢。

~~~sql
WITH eligible_financials AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (
            PARTITION BY f.stock_id
            ORDER BY f.available_date DESC, f.announcement_date DESC
        ) AS rn
    FROM financials AS f
    WHERE f.available_date <= :rebalance_date
)
SELECT *
FROM eligible_financials
WHERE rn = 1;
~~~

## 7. 資料擷取與品質控制

### 7.1 Adapter 合約

~~~python
from typing import Protocol
import pandas as pd

class MarketDataAdapter(Protocol):
    def fetch_prices(
        self, start_date: str, end_date: str | None
    ) -> pd.DataFrame: ...

    def fetch_financials(
        self, start_date: str, end_date: str | None
    ) -> pd.DataFrame: ...

    def fetch_institutional(
        self, start_date: str, end_date: str | None
    ) -> pd.DataFrame: ...
~~~

FinMindAdapter 實作主要資料來源。YFinanceAdapter 只能做價格備援或交叉檢核。所有 Adapter 必須回傳標準化欄位，禁止把供應商原始欄位名稱傳給下游模組。

### 7.2 ETL 流程

1. 讀取最後成功的 data end date 和本次更新目標日。
2. 下載增量價格、財報和籌碼資料。
3. 驗證日期、代號、價格、成交量、成交金額和主鍵。
4. 統一欄位、日期和金額單位。
5. 以日期和代號進行 upsert。
6. 寫入 pipeline runs，包括資料截止日、筆數、狀態和錯誤摘要。

### 7.3 資料品質規則

| 類別 | 規則 | 失敗處理 |
| --- | --- | --- |
| 價格 | OHLC 必須大於零，且 high 不小於 low。 | 拒絕資料並標記 ETL 失敗。 |
| 日期 | 同一股票與交易日僅能一筆。 | 資料庫複合主鍵阻擋重複。 |
| 財報 | available date 不得早於 announcement date。 | 拒絕該筆資料。 |
| 股票狀態 | delisted date 後不可進入宇宙。 | universe 模組排除。 |
| 缺值 | 財務資料不能以未公告資料補值。 | 依因子規則處理並記錄 missing flag。 |

## 8. 歷史可交易宇宙

每個 rebalance date 必須獨立建構宇宙，禁止以今天仍存在的股票清單過濾全部歷史。

~~~text
當月已上市且未下市
  -> 具有效當日價格
  -> 收盤價大於 10 元
  -> 市值大於 50 億元
  -> 過去 20 交易日平均成交金額大於 2,000 萬元
  -> 非 KY 處置或全額交割
  -> 當月可交易宇宙
~~~

~~~python
def build_universe(
    rebalance_date: str,
    prices: pd.DataFrame,
    stocks: pd.DataFrame,
    security_flags: pd.DataFrame | None,
    config: dict,
) -> pd.DataFrame:
    """回傳 stock id、市值、20 日均成交額、收盤價和排除旗標。"""
~~~

## 9. 特徵工程

### 9.1 處理順序

~~~text
原始價格 財報 籌碼
  -> Point-in-Time 篩選
  -> 因子計算
  -> 1% 和 99% Winsorize
  -> 同月橫截面排名
  -> 標準化
  -> 移除絕對相關係數大於 0.85 的重複因子
  -> 寫入 features
~~~

每個月份必須獨立做縮尾、排名和標準化。不可使用未來月份的分布資訊。

### 9.2 因子欄位

| 群組 | 欄位 | 計算摘要 |
| --- | --- | --- |
| 動能與技術 | momentum 20d、60d、120d | close t 除以過去 close 減 1。 |
| 動能與技術 | momentum 20d ex 5d | close t-5 除以 close t-20 減 1。 |
| 動能與技術 | ma20 ma60 gap | MA20 除以 MA60 減 1。 |
| 動能與技術 | rsi14 | 14 日 RSI。 |
| 動能與技術 | price ma20 gap | close 除以 MA20 減 1。 |
| 動能與技術 | volume ma20 ma60 | 20 日均量除以 60 日均量。 |
| 價值 | earnings yield | 本益比倒數。 |
| 價值 | book to market | 股價淨值比倒數。 |
| 價值 | sales yield | 股價營收比倒數。 |
| 價值 | dividend yield | 近 12 個月現金股利除以股價。 |
| 品質與成長 | roe、roa | 近四季淨利除以平均權益或平均資產。 |
| 品質與成長 | revenue yoy、revenue mom | 當月營收年增率與月增率。 |
| 品質與成長 | operating income qoq | 單季營業利益季增率。 |
| 品質與成長 | accrual assets | 淨利減營業現金流，再除以總資產。 |
| 風險 | volatility 60d | 60 日日對數報酬標準差乘以 sqrt 252。 |
| 風險 | beta 60d | 個股與 TAIEX 報酬 60 日 Beta。 |
| 風險 | max drawdown 120d | 120 日最大回撤。 |
| 風險 | turnover 60d | 60 日平均周轉率。 |
| 法人與籌碼 | foreign net buy float | 20 日外資買賣超除以流通股。 |
| 法人與籌碼 | trust net buy float | 20 日投信買賣超除以流通股。 |
| 法人與籌碼 | margin balance change | 20 日融資餘額變化率。 |
| 法人與籌碼 | close 60d high | 收盤價除以 60 日最高價。 |
| 規模與流動性 | log market cap | 市值自然對數。 |
| 規模與流動性 | amihud illiquidity | 日絕對報酬除以成交金額的平均。 |
| 補充 | short margin ratio | 融券餘額除以融資餘額。 |
| 補充 | operating margin | 營業利益除以營業收入。 |

### 9.3 缺值處理

- 財務因子僅能用當時已公告的資料 forward fill。
- 負值估值因子不可直接當成低估，採同月橫截面中性或中位數處理，並設 missing flag。
- 分母為零不得產生無限值。
- features 模組要輸出每月、每因子的覆蓋率與缺值比例。

## 10. 標籤 模型與最佳化

### 10.1 標籤

~~~text
future return 20d = close[t + 20] / close[t] - 1
label = 1，當日橫截面報酬位於最高 20%
label = 0，其他股票
~~~

建立標籤時可讀取未來價格，但任何未來價格不得進入特徵欄位。

### 10.2 模型介面

~~~python
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelArtifact:
    model_version: str
    feature_columns: list[str]
    best_params: dict
    validation_rank_ic: float

def train_xgb_classifier(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    valid_features: pd.DataFrame,
    valid_labels: pd.Series,
    config: dict,
) -> ModelArtifact: ...

def predict_probabilities(
    artifact: ModelArtifact,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """回傳 stock id、prediction probability 和 rank。"""
~~~

### 10.3 比較模型

| 類別 | 模型 | 目的 |
| --- | --- | --- |
| 基準 | Momentum 20D | 驗證單一動能訊號。 |
| 基準 | Momentum 20D 加 60D | 比較短中期動能。 |
| 基準 | Equal-weight Factor Rank | 比較非 ML 多因子方法。 |
| 簡單模型 | Logistic Regression | 檢視非線性複雜度是否有增益。 |
| 主模型 | XGBClassifier | 預測最高五分位機率。 |

因子消融至少比較：動能、動能加籌碼、動能加籌碼加基本面、全部因子。

### 10.4 Optuna

~~~text
主要目標：Validation Mean Rank IC
次要觀察：Validation Top 15 Portfolio Return
Trials：50
~~~

搜尋參數包含 max depth、learning rate、n estimators、subsample、colsample bytree、min child weight、reg alpha 和 reg lambda。每次 trial 的參數與結果都要保存。Test 資料完全隔離。

## 11. Walk Forward 驗證

### 11.1 切割

~~~text
Train 4 年 -> Validation 1 年 -> Purge 20 個交易日 -> Test 1 年
~~~

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| 1 | 2015 至 2018 | 2019 | 2020 |
| 2 | 2016 至 2019 | 2020 | 2021 |
| 3 | 2017 至 2020 | 2021 | 2022 |

實際 fold 數依資料最後日期自動產生。Purge 天數必須等於 label horizon 的 20 個交易日。最終績效只能使用所有 Test 期間串接後的 OOS 結果。

### 11.2 必要輸出

- 每個 Fold 的 CAGR、Sharpe、Sortino、Calmar、Max Drawdown、勝率、Turnover 和 Rank IC。
- 成本前後績效及手續費、交易稅、滑價成本。
- Top 10、15、20、30 的敏感度分析。
- 因子消融、模型比較、SHAP Top 10、Monthly IC、rolling IC 和 ICIR。

SHAP 僅表示模型特徵重要性，不能表述為因子收益歸因。

## 12. 投組建構與風控

### 12.1 排名緩衝

~~~text
Entry：新股票 rank 小於或等於 15
Hold：既有持股 rank 小於或等於 30
Exit：既有持股 rank 大於 30
~~~

portfolio 模組必須接受前一期 positions，決定 BUY、HOLD、SELL 與 NONE。單一股票在同一訊號日不得出現衝突訊號。

### 12.2 權重與曝險

~~~text
sigma i = std(log return i, 60 trading days) * sqrt(252)
w raw i = 1 / sigma i
w relative i = w raw i / sum(w raw)

sigma p = sqrt(w transpose Sigma w)
equity exposure = min(1, target annual volatility / sigma p)
target weight i = equity exposure * w relative i
cash weight = 1 - equity exposure
~~~

- 個別股票最高權重固定為 10%。超過上限時，截斷後將餘額分配給未觸及上限的股票。
- 協方差矩陣資料不足時，不得以零相關假設代替；該月投組建立必須失敗並留下錯誤紀錄。
- TAIEX 收盤高於 MA60 時可配置 100% 股票曝險，低於或等於 MA60 時最高 50%。

## 13. 執行與回測

### 13.1 時點

~~~text
月末收盤 -> 特徵 -> 預測 -> 目標權重 -> 次一交易日開盤成交
~~~

signal date 必須早於 execution date。此規則要由資料庫 CHECK 與單元測試共同保護。

### 13.2 成本

| 項目 | 買進 | 賣出 |
| --- | ---: | ---: |
| 手續費 | 0.1425% | 0.1425% |
| 交易稅 | 0% | 0.3% |
| 單邊滑價 | 0.1% | 0.1% |

~~~python
def executed_price(open_price: float, side: str, slippage: float) -> float:
    if side == "BUY":
        return open_price * (1 + slippage)
    if side == "SELL":
        return open_price * (1 - slippage)
    raise ValueError(side)

def transaction_cost(notional: float, side: str, cfg: dict) -> dict:
    broker_fee = notional * cfg["broker_fee_rate"]
    transaction_tax = notional * cfg["sell_tax_rate"] if side == "SELL" else 0.0
    slippage_cost = notional * cfg["slippage_rate_per_side"]
    return {
        "broker_fee": broker_fee,
        "transaction_tax": transaction_tax,
        "slippage_cost": slippage_cost,
        "total_cost": broker_fee + transaction_tax + slippage_cost,
    }
~~~

### 13.3 VectorBT

backtest 模組先以 signals 與 target weight 建立可稽核的訂單，再交給 VectorBT 模擬。VectorBT 的 fees 和 slippage 設定必須與 orders 表的成本一致。每個結果都要能追溯：

~~~text
run id -> signal date -> execution date -> order id -> executed price -> total cost
~~~

成本敏感度固定測試單邊滑價 0.05%、0.10%、0.20% 和 0.30%，並輸出 CAGR、Sharpe、Max Drawdown、Turnover。

## 14. Dashboard 報表與排程

### 14.1 Streamlit

| 頁面 | 內容 |
| --- | --- |
| 總覽 | CAGR、Sharpe、Sortino、MDD、Turnover、IC、Equity Curve、Drawdown、月報酬熱圖。 |
| 投組 | 持股、排名、預測機率、權重、個股波動率和 Beta。 |
| 模型 | SHAP Top 10、Feature Importance、Monthly IC、預測分布。 |
| 風險 | 曝險、預測與實際波動、MDD、Turnover、market regime。 |
| 研究比較 | 基準、因子消融、Top N、成本敏感度。 |

### 14.2 報表

report 模組產出 performance.csv、factor ic.csv、run summary.json 和一頁式 report.pdf。run summary 必須包含 run id、資料截止日、特徵與模型版本、完整參數和 OOS 指標。

### 14.3 排程

| 模式 | 頻率 | 功能 |
| --- | --- | --- |
| 研究回測 | 手動或 CI | 重建歷史資料、Walk-forward 與 OOS。 |
| 日常更新 | 每日 09:00 | 更新資料、驗證、更新 Dashboard 與風險資訊。 |
| 月度換倉 | 月末收盤後 | 產生訊號與目標投組，模擬次日開盤成交。 |

每日更新不代表每日換倉。

## 15. 測試與驗收

### 15.1 單元測試

| 模組 | 必測案例 |
| --- | --- |
| database | upsert 不重複、外鍵有效、日期與數值約束。 |
| universe | 下市、低市值、低成交額、低價格與排除旗標。 |
| features | 不使用未來資料、分母為零、缺值旗標、每月標準化。 |
| labels | 固定 20 個交易日預測期，未來報酬不進特徵。 |
| walk forward | 同一日期不可跨集合，Purge 等於標籤期。 |
| portfolio | Entry、Hold、Exit、個股上限、權重和現金。 |
| risk | 協方差、波動率曝險和 MA60 濾網。 |
| backtest | 訊號早於成交、買賣滑價方向、費用與賣出稅。 |

### 15.2 驗收條件

1. Point-in-Time 和 Look-ahead Bias 測試通過。
2. 歷史宇宙不以現存股票回測過去。
3. 模型、訊號、成交、成本與報表可追溯到同一 run id。
4. OOS 結果包含成本前後、各 Fold 與敏感度分析。
5. Dashboard 能呈現要求的績效、持股、模型和風險資訊。

## 16. 安全性與操作限制

- FinMind、Telegram、Gemini 等 Token 只可經環境變數注入，不得寫入設定檔、SQLite、Notebook 或 Git。
- 不保存券商帳密、個人投資帳戶或個人識別資料。
- 資料來源失敗、Point-in-Time 檢查失敗、因子覆蓋率不足或協方差無法估計時，必須停止該期訊號產生。
- LLM 模組失敗不得阻塞 ETL、模型、訊號或回測。

## 17. 待確認項目

| 項目 | 需確認決策 |
| --- | --- |
| 調整後價格 | 特徵、標籤與成交是否採同一套除權息價格定義。 |
| 市值與流通股 | FinMind 欄位可用性與精確計算來源。 |
| 處置與全額交割 | 歷史旗標來源與生效日期。 |
| 交易日曆 | 休市、颱風休市與臨時休市資料來源。 |
| 部位圓整 | 回測採整股、零股或連續權重。 |
| 公司行動 | 股利、減資、分割、代號變更的處理。 |
| LLM 摘要 | 公告資料來源、授權與引用方式。 |

## 18. PRD 對應矩陣

| PRD 領域 | SDD 章節 | 主要產物 |
| --- | --- | --- |
| 資料與 Point-in-Time | 6、7、8 | database、etl、universe、SQLite schema。 |
| 因子工程 | 9 | features、features 表、覆蓋率報表。 |
| 標籤與模型 | 10 | labels、model、optimization。 |
| Walk-forward 與 OOS | 11 | walk forward、fold 報表、OOS equity curve。 |
| 投組與風控 | 12 | portfolio、risk、signals、positions。 |
| 成交與成本 | 13 | backtest、orders、成本敏感度。 |
| 看板與報表 | 14 | app、report、CSV、PDF。 |
| 測試與可重現性 | 15、16 | tests、pipeline runs、設定與版本紀錄。 |
