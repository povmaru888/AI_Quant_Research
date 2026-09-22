# 台股多因子量化交易系統 工作分解結構

| 欄位 | 內容 |
| --- | --- |
| 文件版本 | v1.0 |
| 對應文件 | 台股多因子量化交易系統 SDD v1.0 |
| 專案方式 | 垂直切片加技術層級 |
| 任務狀態 | Phase 0（P0-01 至 P0-08）已完成，Phase 1（P1-01 至 P1-11）已完成，Phase 2（P2-01 至 P2-16）已完成，Phase 3（P3-01 至 P3-09）已完成，Phase 4（P4-01 至 P4-08）已完成，Phase 5（P5-01 至 P5-08）已完成，全 phases 完成 |
| 完成定義 | 目標模組完成、介面可呼叫、驗收測試通過、相依任務狀態正確。 |

## 1. 使用規則

本文件中的每個任務均是獨立可驗證的最小單位。任務完成前不得將下游任務標示為完成。

### 1.1 三個一原則

每個 Task 都有：

1. 一個明確的目標檔案或模組。
2. 一組明確的輸入與輸出，採函式簽名、CLI 或服務介面描述。
3. 一組可自動驗證的驗收準則，優先對應 pytest。

### 1.2 API 邊界決策

SDD 的 MVP 採 Streamlit 與 Python 應用服務層，不新增 HTTP Controller 或 REST API。Phase 2 的 Service API 是穩定的內部函式合約，Phase 4 只能透過這些服務讀取資料。若日後需支援外部客戶端，再以獨立需求新增 FastAPI Controller 與 HTTP Endpoint。

### 1.3 Phase 閘門

| 閘門 | 可進入條件 |
| --- | --- |
| Phase 1 | Phase 0 的設定、依賴、格式檢查與 CI 均可運作。 |
| Phase 2 | Migration 成功，ORM 與 Repository 可用，資料庫測試通過。 |
| Phase 3 | 核心研究流程可在固定 fixture 上產生可重現 OOS 結果。 |
| Phase 4 | 外部資料同步與月度訊號工作可由 CLI 執行。 |
| Phase 5 | Dashboard 能只讀方式顯示已完成 run 的資料。 |
| Release | 可觀測性、安全檢查與端對端驗收均通過。 |

### 1.4 關鍵路徑

~~~text
P0-01 -> P0-02 -> P0-03 -> P0-04
P0-04 -> P1-01 -> P1-02 -> P1-06
P1-02 -> P2-02 -> P2-03 -> P2-04 -> P2-06
P2-06 -> P2-08 -> P2-09 -> P2-10 -> P2-11 -> P2-12
P2-12 -> P2-13 -> P2-14 -> P2-15 -> P2-16
P2-16 -> P3-05 -> P3-06 -> P4-01 -> P4-02
P4-02 -> P5-01 -> P5-06 -> Release
~~~

## 2. Phase 0 骨架與環境設定

### [x] P0-01 建立 Python 專案描述

- **目標**：pyproject.toml
- **相依**：無
- **輸入與輸出**：輸入為專案名稱與 Python 最低版本；輸出為可被工具讀取的 package metadata、pytest 與覆蓋率設定。
- **驗收準則**：
  - python -m pytest 可辨識 tests 目錄。
  - pyproject 內指定 src 為 package source root。
  - 測試：tests/test_project_metadata.py 驗證套件名稱與最低 Python 版本。
- **狀態**：已完成（2026-09-21），`python -m pytest -v` 4 passed。
- **更動重點**：新增 `pyproject.toml`（`taiwan-quant-xgb`、`requires-python >=3.10`、`packages.find.where=["src"]`、`pytest.testpaths=["tests"]`、`coverage.source=["src"]`）、`src/__init__.py`、`tests/__init__.py`、`tests/test_project_metadata.py`。
- **技術注意事項**：本機為 Python 3.12，測試用 `tomllib` 讀取；為相容 `>=3.10` 保留 `tomli` fallback。`dependencies` 留空，套件版本固定留待 P0-02；lint 規則留待 P0-06。

### [x] P0-02 固定執行相依

- **目標**：requirements.txt
- **相依**：P0-01
- **輸入與輸出**：輸入為 SDD 指定的 Python 套件；輸出為包含 pandas、numpy、SQLAlchemy、Alembic、xgboost、optuna、vectorbt、streamlit、shap、pytest、ruff 的固定版本清單。
- **驗收準則**：
  - pip install -r requirements.txt 可在乾淨環境完成。
  - pip check 無相依衝突。
  - 測試：CI 的 dependency installation job 成功。
- **狀態**：已完成（2026-09-21），`pip install -r requirements.txt` + `pip check` 乾淨，`python -m pytest -v` 8 passed。
- **更動重點**：新增 `requirements.txt`（16 個 `==` 固定版本：SDD §5.1 + Roadmap 聯集 + `yfinance`）、`tests/test_requirements.py`（套件存在、版本合法、已安裝一致、`pip check`）。
- **技術注意事項**：以 Python 3.12.10 / Windows 實測解析版本（如 `pandas==3.0.6`、`numpy==2.5.3`、`xgboost==3.4.1`、`vectorbt==1.1.0`、`streamlit==1.64.0`、`shap==0.52.0`）；未改 `pyproject.toml dependencies`（仍留空，待後續整合）；`yfinance==1.7.0` 為 SDD `fallback_source` 補列，不在 SDD §5.1 原清單內。

### [x] P0-03 建立研究設定基線

- **目標**：config.yaml
- **相依**：P0-01
- **輸入與輸出**：輸入為 SDD 的 universe、feature、label、validation、portfolio、execution 預設值；輸出為完整 YAML 設定檔。
- **驗收準則**：
  - 有資料來源、宇宙、特徵、驗證、投組與成本六個設定節點。
  - purge trading days 與 label horizon 均為 20。
  - 測試：tests/test_settings.py 載入 YAML 並驗證必填欄位與數值範圍。
- **狀態**：已完成（2026-09-21），`pytest tests/test_settings.py -v` 4 passed，全量 12 passed。
- **更動重點**：新增 `config.yaml`（SDD §5.2 原樣 8 節點）、`tests/test_settings.py` 基線版（節點存在、purge/horizon=20、數值範圍、無 secrets）。
- **技術注意事項**：`test_settings.py` 採基線範圍，typed `load_settings` 留待 P0-04 擴充同一檔案；`config.yaml` 僅含 `FINMIND_TOKEN` 名稱，`TELEGRAM/GEMINI` 留待 P0-05；解析用 `PyYAML==6.0.3`（P0-02 已固定）。

### [x] P0-04 實作設定載入與驗證

- **目標**：src/settings.py
- **相依**：P0-03
- **輸入與輸出**：load_settings(path: str, env: Mapping) -> Settings；輸入 YAML 路徑與環境變數，輸出不可變的 typed Settings。
- **驗收準則**：
  - 缺少必要設定、負數成本或不合法 Top N 時拋出明確例外。
  - Token 僅從環境變數名稱讀取，不輸出 token 值。
  - 測試：tests/test_settings.py 覆蓋成功、缺值與不合法數值。
- **狀態**：已完成（2026-09-21），`pytest tests/test_settings.py -v` 12 passed，全量 20 passed。
- **更動重點**：新增 `src/settings.py`（frozen `Settings` + 8 巢狀 section、`SettingsError`、`get_finmind_token`）、擴充 `tests/test_settings.py`（保留 P0-03 基線 4 項，追加成功/不可變、缺鍵、非法值、hold_rank、token 隔離共 8 項）。
- **技術注意事項**：僅用 stdlib `dataclasses`（`pydantic` 不在 P0-02 清單）；`load_settings(path, env)` 的 `env` 僅為介面相容，token 值不寫入 `Settings`、`repr` 已驗證不洩漏；後續 P0-05 的三個 token 名稱以 `settings.data.finmind_token_env` 為一致性基準。

### [x] P0-05 建立環境變數範本

- **目標**：.env.example
- **相依**：P0-04
- **輸入與輸出**：輸入為外部服務所需 token 名稱；輸出為僅含 FINMIND_TOKEN、TELEGRAM_BOT_TOKEN、GEMINI_API_KEY 名稱的範本。
- **驗收準則**：
  - 不含任何真實 token、密碼或帳號。
  - 名稱與 Settings 讀取的環境變數一致。
  - 測試：tests/test_secret_hygiene.py 檢查範本值為空白或明確 placeholder。
- **狀態**：已完成（2026-09-21），`pytest tests/test_secret_hygiene.py -v` 3 passed，全量 23 passed。
- **更動重點**：新增 `.env.example`（三個空值 key + 不提交註解）、`tests/test_secret_hygiene.py`（名稱齊備、值空白/placeholder、`FINMIND_TOKEN` 對上 `Settings.data.finmind_token_env`）。
- **技術注意事項**：未改 `src/settings.py`；`TELEGRAM/GEMINI` 為 P3-08/P3-09 預留名，啟用檢查留待 P5-05；工作區確認無 `.env` 真檔。

### [x] P0-06 建立格式與靜態檢查規則

- **目標**：ruff.toml
- **相依**：P0-01、P0-02
- **輸入與輸出**：輸入為 Python lint、import 與 format 規則；輸出為 ruff 可執行設定。
- **驗收準則**：
  - ruff check src tests 通過。
  - ruff format --check src tests 通過。
  - CI 對違反 lint 的 fixture commit 會失敗。
- **狀態**：已完成（2026-09-21），`ruff check` / `ruff format --check` 皆通過，全量 `pytest -q` 23 passed。
- **更動重點**：新增 `ruff.toml`（`py310`、`line-length=100`、`select=E/F/I/W/UP/B`）；格式層級修正 `src/settings.py`、`tests/test_secret_hygiene.py`（`ruff format` 重排）、`tests/test_requirements.py`（B007 未使用變數改 `.values()`），無邏輯變更。
- **技術注意事項**：`line-length=100` 為容納既有最長行（97 字元）所選；`D/ANN/S` 等嚴格規則刻意不開；負向對照已在 TEMP 驗證違規檔會被擋（E401/I001/F401）；完整 CI 閘門留待 P0-08。

### [x] P0-07 建立測試 Fixture 基礎

- **目標**：tests/conftest.py
- **相依**：P0-02、P0-04
- **輸入與輸出**：輸入為臨時 SQLite 路徑與固定交易日資料；輸出為 temp database、settings、sample prices fixtures。
- **驗收準則**：
  - 每個測試使用獨立資料庫。
  - fixture 交易日包含跨月與 20 個以上交易日。
  - 測試：tests/test_fixtures.py 確認 fixture 不共享狀態。
- **狀態**：已完成（2026-09-21），`pytest tests/test_fixtures.py -v` 4 passed，全量 27 passed，`ruff check`/`format --check` 通過。
- **更動重點**：新增 `tests/conftest.py`（`temp_db_path`、`temp_db_conn`、`settings`、`trading_days` 25 平日、`sample_prices` 2 檔×25 日）、`tests/test_fixtures.py`（獨立性、跨月 20+、價格合法、`settings` 不可變）。
- **技術注意事項**：`tmp_path` function-scope 保證 DB 獨立；交易日/價格全公式生成零隨機（2020-01-02 起平日）；`stock_id` 文字型別保前導零；DB fixture 僅 stdlib `sqlite3`，SQLAlchemy engine 留待 P1-02；`Iterator` 取自 `collections.abc`（UP035）。

### [x] P0-08 建立 CI 工作流程

- **目標**：.github/workflows/ci.yml
- **相依**：P0-02、P0-06、P0-07
- **輸入與輸出**：輸入為推送與 pull request 事件；輸出為安裝、lint、unit test 與最小 smoke test 工作。
- **驗收準則**：
  - CI 依序執行依賴安裝、ruff、pytest。
  - secrets 不輸出至 log。
  - 測試：在 GitHub Actions 或等價本機 runner 完整執行一次。
- **狀態**：已完成（2026-09-21），等價本機 runner 全流程通過，全量 `pytest -q` 31 passed。
- **更動重點**：新增 `.github/workflows/ci.yml`（`push`/`PR`、`ubuntu-latest` + Python 3.12，`pip install` → `pip check` → `ruff check` → `format --check` → `pytest -q`）、`tests/test_ci_workflow.py`（觸發、步驟順序、無 secret、版本固定）。
- **技術注意事項**：工作區非 git repo，以等價本機流程驗證（GitHub Actions 待首次推送後確認）；測試相容 YAML 1.1 `on:` 解析為 `True`；`.gitignore`（`.env`/`*.db`）留待 P5-06；Phase 0 閘門已滿足，可進入 Phase 1。

## 3. Phase 1 資料與領域模型

### [x] P1-01 建立初始資料庫 Migration

- **目標**：database/migrations/versions/001_initial_schema.py
- **相依**：P0-02、P0-04
- **輸入與輸出**：upgrade() -> None 與 downgrade() -> None；輸入為 SQLite 資料庫，輸出為 stocks、prices、financials、institutional、pipeline runs、features、predictions、signals、positions、orders、portfolio daily 資料表與索引。
- **驗收準則**：
  - upgrade 後所有 SDD 資料表與索引存在。
  - downgrade 後 schema 回到空白。
  - 測試：tests/test_migrations.py 對新的 SQLite 執行 upgrade 和 downgrade。
- **狀態**：已完成（2026-09-21），`tests/test_migrations.py` 8 passed，全量 `pytest -q` 39 passed。
- **更動重點**：新增 `001_initial_schema.py`（`TABLES` 11 表、`INDEXES` 5 索引、`INITIAL_SCHEMA_SQL` SDD §6.2 逐字＋`IF NOT EXISTS`，`upgrade`/`downgrade` 皆冪等）、`tests/test_migrations.py`（建表、建索引、逐表欄位比對、冪等、downgrade 回空白＋可重跑、prices PK/FK/CHECK、financials PIT、run status 與 orders 時序抽查）。
- **技術注意事項**：僅用 stdlib `sqlite3`（Alembic/SQLAlchemy 留待 P1-02）；`upgrade/downgrade(conn)` 取連線而非路徑，沿用 `temp_db_conn`（FK 由呼叫方 `PRAGMA` 開啟，不寫進 script）；`database/` 非套件，測試以 `importlib` 按路徑載入，未改 `pythonpath`；未建 `database/schema.sql`，migration 為唯一 DDL 來源；完整約束矩陣留給 P1-03～P1-05 ORM 測試。

### [x] P1-02 建立資料庫連線與交易邊界

- **目標**：src/database.py
- **相依**：P1-01、P0-07
- **輸入與輸出**：create_engine_from_settings(settings: Settings) -> Engine；session_scope(engine: Engine) -> ContextManager[Session]。
- **驗收準則**：
  - session 成功時 commit，例外時 rollback。
  - SQLite foreign keys 在每個 connection 開啟。
  - 測試：tests/test_database.py 驗證 rollback 與外鍵限制。
- **狀態**：已完成（2026-09-21），`tests/test_database.py` 6 passed，全量 `pytest -q` 45 passed。
- **更動重點**：新增 `src/database.py`（僅支援 sqlite，他種 scheme 拋 `ValueError`；`connect` listener 逐連線開 FK；`check_same_thread=False`；檔案型 URL 自動建父目錄；`session_scope` 成功 commit、例外 rollback 重拋、finally close）、`tests/test_database.py`（URL 接線、commit、rollback＋重拋、FK＋`dispose` 後新連線仍開 FK、非 sqlite 拒絕、父目錄建立）。
- **技術注意事項**：engine 不跑 migration（建表仍屬 P1-01，測試端先 `upgrade` temp 檔再開 engine）；`database_url` 覆寫用 `dataclasses.replace`（`Settings` frozen，不改 `src/settings.py`）；相對路徑維持 CWD 語義；P1-03 起 ORM/Repository 以 `session_scope` 為交易邊界。

### [x] P1-03 實作證券主檔 ORM

- **目標**：src/models/security.py
- **相依**：P1-02
- **輸入與輸出**：Stock ORM；輸入 stock id、market、上市與下市日期，輸出可持久化的 Stock 實體。
- **驗收準則**：
  - stock id 是唯一主鍵。
  - delisted date 可為空，但日期欄位可正常序列化。
  - 測試：tests/test_models_security.py 驗證建立、查詢與主鍵衝突。
- **狀態**：已完成（2026-09-21），`tests/test_models_security.py` 5 passed，全量 `pytest -q` 50 passed。
- **更動重點**：新增 `src/models/__init__.py`（共用 `Base`）、`src/models/security.py`（`Stock`，2.0 typed mapping）、`tests/test_models_security.py`（建立查詢、主鍵衝突、可空下市＋序列化、`market` 非空、ORM 對 migration 欄位一致）。
- **技術注意事項**：日期欄用 TEXT 存 ISO 字串（SDD §5.3，保前導零、可 JSON 序列化）；ORM 薄映射、約束由 DB 強制；`relationship` 留待 P1-04/P1-05 子表出現後回補；測試內接住 `flush` 的 `IntegrityError` 後須先 `session.rollback()` 再離開 `session_scope`，否則 `commit` 拋 `PendingRollbackError`（用法慣例，後續 ORM 測試比照）。

### [x] P1-04 實作市場與基本面 ORM

- **目標**：src/models/market.py
- **相依**：P1-02、P1-03
- **輸入與輸出**：Price、Financial、Institutional ORM；輸入標準化市場資料，輸出具有複合主鍵與 foreign key 的實體。
- **驗收準則**：
  - Price 的 trade date 與 stock id 不可重複。
  - Financial 的 available date 不可早於 announcement date。
  - 測試：tests/test_models_market.py 覆蓋有效與無效資料。
- **狀態**：已完成（2026-09-21），`tests/test_models_market.py` 6 passed，全量 `pytest -q` 56 passed。
- **更動重點**：新增 `src/models/market.py`（`Price` 複合 PK、OHLC 量額 `Float` NOT NULL；`Financial` 三鍵複合 PK、數值全可空；`Institutional` 複合 PK、數值全可空；三者 `stock_id` 皆 FK）、`tests/test_models_market.py`（建查、複合 PK、FK＋`open>0`、PIT、籌碼可空、三表欄位一致）。
- **技術注意事項**：CHECK 留給 migration DDL，ORM 不重寫（與 P1-03 一致）；未加 `relationship`（需改 `security.py`，留待有導航查詢需求時再加）；`Float`=REAL、日期 TEXT；測試沿用 `flush` 後 `rollback` 慣例。

### [x] P1-05 實作研究與交易 ORM

- **目標**：src/models/research.py
- **相依**：P1-02、P1-03
- **輸入與輸出**：PipelineRun、Feature、Prediction、Signal、Position、Order、PortfolioDaily ORM；輸入為研究與回測資料，輸出可追溯的資料關係。
- **驗收準則**：
  - Prediction 與 Order 都須關聯到 PipelineRun。
  - Order 的 execution date 必須晚於 signal date。
  - 測試：tests/test_models_research.py 驗證約束與關聯。
- **狀態**：已完成（2026-09-21），`tests/test_models_research.py` 6 passed，全量 `pytest -q` 62 passed。
- **更動重點**：新增 `src/models/research.py`（7 ORM；`Feature` 三鍵 PK＋30 可空因子；`Prediction`/`Order` 以 FK 關聯 run）、`tests/test_models_research.py`（run 三態、Prediction 孤兒擋下＋join 追溯、Order 時序、Feature PK＋`missing_flag` 預設 0、Signal/Position 非法擋下、7 表欄位一致）。
- **技術注意事項**：FK 表達關聯、CHECK 留 DDL（與 P1-03/P1-04 一致）；`missing_flag` 僅以 `server_default` 對齊 DDL `DEFAULT 0`；ORM 層至此齊備，P1-06 起 Repository 可直接取用 `Stock`/`Price`/`Financial`/`Institutional`/`PipelineRun` 等實體。

### [x] P1-06 建立股票主檔 Repository

- **目標**：src/repositories/stocks.py
- **相依**：P1-03
- **輸入與輸出**：upsert_stocks(session, rows: DataFrame) -> int；get_active_stocks(session, as_of: date) -> DataFrame。
- **驗收準則**：
  - 同一 stock id 重跑只更新、不新增重複列。
  - get active stocks 排除已下市股票。
  - 測試：tests/test_repositories_stocks.py。
- **狀態**：已完成（2026-09-21），`tests/test_repositories_stocks.py` 5 passed，全量 `pytest -q` 67 passed。
- **更動重點**：新增 `src/repositories/__init__.py`、`src/repositories/stocks.py`（ON CONFLICT upsert、NaN→None、fail-fast 驗證、active 上市判斷、六欄排序回傳）、`tests/test_repositories_stocks.py`（冪等＋`created_at` 不變、空表、驗證、NaN、active 五情境）。
- **技術注意事項**：確立 P1-06～P1-11 共用慣例——repository 取 active `Session`、只 `flush` 不 `commit`（邊界屬 `session_scope`）；回傳列數定義為輸入列數；`as_of` 取 `date` 轉 ISO 比較；`market` 等 NOT NULL 違規留給 DB 報錯。

### [x] P1-07 建立價格 Repository

- **目標**：src/repositories/prices.py
- **相依**：P1-04
- **輸入與輸出**：upsert_prices(session, rows: DataFrame) -> int；load_prices(session, stock_ids, start, end) -> DataFrame。
- **驗收準則**：
  - 載入結果依 stock id、trade date 排序。
  - 重跑同一批資料不改變列數。
  - 測試：tests/test_repositories_prices.py。
- **狀態**：已完成（2026-09-21），`tests/test_repositories_prices.py` 5 passed，全量 `pytest -q` 72 passed。
- **更動重點**：新增 `src/repositories/_frames.py`（共用 `to_records`：驗證＋NaN→None）、`src/repositories/prices.py`（複合鍵 upsert、閉區間排序讀取、空集合短路）、`tests/test_repositories_prices.py`（排序讀回、冪等更新、驗證、篩選、FK 穿透）。
- **技術注意事項**：`_frames.py` 供 P1-08～P1-11 沿用，`stocks.py` 維持原狀（收斂另開清理任務）；排序斷言用 `itertuples` 以避 ruff B905 `zip(strict)`；`load` 的 `start/end` 取 `date`；測試父列用 P1-06 `upsert_stocks` 種、價格用 `sample_prices` fixture。

### [x] P1-08 建立財報與籌碼 Repository

- **目標**：src/repositories/fundamentals.py
- **相依**：P1-04
- **輸入與輸出**：upsert_financials(session, rows) -> int；upsert_institutional(session, rows) -> int；load_pit_financials(session, as_of) -> DataFrame。
- **驗收準則**：
  - PIT 查詢只回傳 available date 不晚於 as of 的資料。
  - 同一 stock id 取最新 available date。
  - 測試：tests/test_repositories_fundamentals.py。
- **狀態**：已完成（2026-09-21），`tests/test_repositories_fundamentals.py` 6 passed，全量 `pytest -q` 78 passed。
- **更動重點**：新增 `src/repositories/fundamentals.py`（財報三鍵／籌碼兩鍵 upsert、window-function PIT 全欄讀取）、`tests/test_repositories_fundamentals.py`（雙冪等、驗證、PIT 可用性、每檔一列＋同日公告晚者勝、PIT CHECK 穿透）。
- **技術注意事項**：PIT 以 `ROW_NUMBER() PARTITION BY` 一次查回（SDD §6.3，無 N+1），`as_of` 參數化傳入；同 `available_date` 取 `announcement_date` 晚者（不同 `report_period` 才可能並存，受 PK 約束）；`load_pit` 回全體不分頁，宇宙交集由 P2-03 處理。

### [x] P1-09 建立特徵與預測 Repository

- **目標**：src/repositories/research.py
- **相依**：P1-05
- **輸入與輸出**：save_features(session, rows) -> int；save_predictions(session, rows) -> int；load_predictions(session, date, model version) -> DataFrame。
- **驗收準則**：
  - Feature 以 rebalance date、stock id、feature version 去重。
  - Prediction 排名可依機率正確讀回。
  - 測試：tests/test_repositories_research.py。
- **狀態**：已完成（2026-09-21），`tests/test_repositories_research.py` 5 passed，全量 `pytest -q` 83 passed。
- **更動重點**：新增 `src/repositories/research.py`（`FACTOR_COLUMNS` 30 因子常數、雙 `save_*` 三鍵冪等、`load_predictions` 機率 DESC 讀回）、`tests/test_repositories_research.py`（雙冪等、`missing_flag` 預設 0、排序與 rank 一致、空結果欄名、驗證＋孤兒 run）。
- **技術注意事項**：`rank` 由呼叫方（P2-08）寫入，本層只持久化；`missing_flag` 未給時靠 DDL `DEFAULT 0`；空 `model_version` fail-fast 防全表誤讀；30 因子欄名以 `FACTOR_COLUMNS` 常數集中管理，供 P2-04/P2-05 沿用。

### [x] P1-10 建立部位與訂單 Repository

- **目標**：src/repositories/trading.py
- **相依**：P1-05
- **輸入與輸出**：save_signals(session, rows) -> int；save_positions(session, rows) -> int；save_orders(session, rows) -> int。
- **驗收準則**：
  - 訂單成本欄位與執行日期完整保存。
  - 部位不可存負數 shares 或 weight。
  - 測試：tests/test_repositories_trading.py。
- **狀態**：已完成（2026-09-21），`tests/test_repositories_trading.py` 4 passed，全量 `pytest -q` 87 passed。
- **更動重點**：新增 `src/repositories/trading.py`（signals／positions 兩鍵冪等；orders 以 `(run_id, signal_date)` 範圍取代）、`tests/test_repositories_trading.py`（雙冪等、orders 縮減＋時序＋空 ID＋混批、驗證＋空表）。
- **技術注意事項**：實作時修正原規劃——`save_orders` 改為範圍取代（刪同批 `order_id` 會在縮減重跑時殘留舊單）；批次限單一 run／signal（混批 `ValueError`）；成本四欄原樣持久化、不重算（計算屬 P2-13）；`order_id` 編碼留給 P2-13/P3-06。

### [x] P1-11 建立 Pipeline Run Repository

- **目標**：src/repositories/runs.py
- **相依**：P1-05
- **輸入與輸出**：start_run(session, metadata) -> PipelineRun；finish_run(session, run id, status, error) -> PipelineRun。
- **驗收準則**：
  - started、succeeded、failed 狀態轉換合法。
  - failed run 保留錯誤摘要。
  - 測試：tests/test_repositories_runs.py。
- **狀態**：已完成（2026-09-21），`tests/test_repositories_runs.py` 6 passed，全量 `pytest -q` 93 passed。
- **更動重點**：新增 `src/repositories/runs.py`（dict 輸入 fail-fast、`started→succeeded|failed` 狀態機、failed 必帶 error、終止態不可變）、`tests/test_repositories_runs.py`（建 run、預設時間、成功、失敗摘要、非法轉換、輸入驗證）。
- **技術注意事項**：與 DataFrame 系 repository 不同，本層以 ORM 物件進出、回傳 attached 物件；重複 `run_id` 的 `IntegrityError` 發生在 `start_run` 內部 `flush`（測試斷言須包住呼叫）；重跑同一 run 須用新 `run_id`；Phase 1 至此全部完成，Phase 2 閘門（Migration＋ORM＋Repository）已滿足。

## 4. Phase 2 核心業務邏輯與應用服務 API

### [x] P2-01 定義跨模組資料合約

- **目標**：src/contracts.py
- **相依**：P0-04、P1-05
- **輸入與輸出**：UniverseSnapshot、FeatureSet、ModelArtifact、PortfolioTarget、BacktestResult dataclass；輸入為原始值，輸出為 type checked domain contract。
- **驗收準則**：
  - 每個 contract 都具備 run id 或日期等必要追溯欄位。
  - 不合法權重、空日期或負 Top N 會被拒絕。
  - 測試：tests/test_contracts.py。
- **狀態**：已完成（2026-09-22），`tests/test_contracts.py` 10 passed，全量 `pytest -q` 103 passed。
- **更動重點**：新增 `src/contracts.py`（`UniverseEntry/UniverseSnapshot` 含保留排除原因、`FeatureSet` 含 coverage＋missing flag 欄檢查、`ModelArtifact` 對齊 SDD §10.2、`PortfolioTarget` 權重加總恆等式、`BacktestResult` 含 NAV／orders 檢查）、`tests/test_contracts.py`（5 合約 happy path＋拒絕案例）。
- **技術注意事項**：合約層零依賴（不 import settings／ORM，防循環）；`frozen`＋含 DataFrame 者 `eq=False`；`weights keys == {BUY,HOLD}` 且 `sum≈exposure、cash≈1-exposure`（eps 1e-9）；空宇宙合法（停機決策留服務層）；NAV 全 >= 0（long-only 無槓桿）。

### [x] P2-02 實作可交易宇宙服務

- **目標**：src/services/universe_service.py
- **相依**：P1-06、P1-07、P2-01
- **輸入與輸出**：build_universe(as_of: date, settings: Settings) -> UniverseSnapshot。
- **驗收準則**：
  - 套用上市、下市、股價、市值、20 日均成交額與排除旗標。
  - 回傳每檔股票的保留或排除原因。
  - 測試：tests/test_universe_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_universe_service.py` 6 passed，全量 `pytest -q` 109 passed。
- **更動重點**：新增 `src/services/__init__.py`、`src/services/universe_service.py:build_universe(prices, stocks, as_of, settings, run_id)`（SDD §8 六關卡順序篩選、首敗關卡為機器可讀 reason）、`tests/test_universe_service.py`（全通過、六排除原因、歷史不足、歷史視角、非法輸入）。
- **技術注意事項**：簽名較路線圖多 `prices/stocks` DataFrame 參數（服務保持 DB 無關、可測；DB 接線留 P2-16）；`market_cap` 缺欄 fail-loud（`market_cap:missing` 全排除，不靜默跳過）；20 日窗不足整窗即排除（防新上市灌水）；下市前快照仍可入選（無倖存者偏差）；`prices` 需含 `close/traded_value`、`stocks` 需含上市／下市日。

### [x] P2-03 實作 Point-in-Time 快照服務

- **目標**：src/services/pit_service.py
- **相依**：P1-07、P1-08、P2-02
- **輸入與輸出**：build_pit_snapshot(universe: UniverseSnapshot, as_of: date) -> DataFrame。
- **驗收準則**：
  - 每一財務欄位資料的 available date 不晚於 as of。
  - 每個 stock id 最多採用一筆最新財務資料。
  - 測試：tests/test_pit_service.py 包含故意延後公告資料。
- **狀態**：已完成（2026-09-22），`tests/test_pit_service.py` 5 passed，全量 `pytest -q` 114 passed。
- **更動重點**：新增 `src/services/pit_service.py:build_pit_snapshot`（DataFrame 版 PIT join：財務 `available_date<=as_of` 取最新、籌碼 `trade_date<=as_of` 取最新、併當日收盤為 `as_of_close`）、`tests/test_pit_service.py`（形狀、延後公告陷阱、缺財報留 NaN、空宇宙、日期錯配）。
- **技術注意事項**：簽名較路線圖多三個 DataFrame 參數（服務 DB 無關，DB 接線留 P2-16）；排序鍵與 P1-08 視窗函數同序；籌碼以 `trade_date` 為 PIT 鍵（該表無公告延遲欄）；宇宙內無財報股票保留 NaN（缺值由 P2-04 設 missing flag）；`universe.as_of` 與參數不一致即 `ValueError`。

### [x] P2-04 實作原始因子服務

- **目標**：src/services/feature_service.py
- **相依**：P2-03、P1-07、P1-08
- **輸入與輸出**：calculate_raw_features(snapshot: DataFrame, as_of: date) -> DataFrame。
- **驗收準則**：
  - 產生 SDD 定義的 30 個因子欄位。
  - 僅使用 as of 前資料；分母為零不產生無限值。
  - 測試：tests/test_feature_service.py 覆蓋動能、波動、估值與籌碼因子。
- **狀態**：已完成（2026-09-22），`tests/test_feature_service.py` 6 passed，全量 `pytest -q` 120 passed。
- **更動重點**：新增 `src/services/feature_service.py`（`FACTOR_COLUMNS` 30 欄、`calculate_raw_features`：價格窗 14 欄＋財報 10 欄＋籌碼 6 欄、`missing_flag` 任一 NaN 即 1、出口 inf→NaN 最終防線）、`tests/test_feature_service.py`（30 欄齊備、手算動能／YoY／籌碼對照、除零無 inf、負估值遮罩、未來價格不洩漏、缺 TAIEX）。
- **技術注意事項**：簽名較路線圖多 `prices/financials/institutional` 歷史參數（後兩者可選，缺則對應因子 NaN＋flag）；估值＝最新公告值÷當期市值（`as_of_close×float_shares`，MVP 近似）；YoY 取第 5 新公告、MoM／QoQ 取相鄰公告（免 period 解析、PIT 安全）；`dividend_yield` 尚無 ETL 來源恆 NaN（覆蓋率由 P2-05 揭露）；窗不足一律 NaN 不降級；`frame[tuple]` 會被 pandas 誤判單鍵，取欄須 `list(FACTOR_COLUMNS)`。

### [x] P2-05 實作橫截面前處理服務

- **目標**：src/services/feature_preprocess_service.py
- **相依**：P2-04、P0-04
- **輸入與輸出**：preprocess_features(raw: DataFrame, settings: Settings) -> FeatureSet。
- **驗收準則**：
  - 對單月資料做 1% 至 99% winsorize、rank 與標準化。
  - 絕對相關係數大於 0.85 的因子保留一個。
  - missing flag 與因子覆蓋率一併輸出。
  - 測試：tests/test_feature_preprocess_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_feature_preprocess_service.py` 5 passed，全量 `pytest -q` 125 passed。
- **更動重點**：新增 `src/services/feature_preprocess_service.py:preprocess_features`（中位數填補→winsorize→rank pct→z-score→貪婪去重，回傳 `FeatureSet`）、`tests/test_feature_preprocess_service.py`（合約形狀、中位數填補＋覆蓋率、離群收斂、去重性質、非法輸入）。
- **技術注意事項**：簽名較路線圖多 `run_id/as_of`（`FeatureSet` 追溯必填）；去重保留 `FACTOR_COLUMNS` 先出現者（確定性）；常數欄 z 全 0 不除零；全缺欄丟棄且不進 coverage；winsorize 測試需 N=200（N=80 時 p99 內插落兩離群之間無法收斂）；`missing_flag` 透傳不重算。

### [x] P2-06 實作標籤服務

- **目標**：src/services/label_service.py
- **相依**：P1-07、P2-02、P0-04
- **輸入與輸出**：build_labels(universe: UniverseSnapshot, as_of: date, horizon: int) -> Series。
- **驗收準則**：
  - 使用未來第 20 個交易日，不以 calendar day 取代。
  - 橫截面最高 20% 為 1，其餘為 0。
  - 資料不足 20 個交易日時，該日期不產生 label。
  - 測試：tests/test_label_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_label_service.py` 5 passed，全量 `pytest -q` 130 passed。
- **更動重點**：新增 `src/services/label_service.py:build_labels(prices, stock_ids, as_of, settings)`（各股自身 t+20 位置法、任一缺數整期回空、分位嚴格大於切割）、`tests/test_label_service.py`（分位手算、日曆陷阱、缺期回空、宇宙外忽略、非法輸入）。
- **技術注意事項**：簽名以 `prices＋stock_ids` 取代路線圖 `universe`（服務 DB 無關；P2-16 組裝時傳 `snapshot.included_ids`）；`horizon/top_quantile` 取自 `Settings.label` 而非參數（防 P2-10 誤傳，呼應 SDD §5.2 唯一入口）；Series name＝`as_of` ISO 利 concat；零／負收盤視為壞數整期回空。

### [x] P2-07 實作基準模型服務

- **目標**：src/services/baseline_service.py
- **相依**：P2-05、P2-06
- **輸入與輸出**：score_baselines(features: FeatureSet) -> DataFrame。
- **驗收準則**：
  - 輸出 Momentum 20D、Momentum 20D 加 60D、等權因子排名、Logistic Regression 四種分數。
  - 每種分數在同日可轉為唯一 rank。
  - 測試：tests/test_baseline_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_baseline_service.py` 4 passed，全量 `pytest -q` 134 passed。
- **更動重點**：新增 `src/services/baseline_service.py:score_baselines`（動量直取、動能相加、現存欄等權、logit 正類機率）、`tests/test_baseline_service.py`（形狀手算、logit 有序性、缺腿 NaN、非法輸入、唯一 rank）。
- **技術注意事項**：簽名多可選 `labels`（無 label 單月無法擬合，該欄全 NaN；P2-10 傳訓練 label）；`momentum_60d` 被去重時複合欄整欄 NaN（名實相符）；等權用現存欄平均（強健）；logit 單類／缺對齊回 NaN；sklearn import 使該檔約 12s，日後拆檔可加速。

### [x] P2-08 實作 XGBoost 模型服務

- **目標**：src/services/xgb_service.py
- **相依**：P2-05、P2-06、P2-01
- **輸入與輸出**：train_xgb(train_x, train_y, valid_x, valid_y, params) -> ModelArtifact；predict_xgb(artifact, features) -> DataFrame。
- **驗收準則**：
  - artifact 保存 feature columns、model version、參數與 validation Rank IC。
  - predict 回傳 stock id、prediction probability、rank。
  - 測試：tests/test_xgb_service.py 使用小型固定資料訓練與預測。
- **狀態**：已完成（2026-09-22），`tests/test_xgb_service.py` 5 passed，全量 `pytest -q` 139 passed。
- **更動重點**：新增 `src/services/xgb_service.py`（`DEFAULT_PARAMS` SDD §10.4 八參數、`train_xgb` 回傳 artifact＋booster、`predict_xgb` 回傳 probability＋唯一 rank、`rank_ic` Spearman）、`tests/test_xgb_service.py`（artifact 齊備、rank 唯一有序、欄位漂移拒收、壞參數拒收、IC 手算）。
- **技術注意事項**：簽名較路線圖多版本字串／run_id（合約追溯必填）且 `train` 另回 booster（frozen 合約不藏模型本體，`predict` 顯式傳）；`params` 缺鍵補預設、未知鍵拒收；Spearman 用 pandas 免新依賴；`random_state` 缺省 42（P2-09 改傳 settings）；MVP 無 early stopping；P2-16 smoke 補記——退化驗證（常數分數／單類）的 NaN IC 在 `train_xgb` 內墊底 -1.0（P2-09 原 floor 寫在 artifact 構造後永遠跑不到，見 `test_train_xgb_degenerate_validation_floors_to_minus_one`）。

### [x] P2-09 實作 Optuna 最佳化服務

- **目標**：src/services/optimization_service.py
- **相依**：P2-08、P0-04
- **輸入與輸出**：optimize_xgb(train, valid, settings) -> ModelArtifact。
- **驗收準則**：
  - 僅用 Validation Mean Rank IC 做主要 objective。
  - trial 數讀取 config，MVP 為 50。
  - 每次 trial 參數與結果保存為 artifact。
  - 測試：tests/test_optimization_service.py 驗證 Test 資料不會傳入。
- **狀態**：已完成（2026-09-22），`tests/test_optimization_service.py` 3 passed，全量 `pytest -q` 142 passed。
- **更動重點**：新增 `src/services/optimization_service.py:optimize_xgb`（SDD §10.4 八參數搜尋域、TPESampler 種子化、回傳最佳 artifact＋booster＋trial_log）、`tests/test_optimization_service.py`（雙 trial 跑通、trial 數讀 config、簽名無 test）。
- **技術注意事項**：`optimize_xgb` 多回 booster＋trial_log（`ModelArtifact` 不藏本體，每 trial artifact 由 `train_xgb` 產生）；NaN objective 墊底 -1.0（Optuna 拒 NaN）；最佳並列取首個；測試以 monkeypatch 縮 `n_estimators`、以 `dataclasses.replace` 縮 trial 數（frozen 不可 setattr）。

### [x] P2-10 實作 Walk-forward 服務

- **目標**：src/services/walk_forward_service.py
- **相依**：P2-05、P2-06、P2-09
- **輸入與輸出**：run_walk_forward(features, labels, settings) -> list[ModelArtifact]。
- **驗收準則**：
  - 依 4 年 Train、1 年 Validation、1 年 Test 自動生成合法 folds。
  - 同一日期不跨集合；Purge 等於 20 個交易日。
  - Test 資料完全不傳入 optimize。
  - 測試：tests/test_walk_forward_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_walk_forward_service.py` 4 passed，全量 `pytest -q` 146 passed。
- **更動重點**：新增 `src/services/walk_forward_service.py`（`Fold`／`WalkForwardResult`、`build_folds` 月軸滑窗、`run_walk_forward` 逐 fold 調參與 OOS 串接）、`tests/test_walk_forward_service.py`（雙 fold 不交疊、月不足回空、purge 列數、OOS 形狀、無交集欄報錯）。
- **技術注意事項**：回傳 `WalkForwardResult`（較路線圖 `list[ModelArtifact]` 多 booster＋OOS，P2-14 需 OOS）；purge 以剔除 valid 前最後 1 月近似 20 交易日（月切精度限制）；跨月取特徵交集（各月去重欄可能不同，無交集即錯）；test 月只進 `predict_xgb`（間諜 optimize 驗列數）；`optimize` 可注入假函式利測。

### [x] P2-11 實作排名緩衝與目標持股服務

- **目標**：src/services/portfolio_service.py
- **相依**：P1-10、P2-10、P0-04
- **輸入與輸出**：build_target_holdings(predictions, previous_positions, settings) -> PortfolioTarget。
- **驗收準則**：
  - 新持股 rank 小於或等於 15。
  - 舊持股 rank 小於或等於 30 時維持，超過 30 時賣出。
  - 結果包含 BUY、HOLD、SELL、NONE 與 target stock ids。
  - 測試：tests/test_portfolio_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_portfolio_service.py` 6 passed，全量 `pytest -q` 152 passed。
- **更動重點**：新增 `src/services/portfolio_service.py:build_target_holdings`（四態判定、等權滿曝險／全現金、回傳 `PortfolioTarget`）、`tests/test_portfolio_service.py`（四態、15/16 與 30/31 邊界、缺預測出場、首期、全現金、非法輸入）。
- **技術注意事項**：舊股 rank≤15 仍為 HOLD（非 BUY，免重複下單）；前持股缺預測列即 SELL（下市／剔除）；N＝0 全現金 `exposure=0/cash=1`（合約允許）；等權 `1/N` 由 P2-12 重塑，此處滿曝險。

### [x] P2-12 實作權重與市場風控服務

- **目標**：src/services/risk_service.py
- **相依**：P2-11、P1-07、P0-04
- **輸入與輸出**：apply_risk_controls(target, returns, taiex, settings) -> PortfolioTarget。
- **驗收準則**：
  - 先計算反波動相對權重，再以協方差計算組合波動。
  - 個股權重不高於 10%，總股票曝險不高於 100%。
  - TAIEX 小於或等於 MA60 時最終曝險不高於 50%。
  - 協方差不足時回傳可辨識的失敗結果。
  - 測試：tests/test_risk_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_risk_service.py` 6 passed，全量 `pytest -q` 158 passed。
- **更動重點**：新增 `src/services/risk_service.py:apply_risk_controls`（反波動→協方差曝險→10% 截斷再分配→MA60 濾網，`RiskFailure(ValueError)`）、`tests/test_risk_service.py`（等權、反比、上限、MA60、轉 SELL、協方差失敗）。
- **技術注意事項**：σ 缺失股 action→SELL（合約恆等式要求）；N×cap<1 餘額轉現金（exposure 同比縮）；協方差需 ≥30 重疊日無 NaN；TAIEX 史不足亦 `RiskFailure`；測試以 `dataclasses.replace` 放寬上限隔離權重邏輯（少股等權本就觸頂）；P2-16 smoke 補記——`_apply_cap` 遇 N×cap<1 不可行時直接全數封頂（原注水迴圈 10 輪不收斂，曾產出 0.14 超頂權重；迴圈上界改 N+1），見 `test_infeasible_cap_parks_residual_in_cash`。

### [x] P2-13 實作成交與成本服務

- **目標**：src/services/execution_service.py
- **相依**：P2-12、P1-07、P0-04
- **輸入與輸出**：create_orders(target, signal_date, next_open_prices, settings) -> DataFrame。
- **驗收準則**：
  - execution date 為訊號後第一個交易日。
  - Buy executed price 使用 open 乘以 1 加單邊滑價；Sell 使用 open 乘以 1 減單邊滑價。
  - 買賣均收 0.1425% 手續費；賣出另收 0.3% 交易稅。
  - 測試：tests/test_execution_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_execution_service.py` 6 passed，全量 `pytest -q` 164 passed。
- **更動重點**：新增 `src/services/execution_service.py`（SDD §13.2 `executed_price`／`transaction_cost` 原樣落實、`create_orders` 輸出 13 欄對齊 orders 表）、`tests/test_execution_service.py`（價稅手算、時序、HOLD 三態、缺價跳過、非法輸入）。
- **技術注意事項**：簽名多 `current_holdings/portfolio_value/run_id`（股數換算與 order_id 必需）；SELL 全數出清、HOLD 差額調節（含減碼 SELL）；整股四捨五入、零股不下單（SDD §17 待確認）；缺價跳過不報錯；`execution_date` 單值且晚於 signal（呼應 DB CHECK）。

### [x] P2-14 實作 VectorBT 回測服務

- **目標**：src/services/backtest_service.py
- **相依**：P2-13、P1-10
- **輸入與輸出**：run_backtest(orders, prices, initial_cash, settings) -> BacktestResult。
- **驗收準則**：
  - VectorBT fees 與 slippage 同步 orders 的成本計算。
  - 回傳 nav、daily exposure、realized volatility、drawdown 與 order ledger。
  - 測試：tests/test_backtest_service.py 對固定訂單比對成本與 NAV。
- **狀態**：已完成（2026-09-22），`tests/test_backtest_service.py` 5 passed，全量 `pytest -q` 169 passed。
- **更動重點**：新增 `src/services/backtest_service.py:run_backtest`（pandas 現金＋持股帳本、收盤估值、`total_cost`＝實扣加總、回傳 `BacktestResult`）、`tests/test_backtest_service.py`（手算 NAV、現金不足跳單、空單恆現金、出清、非法輸入）。
- **技術注意事項**：與路線圖偏差——本層純 pandas 帳本為稽核真相（成本已在 orders 內，VectorBT 再設 fees 即雙計；`from_orders＋fees=0/slippage=0` 應得同 NAV，列 P5 對帳）；BUY 股數＝`target_shares−現持`、SELL 全出清或減碼至目標；現金不足整筆跳過；`execution<=signal` 拒收；除權息假設價格已調整（SDD §17）。

### [x] P2-15 實作績效與敏感度服務

- **目標**：src/services/metrics_service.py
- **相依**：P2-14、P2-10
- **輸入與輸出**：calculate_metrics(result) -> DataFrame；run_cost_sensitivity(inputs, rates) -> DataFrame。
- **驗收準則**：
  - 輸出 CAGR、Sharpe、Sortino、Calmar、MDD、Win Rate、Turnover、Rank IC、ICIR。
  - 滑價情境固定包含 0.05%、0.10%、0.20%、0.30%。
  - 成本前後結果可同表比較。
  - 測試：tests/test_metrics_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_metrics_service.py` 4 passed，全量 `pytest -q` 173 passed。
- **更動重點**：新增 `src/services/metrics_service.py`（`calculate_metrics` 九指標單列、`run_cost_sensitivity` 四滑價重放同表含 total_cost）、`tests/test_metrics_service.py`（等比 CAGR／MDD／勝率、回撤 Calmar、成本單調、非法輸入）。
- **技術注意事項**：年化 252 天；Rank IC／ICIR 由呼叫方傳入（跨月彙總屬 P2-16）、缺省 NaN；Turnover＝名目本金／平均 NAV；敏感度重放需 prices 含 open（P2-14 只需 close）；SELL 股數以持股追蹤還原；等比數列 Sharpe 爆表（浮點非零變異），測試以 >100 斷言。

### [x] P2-16 建立研究流程應用服務

- **目標**：src/services/research_service.py
- **相依**：P2-02 至 P2-15、P1-11
- **輸入與輸出**：run_research(as_of, settings) -> BacktestResult；get_dashboard_snapshot(run_id) -> dict。
- **驗收準則**：
  - 執行時建立 PipelineRun，成功或失敗時正確完成狀態。
  - 結果可追溯 run id、feature version、model version、parameter version。
  - 對固定 fixture 重跑的結果一致。
  - 測試：tests/test_research_service.py 的端對端 smoke test。
- **狀態**：已完成（2026-09-22），`tests/test_research_service.py` 4 passed，全量 `pytest -q` 179 passed。
- **更動重點**：新增 `src/services/research_service.py`（`ResearchStore` Protocol、`run_research` 串 P2-02〜P2-15、`get_dashboard_snapshot`、`_parameter_version` 參數雜湊、`_stratified_split`）、`tests/test_research_service.py`（FakeStore smoke、重跑一致、失敗關帳、snapshot keys）。
- **技術注意事項**：簽名以 `store` 取代 DB 直連（Phase 5 實 DB 版 store；repository 委派藏於 store 內）；單月 smoke 用 70/30 分層切分調參（多月串接留 P2-10／P5 排程）；空宇宙／無標籤／單類切分即 failed 關帳重拋；`parameter_version`＝策略參數 sha1（`params_<12hex>`）；smoke 抓出 P2-09 floor 位置錯與 P2-12 注水不收斂兩真 bug（見該項補記）。

## 5. Phase 3 外部整合與非同步任務

MVP 不採 Queue 或快取服務。日常作業以可重跑的 CLI job 與 Windows Task Scheduler 執行；此決策避免在兩週範圍引入未被 SDD 要求的基礎設施。

### [x] P3-01 實作 FinMind 價格 Client

- **目標**：src/integrations/finmind_prices.py
- **相依**：P0-04、P1-07
- **輸入與輸出**：fetch_prices(start, end, token) -> DataFrame。
- **驗收準則**：
  - 回傳符合 Price Repository 的標準欄位。
  - 缺少 token 或非成功回應提供不含敏感值的例外。
  - 測試：tests/test_finmind_prices.py 以 HTTP mock 驗證轉換與錯誤。
- **狀態**：已完成（2026-09-22），`tests/test_finmind_prices.py` 5 passed，全量 `pytest -q` 184 passed。
- **更動重點**：新增 `src/integrations/__init__.py`、`src/integrations/finmind.py`（`FinMindError`、`_get` 共用底層）、`src/integrations/finmind_prices.py:fetch_prices`（`TaiwanStockPrice`→標準九欄）、`tests/test_finmind_prices.py`（映射、無 token 錯誤遮罩、壞列丟棄、日期驗證）。
- **技術注意事項**：token 只放 Authorization header，例外字串逐一斷言不含 token；`requester` 注入免新依賴、零真實網路；壞價列（OHLC≤0、高低反轉、缺數）丟棄（SDD §7.3）；空結果回傳空標準欄。

### [x] P3-02 實作 FinMind 基本面與籌碼 Client

- **目標**：src/integrations/finmind_fundamentals.py
- **相依**：P0-04、P1-08
- **輸入與輸出**：fetch_financials(start, end, token) -> DataFrame；fetch_institutional(start, end, token) -> DataFrame。
- **驗收準則**：
  - 財報輸出包含 report period、announcement date、available date。
  - 籌碼輸出含外資、投信、融資、融券與流通股欄位。
  - 測試：tests/test_finmind_fundamentals.py。
- **狀態**：已完成（2026-09-22），`tests/test_finmind_fundamentals.py` 6 passed，全量 `pytest -q` 190 passed。
- **更動重點**：新增 `src/integrations/finmind_fundamentals.py`（`fetch_financials` 財報 11 欄、`fetch_institutional` 三源外連接 8 欄）、`tests/test_finmind_fundamentals.py`（映射＋available 計算、季別正規化、壞列丟棄、三源合併、缺源容忍、遮罩）。
- **技術注意事項**：`available_date`＝公告＋1 日曆日（全域 PIT 單一定義）；季別 `2019/Q4→2019Q4`；金額零／負保留（真實訊號）；浮動股數欄三候選名擇一；任一籌碼源空→該欄 NaN，三源全空→空標準欄。

### [x] P3-03 實作 yfinance 備援 Client

- **目標**：src/integrations/yfinance_prices.py
- **相依**：P1-07
- **輸入與輸出**：fetch_fallback_prices(symbols, start, end) -> DataFrame。
- **驗收準則**：
  - 僅回傳標準化價格欄位與 source 等於 yfinance。
  - 不因單一 symbol 失敗而丟失其他 symbol 結果。
  - 測試：tests/test_yfinance_prices.py 以 mock 驗證部分失敗。
- **狀態**：已完成（2026-09-22），`tests/test_yfinance_prices.py` 5 passed，全量 `pytest -q` 195 passed。
- **更動重點**：新增 `src/integrations/yfinance_prices.py`（`to_yahoo_symbol` TWSE/TPEX 後綴、`fetch_fallback_prices` 逐股隔離）、`tests/test_yfinance_prices.py`（映射、部分失敗、壞價、日期）。
- **技術注意事項**：`traded_value＝close×volume`（yahoo 無成交金額）；失敗名單掛 `frame.attrs["failed"]` 不污染欄；`downloader` 注入零真實網路；曾因 `Series.all(axis=1)` 非法致全股靜默失敗——隔離 except 會吞程式錯，單股轉換錯誤應先讓單測覆蓋正常路。

### [x] P3-04 建立可重試資料同步服務

- **目標**：src/services/sync_service.py
- **相依**：P3-01、P3-02、P3-03、P1-07、P1-08
- **輸入與輸出**：sync_market_data(start, end, settings) -> SyncSummary。
- **驗收準則**：
  - 短暫錯誤依設定重試並記錄嘗試次數。
  - FinMind 價格同步失敗時，只允許價格資料改用 yfinance。
  - 同步成功後可安全重跑，無重複資料。
  - 測試：tests/test_sync_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_sync_service.py` 5 passed，全量 `pytest -q` 200 passed。
- **更動重點**：新增 `src/services/sync_service.py`（`FeedResult`／`SyncSummary.ok`／`SyncStore` Protocol／`sync_market_data` 三源重試＋價格備援）、`tests/test_sync_service.py`（全綠冪等、重試成功、備援切換、財報失敗無備援、ValueError 不重試、缺 token）。
- **技術注意事項**：簽名多 `store/run_id/token/fetch 注入/symbols/max_attempts`（DB 無關可測；token 由呼叫方經 `get_finmind_token` 注入）；僅 `FinMindError`／傳輸例外重試，`ValueError` 直落；缺 token 零嘗試記錯但價格仍可備援；財報籌碼永不降級；`attempts` 含成功那次。

### [x] P3-05 建立日常更新 CLI Job

- **目標**：src/jobs/daily_update.py
- **相依**：P3-04、P2-16
- **輸入與輸出**：python -m src.jobs.daily_update --as-of YYYY-MM-DD；輸出為 SyncSummary、資料品質結果與完成的 PipelineRun。
- **驗收準則**：
  - CLI 可接受指定日期或預設最新交易日。
  - 失敗時 exit code 非零，run 狀態為 failed。
  - 成功時輸出 run id 且不輸出 secret。
  - 測試：tests/test_daily_update_job.py。
- **狀態**：已完成（2026-09-22），`tests/test_daily_update_job.py` 3 passed，全量 `pytest -q` 203 passed。
- **更動重點**：新增 `src/jobs/__init__.py`、`src/jobs/daily_update.py`（`run_daily_update`＋`main` 回 0/1/2）、`tests/test_daily_update_job.py`（成功摘要、失敗關帳、exit 碼、無 secret）。
- **技術注意事項**：`main` 多 `store_factory` 注入（DB 版 Phase 5 前 `_build_store` 拋 NotImplementedError）；同步窗為全量 price_start_date→as_of（upsert 冪等；增量游標留 Phase 5）；部分同步即整 job 失敗（SDD §16）；`--as-of` 缺省取 store 最新日、無則今日；stdout 僅 run_id／日期／筆數。

### [x] P3-06 建立月度訊號 CLI Job

- **目標**：src/jobs/monthly_rebalance.py
- **相依**：P2-16、P2-13、P3-05
- **輸入與輸出**：python -m src.jobs.monthly_rebalance --signal-date YYYY-MM-DD；輸出為保存的 predictions、signals、positions 與 orders。
- **驗收準則**：
  - 僅接受有效月末交易日作為 signal date。
  - 所有 order execution date 都是下一交易日。
  - 失敗時不可留下半完成的訂單集合。
  - 測試：tests/test_monthly_rebalance_job.py。
- **狀態**：已完成（2026-09-22），`tests/test_monthly_rebalance_job.py` 6 passed，全量 `pytest -q` 209 passed。
- **更動重點**：新增 `src/jobs/monthly_rebalance.py`（`run_monthly_rebalance`＋`main` 0/1/2）、`tests/test_monthly_rebalance_job.py`（成功、非月末拒收、執行日複驗、取代失敗無殘留、exit 碼）。
- **技術注意事項**：`main` 多 `store_factory`＋`research_fn` 雙注入；月末由 `store.verify_month_end` 判定（DB 日曆，非 CLI 自算）；`replace_orders` 單交易範圍取代（P1-10 語義），研究內 `save_orders` 在此 job 的 DB store 中須為暫存（Phase 5 接線備註）；execution 單一且晚於 signal 複驗防迴歸。

### [x] P3-07 建立本機排程設定文件

- **目標**：docs/windows-task-scheduler.md
- **相依**：P3-05、P3-06
- **輸入與輸出**：輸入為 Python 虛擬環境與 job CLI；輸出為每日 09:00 與月末收盤後的 Windows Task Scheduler 設定程序。
- **驗收準則**：
  - 文件使用環境變數，不寫入任何 token。
  - 包含工作目錄、log 檔位置、失敗時檢查步驟與手動重跑指令。
  - 驗證：依文件在測試環境建立一次排程並執行 dry run。
- **狀態**：已完成（2026-09-22），`tests/test_scheduler_docs.py` 4 passed，全量 `pytest -q` 213 passed。
- **更動重點**：新增 `docs/windows-task-scheduler.md`（前置檢查、每日 09:00、月末 LASTDAY 觸發＋手填 signal-date、exit 碼對照、重跑指令）、`tests/test_scheduler_docs.py`（程序關鍵字、無 secret 掃描、subprocess 雙 `--help` 回 0）；另修兩 job 入口 sys.path bootstrap。
- **技術注意事項**：路徑全 `<REPO>`／`<VENV>` 佔位符；月末交易日 Scheduler 無原生觸發器，以 LASTDAY＋手填＋`verify_month_end` 把關；驗證抓到 `python -m src.jobs.*` 在 repo 根跑不起來（內部絕對 import）→ 兩 job 頂部加 `sys.path.insert(src)` bootstrap，subprocess 迴歸測試鎖定。

### [x] P3-08 建立 Telegram 通知 Adapter

- **目標**：src/integrations/telegram.py
- **相依**：P0-05、P2-16
- **輸入與輸出**：send_research_alert(summary: dict, token: str, chat_id: str) -> None。
- **驗收準則**：
  - 僅在設定啟用時呼叫。
  - 訊息可包含 Top 15、MDD、缺漏資料與 pipeline error 摘要。
  - 外部傳送以 mock 測試，不在 unit test 傳送真實訊息。
  - 優先級：P1，可在 MVP 後執行。
- **狀態**：已完成（2026-09-22），`tests/test_telegram.py` 5 passed，全量 `pytest -q` 218 passed。
- **更動重點**：新增 `src/integrations/telegram.py`（`format_research_alert` 純文字＋跳脫、`send_research_alert` 回 sent/disabled）、`tests/test_telegram.py`（內容行、跳脫、停用零呼叫、成功 post、錯誤遮罩）。
- **技術注意事項**：簽名多回傳字串（便於 Phase 5 排程記 log）；token 只在 URL path，例外僅狀態碼；純文字不用 Markdown parse_mode（代號特殊字元）；4000 字截斷；P1 優先但已隨 Phase 3 完成（呼叫方 Phase 5 接線）。

### [x] P3-09 建立 Gemini 摘要 Adapter

- **目標**：src/integrations/gemini_report.py
- **相依**：P0-05、P2-16
- **輸入與輸出**：generate_summary(context: dict, api_key: str) -> str。
- **驗收準則**：
  - 輸入限定為 Top 15、SHAP、市場 regime、投組風險與核可的公告內容。
  - 輸出不改變 signal、weight、order 或模型資料。
  - 外部呼叫錯誤不影響研究 run 成功。
  - 優先級：P1，可在 MVP 後執行。
- **狀態**：已完成（2026-09-22），`tests/test_gemini_report.py` 5 passed，全量 `pytest -q` 223 passed。
- **更動重點**：新增 `src/integrations/gemini_report.py`（`ALLOWED_CONTEXT_KEYS` 白名單、`build_prompt` 約束句、`generate_summary` REST 直打）、`tests/test_gemini_report.py`（白名單、成功串接、狀態不變、遮罩、壞體）。
- **技術注意事項**：零新依賴（`requests` 直打 generateContent，不裝 SDK）；key 走 `x-goog-api-key` header 不放 URL；`announcements` 須為核可字串 list；外部錯一律 `GeminiError` 由呼叫方吸收保 run；P1 優先但已隨 Phase 3 完成（呼叫方 Phase 5 接線）。

## 6. Phase 4 前端與用戶端介面

### [x] P4-01 建立 Streamlit 應用殼層

- **目標**：app.py
- **相依**：P2-16、P0-04
- **輸入與輸出**：main() -> None；輸入為選定 run id 與日期篩選，輸出為頁面導覽、設定載入與錯誤狀態。
- **驗收準則**：
  - 可啟動 streamlit run app.py。
  - 選單只顯示 succeeded run。
  - 無資料時顯示可操作的空狀態，而非 stack trace。
  - 測試：tests/test_app_bootstrap.py。
- **狀態**：已完成（2026-09-22），`tests/test_app_bootstrap.py` 8 passed，全量 `pytest -q` 231 passed。
- **更動重點**：新增 `app.py`（`PAGES` 五頁、`RunStore` 極小接縫、`available_runs` 只留 succeeded、`main(st/store/load_settings_fn/loaders/pages)` 全注入）、`tests/test_app_bootstrap.py`（fake st＋fake store，不啟瀏覽器）。
- **技術注意事項**：streamlit 延遲 import（測試零依賴）；預設 loaders/pages 走 lazy import，P4-02〜P4-07 落地後自動接通（此前顯示就緒提示不 traceback）；`_default_loaders` 已預留 `get_model_data`／`get_comparison` 兩個 P4-02 介面（P4-02 實作時必須補上，否則模型／比較頁 lazy import 成功但屬性缺失→錯誤狀態）；`python app.py` bare-mode 可跑完（store=None→info），`streamlit run` 所用 API（selectbox/radio/plotly_chart/dataframe）已對 1.64.0 確認存在。

### [x] P4-02 建立 Dashboard 讀取服務

- **目標**：src/services/dashboard_service.py
- **相依**：P2-16、P1-09、P1-10
- **輸入與輸出**：get_overview(run_id) -> dict；get_holdings(run_id, as_of) -> DataFrame；get_risk(run_id) -> dict。
- **驗收準則**：
  - 所有讀取服務僅讀取已完成 run。
  - 回傳資料具有明確欄位，不直接暴露 ORM 實體。
  - 測試：tests/test_dashboard_service.py。
- **狀態**：已完成（2026-09-22），`tests/test_dashboard_service.py` 6 passed，全量 `pytest -q` 237 passed。
- **更動重點**：新增 `src/services/dashboard_service.py`（`DashboardStore` 六方法接縫、五讀取函式：`get_overview`／`get_holdings`／`get_model_data`／`get_risk`／`get_comparison`）、`tests/test_dashboard_service.py`。
- **技術注意事項**：函式簽名一律多 `store` 參數（DB-free 慣例）；`_require_completed` 統一擋非 succeeded（未知 run→ValueError）；dict 僅白名單鍵＋NaN→None；`get_holdings` 會回收 `stock_id` index（驗收時 fake 抓到 index 化 frame 案例）；P4-01 預留的 `get_model_data`／`get_comparison` 已兌現，殼層免改。

### [x] P4-03 實作總覽頁

- **目標**：src/ui/overview_page.py
- **相依**：P4-01、P4-02
- **輸入與輸出**：render_overview(snapshot: dict) -> None。
- **驗收準則**：
  - 顯示 CAGR、Sharpe、Sortino、MDD、Turnover、IC、Equity Curve、Drawdown 與月報酬熱圖。
  - 指標缺值時顯示 N A，不產生圖表例外。
  - 測試：tests/test_overview_page.py。
- **狀態**：已完成（2026-09-22），`tests/test_overview_page.py` 4 passed，全量 `pytest -q` 241 passed。
- **更動重點**：新增 `src/ui/__init__.py`、`src/ui/overview_page.py`（`render_overview`＋`equity_curve_figure`／`drawdown_figure`／`monthly_heatmap_figure` 純 builder）、`tests/test_overview_page.py`。
- **技術注意事項**：drawdown 由 NAV 自算（忽略輸入口徑）；壞點逐列跳過、全壞→空圖＋info；`monthly_heatmap_figure` 壞輸入回無 trace 圖（render 以 `figure.data` 非空才 chart）；後續 ui 頁沿用同一 fake-st＋純 builder 模式。

### [x] P4-04 實作投組頁

- **目標**：src/ui/portfolio_page.py
- **相依**：P4-01、P4-02
- **輸入與輸出**：render_portfolio(holdings: DataFrame) -> None。
- **驗收準則**：
  - 顯示 stock id、排名、預測機率、權重、波動率與 Beta。
  - 權重總和和股票數量與選定 run 一致。
  - 測試：tests/test_portfolio_page.py。
- **狀態**：已完成（2026-09-22），`tests/test_portfolio_page.py` 4 passed，全量 `pytest -q` 245 passed。
- **更動重點**：新增 `src/ui/portfolio_page.py`（`render_portfolio`＋`summarize_holdings` 純彙總）、`tests/test_portfolio_page.py`。
- **技術注意事項**：依 rank 升冪顯示；權重非有限即 `ValueError`（壞權重不出表）；空表為合法「無持股」info 狀態。

### [x] P4-05 實作模型頁

- **目標**：src/ui/model_page.py
- **相依**：P4-01、P4-02、P2-08
- **輸入與輸出**：render_model(model_data: dict) -> None。
- **驗收準則**：
  - 顯示 SHAP Top 10、Feature Importance、Monthly IC 與預測分布。
  - 清楚標註 SHAP 是特徵重要性，不是因子收益歸因。
  - 測試：tests/test_model_page.py。
- **狀態**：已完成（2026-09-22），`tests/test_model_page.py` 3 passed，全量 `pytest -q` 248 passed。
- **更動重點**：新增 `src/ui/model_page.py`（`SHAP_DISCLAIMER` 置頂、`shap_figure` 限 Top10／`monthly_ic_figure`／`prediction_dist_figure` 純 builder）、`tests/test_model_page.py`。
- **技術注意事項**：四段獨立降級（有 SHAP 無 SHAP 都寫免責）；`pandas` 頂層 import（中途曾寫進函式內，已移出）。

### [x] P4-06 實作風險頁

- **目標**：src/ui/risk_page.py
- **相依**：P4-01、P4-02
- **輸入與輸出**：render_risk(risk_data: dict) -> None。
- **驗收準則**：
  - 顯示股票曝險、預測與實際波動、MDD、Turnover、TAIEX regime。
  - MA60 下的曝險限制可由資料驗證。
  - 測試：tests/test_risk_page.py。
- **狀態**：已完成（2026-09-22），`tests/test_risk_page.py` 2 passed，全量 `pytest -q` 250 passed。
- **更動重點**：新增 `src/ui/risk_page.py`（`check_ma60_exposure` 純檢核＋`render_risk`）、`tests/test_risk_page.py`。
- **技術注意事項**：檢核四分支（符合／超限／不適用／資料不足）；`below_ma60` 比較採 `<=`（等於上限為合法）；regime 缺失→整段資料不足而非預設順勢。

### [x] P4-07 實作研究比較頁

- **目標**：src/ui/comparison_page.py
- **相依**：P4-01、P4-02、P2-15
- **輸入與輸出**：render_comparison(metrics: DataFrame) -> None。
- **驗收準則**：
  - 可比較基準模型、因子消融、Top N 與成本敏感度。
  - 成本前與成本後指標同時可見。
  - 測試：tests/test_comparison_page.py。
- **狀態**：已完成（2026-09-22），`tests/test_comparison_page.py` 3 passed，全量 `pytest -q` 253 passed。
- **更動重點**：新增 `src/ui/comparison_page.py`（`paired_metrics` 純配對＋`render_comparison` 全表＋逐情境 Δ 行）、`tests/test_comparison_page.py`。
- **技術注意事項**：配對以後綴掃描（單邊欄不顯示 Δ）；Δ 由頁面自算（after−before）；非數值→該格 N/A 不中斷整頁；至此 app.py 預設 lazy import 的五頁＋五 loader 全數存在，殼層已實質接通（缺的只剩 Phase 5 DB store）。

### [x] P4-08 實作研究報表產生器

- **目標**：src/report.py
- **相依**：P2-15、P4-02
- **輸入與輸出**：export_reports(run_id, output_dir) -> ReportPaths。
- **驗收準則**：
  - 產生 performance.csv、factor_ic.csv、run_summary.json 與 report.pdf。
  - JSON 包含 run id、資料截止日、版本、參數與 OOS 結果。
  - 測試：tests/test_report.py 驗證檔案存在與必要欄位。
- **狀態**：已完成（2026-09-22），`tests/test_report.py` 2 passed，全量 `pytest -q` 255 passed。
- **更動重點**：新增 `src/report.py`（`ReportPaths` frozen dataclass、`ReportStore` 四方法接縫、`export_reports`）、`tests/test_report.py`。
- **技術注意事項**：PDF 為手寫最小 PDF 1.4（Helvetica 單頁，xref 偏移已程式驗證），零新依賴；中文指標名寫入 PDF 會被 latin-1 replace（英文鍵不受影響，屬已知限制）；只匯出 succeeded run。

## 7. Phase 5 監控 日誌與安全性強化

### [x] P5-01 建立結構化日誌

- **目標**：src/observability/logging.py
- **相依**：P0-04
- **輸入與輸出**：configure_logging(run_id: str | None) -> Logger。
- **驗收準則**：
  - 每筆 log 至少含 timestamp、level、event、run id 與 module。
  - Token、API key、完整外部回應不寫入 log。
  - 測試：tests/test_logging.py。
- **狀態**：已完成（2026-09-22），`tests/test_logging.py` 6 passed，全量 `pytest -q` 261 passed。
- **更動重點**：新增 `src/observability/__init__.py`、`src/observability/logging.py`（`configure_logging`＋公開 `RunContextFilter`／`JsonFormatter`）、`tests/test_logging.py`（fixture 掛 JSON handler 擷取）。
- **技術注意事項**：單行 JSON、UTC ISO timestamp；`event` 缺省為 `unspecified`；遮罩三層（mapping args 敏感鍵、頂層 extra 敏感鍵、巢狀 dict；`%` 內插被引用的 secret 顯示 `***`，未被引用的直接丟棄）；超 2000 字元欄位截斷＋`…[truncated]`；重複 configure 冪等（marker 認 handler 並更新 run_id）；`propagate=False`；message 內已內插的 secret 無法回溯遮罩——呼叫方不得把 secret 寫進訊息字串（僅 extra 傳）。

### [x] P5-02 將 Run Lifecycle 接入所有工作

- **目標**：src/observability/run_lifecycle.py
- **相依**：P1-11、P5-01、P2-16
- **輸入與輸出**：managed_run(metadata) -> ContextManager[PipelineRun]。
- **驗收準則**：
  - daily update、monthly rebalance、research run 都會建立開始與結束紀錄。
  - 例外時狀態為 failed 且保存錯誤摘要。
  - 測試：tests/test_run_lifecycle.py。
- **狀態**：已完成（2026-09-22），`tests/test_run_lifecycle.py` 6 passed，全量 `pytest -q` 267 passed。
- **更動重點**：新增 `src/observability/run_lifecycle.py`（`managed_run`＋`summarize_error`）、`tests/test_run_lifecycle.py`（真 SQLite＋migration，三種 run_id 前綴各一案例）。
- **技術注意事項**：只包 ORM 版 P1-11（P3 jobs 的 store 版 `start/finish` 不動，依 AGENT.md）；metadata 須合 P1-11 白名單（`job` 鍵不合法——以 `daily-`／`monthly-`／`research-` run_id 前綴區分工作種類）；空錯誤訊息補 `unknown error` 防 finish 二度炸；terminal run 重開是 PK IntegrityError（預期），不可二度關閉由 `finish_run` 的 already 守衛保證；原例外一律重拋。

### [x] P5-03 建立資料品質稽核報告

- **目標**：src/observability/data_quality.py
- **相依**：P1-07、P1-08、P5-02
- **輸入與輸出**：audit_data_quality(as_of, session) -> DataQualityReport。
- **驗收準則**：
  - 報告缺漏價格、重複主鍵、無效價格、財務 PIT 違規與因子覆蓋率。
  - 重大違規會阻止訊號 job 繼續。
  - 測試：tests/test_data_quality.py。
- **狀態**：已完成（2026-09-22），`tests/test_data_quality.py` 3 passed，全量 `pytest -q` 270 passed。
- **更動重點**：新增 `src/observability/data_quality.py`（`DataQualityIssue`／`DataQualityReport` frozen dataclass、`audit_data_quality` 五探測＋`assert_no_blockers`）、`tests/test_data_quality.py`（真 SQLite＋migration，髒／淨雙 fixture）。
- **技術注意事項**：唯讀（僅 select）；blocker＝無效 OHLC／零 bar 股／無 features，warning＝零星缺 bar／重複鍵／畸形 report_period／覆蓋率<0.9；`duplicate_keys` 與 `financial_pit_violation` 正常恆為零（PK＋CHECK 由 DDL 保證，屬縱深防禦）；重大發現：同 session 混插父（stocks）子（prices）表會 FK 失敗（UoW 先刷子表，加順序無關），全 repo 既有測試皆分開播種，P5-03 沿用此模式（先 commit 父表），既有 ORM 不動；覆蓋率以最新 rebalance_date 期為準；`assert_no_blockers` 供訊號 job 入口呼叫（接線留待 job 改寫）。

### [x] P5-04 建立研究健康狀態查詢

- **目標**：src/observability/health.py
- **相依**：P1-11、P5-03
- **輸入與輸出**：get_system_health(session) -> dict。
- **驗收準則**：
  - 回傳最近成功 run、資料截止日、最近失敗、因子覆蓋率與資料延遲。
  - 不含 token 或敏感設定。
  - 測試：tests/test_health.py。
- **狀態**：已完成（2026-09-22），`tests/test_health.py` 3 passed，全量 `pytest -q` 273 passed。
- **更動重點**：新增 `src/observability/health.py`（`get_system_health(session, today=None)`）、`tests/test_health.py`（有資料快照／空庫全 None／壞 today）。
- **技術注意事項**：唯讀；排序鍵 run_time＋run_id 雙保險（同秒多 run 可定序）；`data_end_date` 取自最近成功 run（失敗 run 無可信截止日）；lag 為日曆天差（<=0 鉗 0）；覆蓋率口徑同 P5-03；`today` 可注入測穩，預設 UTC 今日。

### [x] P5-05 建立 Secret 與設定安全檢查

- **目標**：src/security/secrets.py
- **相依**：P0-05、P0-04
- **輸入與輸出**：validate_runtime_secrets(settings, env) -> None。
- **驗收準則**：
  - 啟用 FinMind、Telegram 或 Gemini 時缺少對應 secret 必須失敗。
  - 不啟用的 P1 integration 不要求 secret。
  - 測試：tests/test_secrets.py。
- **狀態**：已完成（2026-09-22），`tests/test_secrets.py` 6 passed，全量 `pytest -q` 279 passed。
- **更動重點**：新增 `src/security/__init__.py`、`src/security/secrets.py`（`SecretError`＋`is_enabled`／`secret_env_name`／`validate_runtime_secrets`）、`tests/test_secrets.py`（全注入 env，真值不落地）。
- **技術注意事項**：Settings 無 enabled 欄（P0-04 凍結，依 AGENT.md 不加欄），啟用旗標走 env（`FINMIND_ENABLED`／`TELEGRAM_ENABLED`／`GEMINI_ENABLED`，`1/true/yes` 大小寫寬容）；FinMind 變數名以 `settings.data.finmind_token_env` 為準（P0-05 約定），另兩名固定；例外只含變數名；`env=None` 讀 `os.environ`（jobs 呼叫形態）。

### [x] P5-06 建立依賴與敏感檔案 CI 檢查

- **目標**：.github/workflows/security.yml
- **相依**：P0-08、P5-05
- **輸入與輸出**：輸入為 pull request；輸出為 dependency audit 與 secret scan 結果。
- **驗收準則**：
  - 對 requirements 執行套件弱點檢查。
  - 拒絕提交 .env、SQLite 實體資料庫和疑似 token。
  - 驗證：使用測試 fixture 觸發拒絕規則。
- **狀態**：已完成（2026-09-22），`tests/test_security_workflow.py` 4 passed，全量 `pytest -q` 283 passed。
- **更動重點**：新增 `.github/workflows/security.yml`（push＋PR，`pip-audit -r requirements.txt`＋secret 掃描＋敏感檔守衛）、`tests/test_security_workflow.py`（結構斷言＋規則鏡像 fixture）。
- **技術注意事項**：pip-audit 當步安裝不進 pin 表；secret pattern 掃「疑似值」形態（AWS／ghp／xox／私鑰／sk-live），`.example` 豁免；工作區非 git repo 故以本機等價驗證（P0-08 前例）：規則鏡像在 Python `re` 重放＋工作流原掃描邏輯實跑全樹零命中；驗證抓到測試自身 fixture 會被掃中→fixture 改拼接＋掃描排除 `__pycache__/.pytest_cache/.ruff_cache` 生成物（另發現 CPython 會常量折疊拼接字串，故排除目錄才是正解）；file-guard 以 `git diff` 為主、`git ls-files` 兜底。

### [x] P5-07 建立營運 Runbook

- **目標**：docs/runbook.md
- **相依**：P3-05、P3-06、P5-03、P5-04
- **輸入與輸出**：輸入為健康狀態與 job 失敗情境；輸出為資料失敗、PIT 違規、模型失敗、協方差不足與報表失敗的排查步驟。
- **驗收準則**：
  - 每個重大失敗情境都有觀察、判斷、重跑或停止的步驟。
  - 指令不得包含 secret。
  - 驗證：以一個 fixture 失敗案例依 Runbook 完成診斷。
- **狀態**：已完成（2026-09-22），`tests/test_runbook.py` 3 passed，全量 `pytest -q` 286 passed。
- **更動重點**：新增 `docs/runbook.md`（09:00 例行＋五情境＋重跑指令附錄）、`tests/test_runbook.py`（情境覆蓋／無 secret／髒 fixture 診斷演練）。
- **技術注意事項**：每情境固定「觀察→判斷→重跑或停止」三段；停止條件明確（半套資料停訊號、參數改壞停重跑、store 漂移停發布）；重跑指令與 P3-07 調度文件一致（`--as-of`／`--signal-date`）；無 secret 斷言含賦值形態掃描。

### [x] P5-08 建立 Release 驗收清單

- **目標**：docs/release-checklist.md
- **相依**：P4-08、P5-04、P5-06、P5-07
- **輸入與輸出**：輸入為候選版本；輸出為可勾選的資料、研究、回測、Dashboard、安全與文件驗收紀錄。
- **驗收準則**：
  - 對應 SDD 的 Point-in-Time、Survivorship Bias、Next-day Execution、成本、OOS 與可重現性要求。
  - 包含最終 OOS 成本前後報告與敏感度分析確認。
  - 驗證：以 MVP fixture 完成一次完整 checklist。
- **狀態**：已完成（2026-09-22），`tests/test_release_checklist.py` 2 passed，全量 `pytest -q` 288 passed。
- **更動重點**：新增 `docs/release-checklist.md`（六節可勾項＋頁首簽核行）、`tests/test_release_checklist.py`（SDD 門檻覆蓋斷言＋MVP fixture 走查：audit→health→export 全串）。
- **技術注意事項**：走查即 checklist 證據鏈的程式等價（稽核 passed／health 對上 run／四檔報表齊）；Survivorship 在清單列為稽核項（下市保留史由 universe 層保證，P2-02）；簽核行含版本／截止日／簽核人三欄。

## 8. 依賴索引

| 任務 | 直接依賴 |
| --- | --- |
| P1-01 | P0-02、P0-04 |
| P1-02 | P1-01、P0-07 |
| P1-06 至 P1-11 | 對應 ORM 任務與 P1-02 |
| P2-02 | P1-06、P1-07、P2-01 |
| P2-03 至 P2-16 | 依各任務中列出的上游資料、模型或交易服務 |
| P3-05 | P3-04、P2-16 |
| P3-06 | P3-05、P2-13、P2-16 |
| P4-01 至 P4-08 | P2-16 與 Dashboard 讀取服務 |
| P5-01 至 P5-08 | Run Repository、核心 jobs 與報表完成 |

## 9. 第一個可展示的垂直切片

完成下列任務後，可在固定 historical fixture 上展示第一個端到端研究結果：

~~~text
P0-01, P0-02, P0-03, P0-04, P0-07
P1-01, P1-02, P1-03, P1-04, P1-05, P1-06, P1-07, P1-08, P1-09, P1-10, P1-11
P2-01, P2-02, P2-03, P2-04, P2-05, P2-06, P2-08, P2-09, P2-10, P2-11, P2-12, P2-13, P2-14, P2-15, P2-16
P4-01, P4-02, P4-03
~~~

此切片刻意不依賴真實 API、排程、Telegram 或 Gemini，先驗證資料可得性、模型、風控、成交、成本與 OOS 報表的完整鏈路。
