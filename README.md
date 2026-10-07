# AI Quant Research

An end-to-end research platform for multi-factor equity selection in the Taiwan stock market. The repository covers data ingestion, point-in-time feature engineering, monthly panel construction, model training, out-of-sample portfolio simulation, research diagnostics, and a Streamlit dashboard.

The current research focus is cross-sectional stock ranking: buy the highest-ranked 15 stocks at each monthly rebalance and retain existing holdings until they fall below rank 30. The system compares classification, learning-to-rank, robust regression, and linear baselines under the same portfolio and transaction-cost assumptions.

> This is a research project, not investment advice. Historical and simulated results do not guarantee future performance.

## Research highlights

- **Point-in-time data discipline.** Historical decisions use information available on each signal date. Factor v4 requires same-day market value, announced financial data, sufficient adjusted-price history, and a minimum listing age.
- **Corporate-action-aware returns.** Technical factors, labels, and forward 20-trading-day returns use adjusted prices. Nominal prices remain in use where the economic definition requires them, such as market capitalization and trade execution.
- **Leakage-aware validation.** Annual experiments use four training years, one validation year, a 20-trading-day purge, and a separate OOS year.
- **Several modeling objectives.** The training interface supports XGBoost binary classification, `rank:pairwise`, pseudo-Huber regression, and a fixed Logistic Regression baseline.
- **Portfolio-level evaluation.** Runs include transaction fees, securities transaction tax, slippage, volatility targeting, individual-weight limits, and a TAIEX MA60 exposure rule.
- **Selection diagnostics.** In addition to portfolio P&L, the dashboard reports continuous Rank IC, Top-15 actual excess return, Top-Bottom spread, SHAP values, and XGBoost gain.
- **Reproducible research artifacts.** Monthly panels and trained model artifacts are versioned so existing experiments can be inspected without rebuilding every intermediate dataset.

## Current OOS evidence

The table below summarizes five annual factor v4, no-refit OOS runs from 2020 through 2024. Values are arithmetic averages of the five annual run metrics; they are not a single linked five-year portfolio return.

| Method | Mean annual CAGR | Mean Sharpe | Mean continuous Rank IC | Mean annual turnover |
| --- | ---: | ---: | ---: | ---: |
| XGBoost binary | 7.35% | 0.78 | -0.063 | 5.71x |
| Logistic Regression | 5.05% | 0.48 | -0.064 | 6.21x |
| XGBoost pairwise ranking | 5.08% | 0.49 | -0.069 | 7.69x |
| XGBoost pseudo-Huber | **12.71%** | **0.92** | **+0.034** | 10.09x |

Pseudo-Huber produced the strongest average return and the only positive mean continuous Rank IC in this comparison. Its main weakness was turnover: about 10.1 times per year. Score smoothing and wider holding buffers reduced turnover only modestly and generally lowered average performance. The strategy also lagged the TAIEX in several strong-market years, so the result should be viewed as a promising research direction rather than evidence of stable market-neutral alpha.

The concise interview presentation is available at [`output/pptx/taiwan_quant_research_interview_2026-10-05.pptx`](output/pptx/taiwan_quant_research_interview_2026-10-05.pptx).

## System overview

```text
FinMind / official sources / Shioaji / yfinance
                    |
                    v
         SQLite point-in-time data store
                    |
                    v
        Monthly universe and factor panels
                    |
                    v
   Train -> validate -> freeze model -> OOS score
                    |
                    v
 Portfolio construction, costs, risk controls, metrics
                    |
                    v
             Streamlit dashboard
```

### Data and universe

The default factor v4 configuration applies these core rules:

- Taiwan-listed securities with at least 252 trading days of history
- Minimum market capitalization of TWD 5 billion
- Minimum 20-day average traded value of TWD 20 million
- Minimum nominal share price of TWD 10
- At least 121 adjusted-price observations
- Required point-in-time financial fields: net income and equity
- Missing critical data removes a stock from the eligible universe before label ranking
- Non-critical cross-sectional missing values use monthly median imputation without missing-value indicator features

The label is the stock's adjusted 20-trading-day forward return, ranked within that month's eligible universe. The binary objective marks the top 20% as positive. The pseudo-Huber objective predicts continuous excess return relative to the monthly universe mean.

### Portfolio rules

The baseline portfolio configuration uses:

- Buy rank: Top 15
- Holding buffer: retain until rank falls below 30
- Maximum individual weight: 10%
- Target annual volatility: 15%
- Maximum equity exposure: 100%
- Reduced exposure below the TAIEX 60-day moving average: 50%
- Signal at month-end close and execution at the next trading-day open
- Broker fee, sell tax, and per-side slippage included in simulation

## Dashboard

The Streamlit dashboard contains five research views:

1. **Overview** — strategy and TAIEX CAGR, Sharpe, Sortino, drawdown, NAV, drawdown path, and monthly returns.
2. **Portfolio** — historical holdings by signal month, stock names, weights, ranks, volatility, and beta.
3. **Model** — continuous IC, Top-15 excess return, Top-Bottom spread, SHAP, gain, and prediction distributions.
4. **Risk** — equity exposure, forecast and realized volatility, market regime, drawdown, and turnover.
5. **Research comparison** — transaction-cost and sensitivity comparisons across completed runs.

Runs can be filtered by factor version, OOS year, method, and score-smoothing coefficient.

Start the dashboard on Windows:

```powershell
.\start_dashboard.bat
```

Or start it directly from an activated environment:

```bash
python -m streamlit run app.py --server.port 8501
```

Then open <http://localhost:8501/>.

## Installation

Python 3.12 is the tested environment; project metadata allows Python 3.10 or newer.

```bash
git clone https://github.com/povmaru888/AI_Quant_Research.git
cd AI_Quant_Research
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the environment template and add only the credentials required for the data sources you intend to use:

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

Supported environment keys include `FINMIND_TOKEN`, `SHIOAJI_API_KEY`, and `SHIOAJI_SECRET_KEY`. `.env` and live SQLite databases are intentionally excluded from version control.

## Reproducing the pipeline

### 1. Initialize the database

```bash
python tools/init_db.py --config config.factor_v4.yaml
```

### 2. Seed the stock master and TAIEX

```bash
python tools/seed_stocks.py --config config.factor_v4.yaml --start 2014-06-01
```

### 3. Synchronize historical data

FinMind access varies by account tier. The repository provides full-market and per-symbol workflows, plus repair utilities for adjusted prices, market-value gaps, and exceptional Saturday trading sessions.

Example per-symbol free-tier synchronization:

```bash
python tools/sync_free.py \
  --config config.factor_v4.yaml \
  --start 2015-01-01 \
  --feeds all
```

Example PIT market-value synchronization:

```bash
python tools/sync_market_values.py \
  --config config.factor_v4.yaml \
  --start 2015-01-01 \
  --end 2024-12-31
```

### 4. Build monthly factor panels

```bash
python tools/build_panels.py \
  --config config.factor_v4.yaml \
  --start 2019-01 \
  --end 2023-12 \
  --out data/panels_v4
```

The panel builder uses fingerprints and atomic writes. Valid cached months are reused unless `--force` is supplied.

### 5. Train and validate a model

Pseudo-Huber example for an OOS 2024 study:

```bash
python tools/train_model.py \
  --config config.factor_v4.yaml \
  --panels data/panels_v4 \
  --train-start 2019-01 \
  --train-end 2022-11 \
  --purge-month 2022-12 \
  --valid-start 2023-01 \
  --valid-end 2023-12 \
  --trials 50 \
  --objective reg:pseudohubererror \
  --out models/model_oos2024_v4_pseudohuber
```

Use `--params-from <prior-report.json>` to reuse previously selected XGBoost parameters and skip Optuna. Logistic Regression uses its fixed baseline specification and does not run Optuna.

### 6. Run a frozen-model OOS portfolio simulation

```bash
python tools/run_oos_portfolio.py \
  --config config.factor_v4.yaml \
  --model models/model_oos2024_v4_pseudohuber \
  --model-version xgb_pseudohuber_oos2024 \
  --start 2024-01 \
  --end 2024-12 \
  --panels data/panels_v4 \
  --run-id oos-2024-factor-v4-pseudohuber-NOrefit
```

The final December signal is valued through its full 20-trading-day horizon in the following January.

## Repository layout

```text
app.py                 Streamlit application shell
config*.yaml           Research, universe, portfolio, and execution settings
src/integrations/      FinMind, Shioaji, yfinance, Telegram, and Gemini adapters
src/repositories/      Database access and idempotent persistence
src/runtime/           Runtime data preparation and database-backed store
src/services/          Features, labels, models, portfolio, risk, and metrics
src/ui/                Dashboard pages and shared visual theme
tools/                 Data, panel, training, OOS, audit, and repair commands
tests/                 Unit, integration, PIT, cache, and failure-path tests
data/panels*/          Versioned monthly research panels
models/                Versioned trained models and reports
output/                Shareable research artifacts
```

## Data availability after cloning

The repository includes versioned panel and model artifacts used for existing research. It does **not** include the live `database/quant.db` file because it is large, machine-specific, and may contain locally synchronized data.

Therefore:

- You can inspect the source code, configuration, committed panels, models, and presentation after cloning.
- To run new data synchronization, build fresh panels, materialize runs, or populate the dashboard, initialize and populate a local database first.
- Dashboard run metadata and detailed holdings are read from the local database, so cloning alone does not reproduce the original machine's complete dashboard state.

## Quality controls

The CI workflow runs dependency validation, Ruff linting, format checks, and pytest on every push and pull request.

```bash
python -m pip check
ruff check src tests
ruff format --check src tests
python -m pytest -q
```

The test suite covers point-in-time boundaries, adjusted-price behavior, labels, panel caching and recovery, model persistence, OOS materialization, database key isolation between runs, dashboard rendering, and failure handling.

## Research limitations

- The research history contains only a limited number of annual OOS regimes.
- Hyperparameter selection and repeated experimentation can still create research overfitting even when each individual run is temporally valid.
- Turnover, taxes, fees, and slippage are modeled, but actual market impact and capacity require further analysis.
- Relative performance varies substantially by market regime; strong absolute returns do not necessarily imply benchmark outperformance.
- Data-source availability and revisions can affect reproducibility across accounts and dates.

The next research priorities are turnover-aware optimization, sector and size neutralization, capacity analysis, and a longer rolling OOS record.

## Technology

Python, pandas, NumPy, SQLAlchemy, SQLite, XGBoost, scikit-learn, Optuna, SHAP, vectorbt, Plotly, Streamlit, pytest, and Ruff.
