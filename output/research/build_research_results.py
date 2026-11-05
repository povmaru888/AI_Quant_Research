"""Read-only research snapshot; regenerate the interview Markdown and chart data."""
from pathlib import Path
import sqlite3
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/research/research_results_assets'
OUT.mkdir(exist_ok=True)
C = sqlite3.connect(f'file:{(ROOT / "database/quant.db").as_posix()}?mode=ro', uri=True)
sources = {}
def artifact(run, kind='metrics'):
    raw = C.execute('SELECT payload_json FROM run_artifacts WHERE run_id=? AND kind=?', (run, kind)).fetchone()
    if raw is None:
        raise ValueError((run, kind))
    sources[f'{run}/{kind}'] = hashlib.sha256(raw[0].encode()).hexdigest()
    return json.loads(raw[0])
def rid(y, suffix='pseudohuber'):
    return f'oos-{y}-factor-v4' + (f'-{suffix}' if suffix else '') + '-NOrefit'
def save(df, name):
    df.to_csv(OUT / (name+'.csv'), index=False, encoding='utf-8-sig')
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'] + ['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(x): return f'{x:.2%}'
def num(x): return f'{x:.3f}'
def metrics(r):
    r = np.asarray(r, dtype=float)
    w = np.r_[1., np.cumprod(1+r)]
    return dict(cumulative_return=w[-1]-1,cagr=w[-1]**(252/len(r))-1,sharpe=r.mean()/r.std(ddof=1)*np.sqrt(252),sortino=r.mean()/r[r<0].std(ddof=1)*np.sqrt(252),max_drawdown=(w/np.maximum.accumulate(w)-1).min())

# Include the opening-day strategy change against each run's initial TWD 30m.
# Match benchmark close-to-close returns on every strategy valuation date.
market = pd.read_sql_query("SELECT trade_date,close FROM prices WHERE stock_id='TAIEX' ORDER BY trade_date", C).set_index('trade_date')['close']
market_returns = market.pct_change()
annual=[]; paths=[]; shaps=[]
for y in range(2020,2026):
    p=artifact(rid(y)); curve=pd.DataFrame(p['equity_curve']).sort_values('date')
    s=curve.nav.pct_change(); s.iloc[0]=curve.nav.iloc[0]/30_000_000-1
    b=market_returns.reindex(curve.date).to_numpy()
    assert np.isfinite(b).all()
    for name,r in [('策略',s),('TAIEX',b)]:
        annual.append(dict(year=y,series=name,**metrics(r)))
    paths.extend(dict(date=d,year=y,strategy_return=float(a),benchmark_return=float(z)) for d,a,z in zip(curve.date,s,b))
    for item in artifact(rid(y),'model_explain')['shap_top']:
        shaps.append(dict(year=y,signal_date=p['meta']['final_signal_date'],**item))
daily=pd.DataFrame(paths)
assert daily.date.is_unique and daily.date.is_monotonic_increasing
calendar=market.loc[daily.date.min():daily.date.max()].index
assert list(calendar)==daily.date.tolist(), 'Missing/overlapping sessions in stitched path'
for prefix in ['strategy','benchmark']:
    daily[prefix+'_wealth']=(1+daily[prefix+'_return']).cumprod()
    daily[prefix+'_drawdown']=daily[prefix+'_wealth']/daily[prefix+'_wealth'].cummax().clip(lower=1)-1
overall=pd.DataFrame([dict(series=name,**metrics(daily[col])) for name,col in [('策略','strategy_return'),('TAIEX','benchmark_return')]])
annual=pd.DataFrame(annual); shap=pd.DataFrame(shaps)
shap['relative_importance']=shap.value/shap.groupby('year').value.transform('sum')
shap_summary=shap.groupby('feature').agg(mean_abs_shap=('value','mean'),mean_relative_importance=('relative_importance','mean')).sort_values('mean_relative_importance',ascending=False).reset_index()
save(daily,'stitched_daily');save(annual,'annual_recomputed');save(overall,'stitched_metrics');save(shap,'shap_snapshots');save(shap_summary,'shap_summary')

def compare(variants, years=range(2020,2025)):
    rows=[]
    for label,suffix in variants:
        for y in years:
            p=artifact(rid(y,suffix)); m=artifact(rid(y,suffix),'monthly_ic')['summary']
            rows.append(dict(group=label,year=y,run_id=rid(y,suffix),**p['metrics'],continuous_ic=m.get('mean_continuous_rank_ic'),top15_excess=m.get('mean_top15_excess_return')))
    detail=pd.DataFrame(rows)
    mean=detail.groupby('group',sort=False)[['cagr','sharpe','sortino','max_drawdown','turnover','continuous_ic','top15_excess']].mean().reset_index()
    return detail,mean
methods, method_mean=compare([('Logistic Regression','logistic'),('XGBoost Binary',''),('XGBoost Pairwise','rank-pairwise'),('XGBoost Pseudo-Huber','pseudohuber'),('LightGBM Huber','lightgbm-huber')])
smooth,smooth_mean=compare([(f'α={a}',s) for a,s in [('1.0','pseudohuber'),('0.8','pseudohuber-smooth08'),('0.6','pseudohuber-smooth06'),('0.4','pseudohuber-smooth04')]])
for name,df in [('methods_annual',methods),('methods_summary',method_mean),('smoothing_annual',smooth),('smoothing_summary',smooth_mean)]:save(df,name)
buffers=pd.read_csv(ROOT/'output/research/three_buffer_strategies_summary_2020_2024.csv')
weights=pd.read_csv(ROOT/'output/research/equal_weight_vs_inverse_volatility_summary_2020_2024.csv')
mid=pd.read_csv(ROOT/'output/research/midmonth_augmented_vs_monthend_oos2024.csv')
cw=pd.read_csv(ROOT/'output/research/binary_classweight4_oos2020_2025_summary.csv');cw=cw[cw.variant.isin(['unweighted','classweight4_fixedparams'])]
boot=pd.read_csv(ROOT/'output/research/pseudohuber_bootstrap_summary_2020_2025.csv');boot=boot[boot.sample_size==5000]
ab=pd.read_csv(ROOT/'output/research/beta60_ablation_2020_2025_summary.csv')
topdetail,topmean=compare([('Hold30','pseudohuber'),('Hold45','pseudohuber-top45'),('Hold60','pseudohuber-top60')], [2020]);save(topdetail,'hold_threshold_2020')
def comp_table(df):
    return table(['方法','平均年度 CAGR','平均 Sharpe','平均 Sortino','平均年度 MDD','年換手 x','平均 Rank IC'],[[r.group,pct(r.cagr),num(r.sharpe),num(r.sortino),pct(r.max_drawdown),f'{r.turnover:.2f}',num(r.continuous_ic)] for r in df.itertuples()])
def image(name,alt):return f'![{alt}](output/research/research_results_assets/{name}.png)'
def src(path,label=None):return f'[{label or Path(path).name}]({path})'
summary_table=table(['六年串接指標','策略','同期 TAIEX'],[[label,(pct if key in ['cagr','max_drawdown','cumulative_return'] else num)(overall.iloc[0][key]),(pct if key in ['cagr','max_drawdown','cumulative_return'] else num)(overall.iloc[1][key])] for key,label in [('cagr','年化報酬 CAGR'),('sharpe','Sharpe'),('sortino','Sortino（專案定義）'),('max_drawdown','MDD'),('cumulative_return','累積報酬')]])
top3=shap_summary.head(3)
shaptext='；'.join(f'**{r.feature}**（{r.mean_relative_importance:.1%}）' for r in top3.itertuples())
annual_rows=[]
for y in range(2020,2026):
    s=annual[(annual.year==y)&(annual.series=='策略')].iloc[0];b=annual[(annual.year==y)&(annual.series=='TAIEX')].iloc[0]
    annual_rows.append([y,pct(s.cumulative_return),pct(b.cumulative_return),pct(s.cagr),pct(b.cagr),f'{s.sharpe:.2f} / {b.sharpe:.2f}',f'{s.sortino:.2f} / {b.sortino:.2f}',f'{s.max_drawdown:.2%} / {b.max_drawdown:.2%}'])
annual_table=table(['訊號年度','策略期間報酬','大盤期間報酬','策略 CAGR','大盤 CAGR','Sharpe 策／盤','Sortino 策／盤','MDD 策／盤'],annual_rows)
bootstrap_rows=[]
for method,label in [('monthly_iid','單月 IID'),('circular_3m','三個月區塊')]:
    sub=boot[boot.method==method];r=sub[sub.metric=='cumulative_return'].iloc[0];e=sub[sub.metric=='event_probabilities'].iloc[0]
    bootstrap_rows.append([label,pct(r['mean']),pct(r['median']),pct(r.p05),pct(r.p95),pct(e.prob_cumulative_return_lt_0),pct(e.prob_mdd_le_minus_20pct)])
text=f'''# 台股多因子機器學習選股｜研究成果

> 數據分析與機器學習職前訓練專題 · 量化分析師／機器學習工程師面試版  
> 整理日期：2026-11-05。首頁為摘要內容單元；Markdown 本身不保證實體頁數，轉 PDF 時依分頁標記排版。

## 第一頁｜摘要與結論

**核心成果：完成可追溯的台股選股研究流程。最終 XGBoost Pseudo-Huber 在六年串接 OOS 呈現正報酬與較淺回撤，但報酬落後同期大盤；較適合定位為控制風險的主動選股研究。**

最終版本：`oos-2020~2025-factor-v4-pseudohuber-NOrefit`（六個年度 run 的統稱）。四年訓練 → 一年驗證 → 次年 OOS；各年度重新訓練及調參，驗證後不合併資料 refit。月末選 Top15，既有持股排名仍在 Top30 可續抱；反波動權重，α=1、不平滑。

```text
FinMind／官方資料等 → SQLite 時點資料 → 股票池＋27 因子月截面
 → Purge／Optuna／凍結年度模型 → 次年 OOS 排名
 → Top15／Hold30＋反波動配置＋大盤 MA60 風控
 → 次日開盤模擬成交／扣成本 → 績效、SHAP、消融、Bootstrap、儀表板
```

{summary_table}

**評估範圍：**2020–2025 為訊號年度，實際估值日期 2020-02-03～2026-01-30，共 {len(daily):,} 個交易日；含各年度首日相對初始資金的報酬，TAIEX 使用相同日期的收盤對收盤報酬（首日基準為 2020-01-31 收盤）。此為年度獨立回測的報酬串接，尚非跨年承接原持股的單一帳戶重跑。大盤為價格指數，非含息指數。

**SHAP Top3：**{shaptext}。此為六個年度「最後一期截面」的 mean(|SHAP|) 各自正規化後等權平均占比，**不是全年 SHAP，也不是報酬貢獻或因果證據**。

**優點：**時點資料與 purge 降低前視風險；將交易成本及曝險控制納入評估；六個年度的平均 Rank IC、Top15 超額與 Top–Bottom spread 均為正；研究輸出可追溯到 run、模型與資料表。

**缺點：**強勢市場追不上大盤；換手偏高（六年度平均約 9.91 倍）；反覆比較 OOS 存在選模偏誤。`revenue_mom` 實為季度 QoQ。2025-08-01 曾發生全市場行情批次缺口，現已用交易所原始行情修復並重跑；系統也增加執行日覆蓋率檢查，避免未來再把資料中斷誤當成不交易。

<div style="page-break-after: always;"></div>

## 第二頁｜研究設計與績效口徑

### 問題與驗證方式

研究問題是「能否利用當時已知的基本面、價量、風險與籌碼因子，排序下期較有潛力的股票，並轉成扣成本的可執行投組？」選股能力和投組績效分開檢驗。

| 項目 | 設計 |
|---|---|
| 股票池 | 市值至少 50 億、20 日均成交值至少 2,000 萬、股價至少 10 元、上市滿 252 交易日；v4 要求關鍵財務與調整價資料 |
| 資料處理 | 財報依可得時點、還原價用於因子與標籤；截面缺值填補與極端值處理；排除規則及門檻見 config.factor_v4.yaml |
| 年度切分 | OOS 年 Y：Y−5 至 Y−2 的訓練區間（末月 purge），Y−1 驗證，Y 測試；20 交易日標籤隔離 |
| 最終模型 | 27 個共同特徵；XGBoost `reg:pseudohubererror`；每年 50 次 Optuna；NOrefit |
| 標籤 | 未來 20 個個股交易日還原價報酬；回歸預測相對當期股票池均值的連續超額報酬；二元分類以截面前 20% 為正類 |
| 個股配置 | Top15／Hold30；最多單檔 10%；反波動權重 |
| 總曝險 | 目標年波動 15%、曝險上限 100%、TAIEX 低於 MA60 時採減碼規則 |
| 執行與成本 | 月底收盤訊號、次交易日開盤；手續費單邊 0.1425%、賣出稅 0.3%、滑價單邊 0.1% |

### 必須區分的三種數字

1. **六年串接績效：**首頁由日報酬重新複利，再計算整段 CAGR、Sharpe、Sortino、MDD。年度初始資金均為 3,000 萬，串接時不把帳戶金額重設誤當損益。
2. **年度期間報酬：**每個訊號年度實際持有期間的累積報酬；不是 1 月至 12 月曆年報酬。下表另列同段 CAGR，兩者不能混稱。
3. **比較研究的年度平均：**後文多數表格是五個年度 artifact 指標的算術平均，並非五年 CAGR，平均年度 MDD 也不是整段最大回撤。

本文件首頁及下表採 `CAGR=累積財富^(252/N)−1`；Sharpe 為日平均／日報酬樣本標準差 × √252，無風險利率設為 0；Sortino 沿用專案「日平均／負報酬日樣本標準差 × √252」，**不是以全樣本下檔半變異數計算的版本**。MDD 含起始財富 1。大盤首日含前收至當日收盤變動；策略年度首日由開盤建倉及成本決定，兩者日內起點並非完全一致。

{annual_table}

{image('01_nav_drawdown','六年串接淨值與回撤，策略及 TAIEX')}

**解讀：**低曝險與減碼有助降低下跌損失，也限制多頭參與度。每年選股診斷為正，不等於每月為正，亦不等於投組必定超越 TAIEX。Top15 excess 的基準是當月股票池均值，不是 TAIEX。

來源：`database/quant.db → run_artifacts(metrics/monthly_ic)`、`prices(stock_id=TAIEX)`；重算底稿 {src('output/research/research_results_assets/annual_recomputed.csv')} 與 {src('output/research/research_results_assets/stitched_daily.csv')}。

<div style="page-break-after: always;"></div>

## 第三頁｜不同 method／objective 比較

共同評估期間為 **2020–2024**，因 Logistic、Pairwise、LightGBM 未具同系列完整 2025 結果。下表取整理當下資料庫的年度 artifact 平均，不將不同期間混排。

{comp_table(method_mean)}

| 方法 | 學習目標與比較意義 |
|---|---|
| Logistic Regression | 固定線性二元分類基準；觀察非線性模型是否增加價值 |
| XGBoost Binary | `binary:logistic`；預測進入前 20% 的機率 |
| XGBoost Pairwise | `rank:pairwise`；學習相對排序 |
| XGBoost Pseudo-Huber | `reg:pseudohubererror`；對連續超額報酬採穩健損失 |
| LightGBM Huber | `huber`；另一個樹模型與穩健回歸實作 |

{image('02_methods','不同模型平均年度報酬及 Sharpe')}

**結論：**在這組已完成實驗中，Pseudo-Huber 的平均報酬、Sharpe 與連續 Rank IC 較佳；Binary 的全截面 IC 可為負但頭部選股仍可能有價值，因此不能只用全域 IC 決定模型。LightGBM Huber 未複製 XGBoost 的結果；此比較同時改變演算法、損失實作與搜尋空間，不能推論 XGBoost 在所有資料上優於 LightGBM，亦非單純 objective 的因果實驗。

舊 README 的 Pseudo-Huber 平均 CAGR 為 12.71%、Sharpe 0.92；目前同名 baseline artifact 為 12.48%、0.967。兩者是不同保存時間的結果，**本表採目前資料庫快照，不把舊數字覆蓋成看似同次實驗**。LightGBM 原始比較摘要另有舊 IC 定義欄位；本表統一讀 `monthly_ic.summary.mean_continuous_rank_ic`。

來源：{src('output/research/research_results_assets/methods_annual.csv')}、{src('lightgbm_huber_vs_xgboost_pseudohuber_summary_2020_2024.csv')}。表格只代表已完成的模型，未虛構其他 XGBoost／LightGBM objective。

<div style="page-break-after: always;"></div>

## 第四頁｜投組緩衝與權重

### 不同緩衝與投組規模（2020–2024 保存實驗）

{table(['規則','平均 CAGR','平均 Sharpe','平均 Sortino','平均年度 MDD','換手 x','平均持股數'],[[r.strategy,pct(r.mean_cagr),num(r.mean_sharpe),num(r.mean_sortino),pct(r.mean_max_drawdown),f'{r.mean_turnover:.2f}',f'{r.mean_avg_holdings:.1f}'] for r in buffers.itertuples()])}

Top15／Hold15 相當於沒有排名緩衝；Hold30 可讓跌出前 15、但仍在前 30 的舊持股續抱，所以實際檔數可能大於 15。加入 Hold30 後平均換手由 10.76 降至 10.09 倍，但 CAGR 也下降，並未同時改善所有指標。Top30／Hold60 同時增加新買入數量與續抱範圍，不能把全部差異歸因於緩衝。

本表保留原比較 CSV 的同批基準（12.71%），不將後來重物化的 12.48% 插入舊對照。尚未重跑全部變體驗證現行資料下是否同方向。

### 更寬的續抱門檻：僅有 2020，不外推五年

{comp_table(topmean)}

此三組固定 Top15。2020 單年 Hold45／Hold60 未勝過現有 Hold30 baseline；尚不足以宣稱寬緩衝在所有年度較差。來源：當前 run artifacts；{src('output/research/research_results_assets/hold_threshold_2020.csv')}。

### 反波動與等權（2020–2024）

{table(['個股權重','平均 CAGR','平均 Sharpe','平均 Sortino','最差年度 MDD','實現波動','換手 x'],[[r.method,pct(r.mean_annual_cagr),num(r.mean_sharpe),num(r.mean_sortino),pct(r.worst_max_drawdown),pct(r.mean_realized_volatility),f'{r.mean_turnover:.2f}'] for r in weights.itertuples()])}

**結論：**反波動個股權重的平均 CAGR 高約 1.05 個百分點，Sharpe 高約 0.146，實現波動較低；換手未改善。兩組平均總曝險相同（55.42%），所以這是**個股權重**差異，不是拿低曝險策略和滿倉等權比較。

來源：{src('output/research/three_buffer_strategies_summary_2020_2024.csv')}、{src('output/research/equal_weight_vs_inverse_volatility_summary_2020_2024.csv')}。原檔的年度 CAGR 複乘欄位不作為連續淨值報酬使用。

<div style="page-break-after: always;"></div>

## 第五頁｜月中資料增廣與分數平滑

### 月中增廣：2024 單一 OOS 年度

「月中資幅」依專案檔案解讀為**月中資料增廣**：增加訓練截面，不是每月多換倉一次。

{table(['訓練資料','樣本數','截面數','驗證 IC','OOS IC','CAGR','Sharpe','MDD','換手 x'],[[r.variant,int(r.train_rows),int(r.train_snapshot_count),num(r.validation_rank_ic),num(r.continuous_rank_ic),pct(r.cagr),num(r.sharpe),pct(r.max_drawdown),f'{r.turnover:.2f}'] for r in mid.itertuples()])}

增加 20,844 筆月中資料後，驗證 IC 與 OOS IC 均下降，CAGR 減少 10.72 個百分點；MDD 改善約 2.93 個百分點，換手亦降低。**資料量變多不保證泛化改善**，可能涉及相鄰截面資訊重複與分布改變；成因尚未單獨驗證，不能作為既定因果結論。

來源：{src('output/research/midmonth_augmented_vs_monthend_oos2024.csv')}。

### 分數平滑 α（2020–2024）

`平滑分數_t = α × 當期截面百分位 + (1−α) × 上期平滑分數`；α=1 代表不平滑。股票首次出現時用當期值。

{comp_table(smooth_mean)}

{image('03_smoothing','平滑係數與平均年度報酬、換手率')}

**結論：**平滑在部分年份有效，但整體降低反應速度、拖累平均報酬與 Sharpe；換手降幅相對有限，因此最終保留 α=1。α=1 是目前 baseline，其他 α 是保存的舊實驗，仍有物化版本差異的限制，適合呈現方向性證據，不能聲稱嚴格單因子因果效果。

來源：{src('output/research/research_results_assets/smoothing_annual.csv')}；實作 `tools/run_oos_portfolio.py::_smooth_percentile_scores`。

<div style="page-break-after: always;"></div>

## 第六頁｜Class weight 與特徵消融

### 二元分類的 class weight（2020–2025）

僅比較不加權與**固定原參數、正類權重 4:1**，依需求排除 class weight＋Optuna 組。權重只作用在訓練損失，驗證／OOS 不加權。這是 Binary 模型的實驗，不能直接當作 Pseudo-Huber 的對照。

{table(['Binary 版本','串接 CAGR','Sharpe','Sortino','串接 MDD','平均 Rank IC','Top15 excess','換手 x'],[[r.variant,pct(r.stitched_cagr),num(r.stitched_sharpe),num(r.stitched_sortino),pct(r.stitched_max_drawdown),num(r.mean_continuous_rank_ic),pct(r.mean_top15_excess_return),f'{r.mean_turnover:.2f}'] for r in cw.itertuples()])}

加權後 CAGR 下降約 1.20 個百分點，Sharpe 下降 0.114，MDD 更深；但 Top15 excess 提升，顯示訓練損失、選股頭部表現與投組損益不必然同方向。以投組結果看，尚無證據支持採用固定 4:1 加權。

**口徑註：**此表沿用原研究的逐年 `pct_change().dropna()` 串接，未納入每年首日對初始資金的變動；不是首頁的含首日口徑。兩組內部可比較，但不直接用本表與首頁排名。

來源：{src('output/research/binary_classweight4_oos2020_2025_summary.csv')}、`tools/report_binary_classweight4_oos2020_2025.py::_stitched_metrics`。

### Beta 消融：SHAP 高，不等於不可替代

{table(['模型','平均 CAGR','平均 Sharpe','平均年度 MDD','平均 OOS IC','Top15 excess'],[[r.variant,pct(r.mean_oos_cagr),num(r.mean_oos_sharpe),pct(r.mean_oos_max_drawdown),num(r.mean_continuous_rank_ic),pct(r.mean_top15_excess_return)] for r in ab.itertuples()])}

固定參數移除 `beta_60d` 後，平均 IC 反而增加 0.009，但平均 Sharpe 降低 0.316、Top15 excess 減少 0.53 個百分點。72 個月配對 Bootstrap 的 IC 差異 95% 區間約 [−0.001, 0.019]，Top15 excess 差異約 [−1.42%, +0.34%]，均涵蓋 0。重新 Optuna 的 Beta 消融仍列為補充實驗；此組並非已排除的 class weight＋Optuna。

**結論：**暫時保留 Beta 有經濟結果支持，但不能因 SHAP 排名高就宣稱它是核心 Alpha；訊號可能與波動、動能及市場狀態共同作用。

來源：{src('output/research/beta60_ablation_report.md')}、{src('output/research/beta60_ablation_2020_2025_paired_monthly_summary.csv')}。

<div style="page-break-after: always;"></div>

## 第七頁｜拔靴法：未來一年可能情境的歷史模擬

以 2020–2025 的 **72 個持有月份**為樣本，每條路徑抽取 12 個月，保留每月內的日報酬。比較單月 IID 與三個月 circular block，各 5,000 條路徑；另有 500／2,000 次結果供收斂檢查。

{table(['方法','一年累積報酬均值','中位數','P5','P95','虧損機率','MDD 至少 20% 機率'],bootstrap_rows)}

{image('04_bootstrap','兩種拔靴法一年累積報酬分布')}

報酬表使用**一年累積報酬**，不是 CAGR；各抽樣路徑交易日數不同，原報告的 CAGR 中位數約 12.98%／12.86%，與此表略異。單月法的 P5～P95 是約 −8.13%～37.81%，三個月區塊法約 −5.44%～33.19%。

**解讀界線：**這是「歷史月份若能重複出現」的情境分布，不是未來預測保證，也不是平均績效的信賴區間。三個月區塊保留部分時間相依性，並不保證尾部風險估得更保守。模擬重用已扣成本的報酬，接縫不重建持股、不追加換倉成本；無法涵蓋歷史未見危機或消除選模偏誤。

Bootstrap 已補回每月首日報酬；原 artifact 月報酬用每月末 NAV／月初 NAV，會漏掉首個交易日。因此後續簡報請使用本底稿的日報酬或 Bootstrap 稽核檔，勿直接複利原 artifact 月報酬。2025-08 路徑已採補價後重新執行的 28 筆訂單。

來源：{src('output/research/pseudohuber_bootstrap_summary_2020_2025.csv')}、{src('output/research/pseudohuber_bootstrap_months_2020_2025.csv')}；{src('output/research/pseudohuber_bootstrap_distributions_2020_2025.html','既有互動式分布圖')}。

<div style="page-break-after: always;"></div>

## 第八頁｜SHAP、限制與後續研究

### 特徵解釋的證據範圍

{table(['年度最後訊號','Top1','Top2','Top3'],[[y,*shap[shap.year==y].sort_values('value',ascending=False).head(3).feature.tolist()] for y in range(2020,2026)])}

{image('05_shap','六個年度末截面的正規化 SHAP 平均重要度')}

首頁 Top3 採各年重要度總和正規化、再六年等權平均，以免單一年度預測尺度較大支配結果。最新 2025 年末截面 Top3 是 `sales_yield`、`close_60d_high`、`beta_60d`，與跨年摘要不同。`beta_60d` 表示市場敏感度；`close_60d_high` 表示股價相對 60 日高點的位置；`sales_yield` 為營收相對市值的估值因子；精確定義以程式為準。

SHAP 使用凍結模型對最後訊號月截面計算 mean(|SHAP|)，未提供方向性 beeswarm 或全年度分布，不能由絕對重要度推論「因子越高越應買」。相關因子也可能分攤重要度。來源：`model_explain.shap_top`；{src('output/research/research_results_assets/shap_snapshots.csv')}。

### 資料與外推限制

| 問題 | 對解讀的影響／下一步 |
|---|---|
| 2025-08-01 曾僅有 323 筆行情 | 已用交易所原始行情補入 1,840 筆嚴格校準資料並重跑；7 月底訊號現有 28 筆訂單。79 筆特殊商品／新股／除權息例外未強行補值，且不含該次 18 檔目標持股 |
| v4 `revenue_mom` 實為季度 QoQ | 應改名或換成真正月營收 MoM，再以新版本驗證，避免回頭覆寫已封存研究 |
| 六個年度獨立資金初始化 | 串接並未模擬前一年持股直接交接與跨年額外交易；下一步建立單帳戶連續回測 |
| 多次查看 2020–2024 OOS 選模型 | 仍有研究選擇偏誤；2025 有預註冊檔，但只是專案內封存證據，不應誇大成外部獨立驗證 |
| TAIEX 為價格指數、策略有成本與現金 | 不是相同曝險／含息／成本的完全公平基準；宜增加含息指數與曝險匹配基準 |
| 換手及執行假設 | 固定滑價無法完整刻畫成交容量、衝擊成本、漲跌停與流動性；需加入容量及成交限制測試 |
| 歷史股票池 | 有時點規則仍不能單憑設計宣稱完全消除倖存者偏誤；須另稽核下市股票與歷史覆蓋 |

### 額外成果：2026 部分年度追蹤

既有 2026 YTD 報告估值至 **2026-10-16**：策略同期間累積報酬 23.65%、TAIEX 48.97%，MDD −8.17%／−16.36%；已成熟 1–8 月訊號平均 Rank IC 0.0282。這是首個執行日 NAV 起算口徑，**不與首頁混接**。9 月訊號的 20 日選股結果尚未成熟，應保持 N/A。此處只是已保存的追蹤快照，不代表截至本文件日期的最新行情。

來源：{src('output/research/factor_v4_oos2026ytd_report.md')}。

### 面試呈現重點

- **量化分析師：**能解釋標籤、樣本外驗證、成本與曝險如何影響績效，並以消融／穩健性檢驗區分統計訊號與經濟價值。
- **機器學習工程師：**能串接資料擷取、特徵版本、凍結模型、run 級輸出與儀表板，且能辨識缺價、語意錯誤與指標口徑問題。
- **可守住的結論：**目前證據支持 Pseudo-Huber 作為後續研究基準；尚不足以宣稱可穩定打敗市場或直接實盤部署。

## 附錄｜圖表與來源管理

所有新圖均另存 PNG（簡報／PDF）及 SVG（向量編輯），位於 `output/research/research_results_assets/`；圖中文字用英文短標籤，正文提供中文解釋。資料庫讀取為唯讀，未訓練模型或改動既有研究檔案。

| 檔案 | 用途 |
|---|---|
| 01_nav_drawdown | 首頁延伸圖：淨值與回撤 |
| 02_methods | 模型方法比較，2020–2024 年度平均 |
| 03_smoothing | α 與報酬／換手的取捨 |
| 04_bootstrap | 兩種方法的一年累積報酬分布 |
| 05_shap | 六個年度末截面 Top10 重要度摘要 |
| stitched_daily.csv／annual_recomputed.csv | 首頁與年度績效可重算底稿 |
| methods_annual.csv／smoothing_annual.csv | 方法與平滑的逐年明細與 run ID |
| shap_snapshots.csv／shap_summary.csv | SHAP 的原值與正規化摘要 |
| provenance.json | 本次使用 artifact 的 SHA-256；後續來源異動可偵測 |

重建文字與底稿：`.venv/Scripts/python.exe output/research/build_research_results.py`。作圖程式：`output/research/plot_research_results.py`（需要 matplotlib、pandas）。既有研究 CSV 與新資料庫快照分開標示；不把不同版本或不同期間的數字拼成單一排名。
'''
(ROOT/'研究成果.md').write_text(text,encoding='utf-8')
csv_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'output/research').glob('*.csv') if 'bootstrap_paths' not in p.name and 'sample_indices' not in p.name}
(OUT/'provenance.json').write_text(json.dumps({'date':'2026-11-05','database':'database/quant.db','daily_count':len(daily),'first_day_included':True,'benchmark':'TAIEX close-to-close on each strategy date','artifact_sha256':sources,'research_csv_sha256':csv_hashes},ensure_ascii=False,indent=2),encoding='utf-8')
print(overall.to_string(index=False))
print(method_mean.to_string(index=False))
print(shap_summary.head(3).to_string(index=False))
C.close()
