"""Static export figures for 研究成果.md; inputs are preserved CSV snapshots."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
# Optional isolated local plotting dependencies; never modify the project venv.
sys.path.append(str(ROOT/'tmp/research_results_plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np

OUT=ROOT/'output/research/research_results_assets'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'#f8fafc','grid.alpha':.2})
blue='#176b91'; orange='#c66b27'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
d=pd.read_csv(OUT/'stitched_daily.csv',parse_dates=['date'])
fig,ax=plt.subplots(2,1,figsize=(10,6.7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
for key,label,col in [('strategy','Pseudo-Huber',blue),('benchmark','TAIEX price index',orange)]:
    ax[0].plot(d.date,d[key+'_wealth'],label=label,color=col,lw=1.8)
    ax[1].plot(d.date,d[key+'_drawdown'],color=col,lw=1.3)
ax[0].set(title='2020–2025 OOS signal years | stitched annual runs',ylabel='Wealth (initial = 1)')
ax[0].legend(frameon=False);ax[1].set(ylabel='Drawdown',xlabel='Valuation date: Feb 2020 – Jan 2026')
ax[1].yaxis.set_major_formatter(PercentFormatter(1))
for a in ax:a.grid(True)
save(fig,'01_nav_drawdown')
m=pd.read_csv(OUT/'methods_summary.csv')
labels=['Logistic','XGB Binary','XGB Pairwise','XGB Pseudo-Huber','LGBM Huber']
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
colors=[blue if 'Pseudo' in x else '#94a3b8' for x in labels]
for a,key,title in zip(axes,['cagr','sharpe'],['Mean annual CAGR','Mean annual Sharpe']):
    a.barh(labels,m[key],color=colors);a.invert_yaxis();a.set_title(title);a.grid(axis='x')
    if key=='cagr':a.xaxis.set_major_formatter(PercentFormatter(1))
fig.suptitle('Method comparison | 2020–2024 annual artifact means',fontsize=14)
save(fig,'02_methods')
s=pd.read_csv(OUT/'smoothing_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
labels=['1.0','0.8','0.6','0.4']
for a,key,title in zip(axes,['cagr','turnover'],['Mean annual CAGR','Mean annual turnover (x)']):
    a.bar(labels,s[key],color=[blue,'#83b6cc','#83b6cc','#83b6cc']);a.set(title=title,xlabel='Smoothing alpha');a.grid(axis='y')
    if key=='cagr':a.yaxis.set_major_formatter(PercentFormatter(1))
fig.suptitle('Score smoothing | 2020–2024 annual artifact means',fontsize=14)
save(fig,'03_smoothing')
p=pd.read_csv(ROOT/'output/research/pseudohuber_bootstrap_paths_2020_2025.csv')
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
bins=np.linspace(p.cumulative_return.min(),p.cumulative_return.max(),65)
for method,label,col in [('monthly_iid','Monthly IID',blue),('circular_3m','Circular 3-month',orange)]:
    v=p.loc[p.method==method,'cumulative_return']
    assert len(v)==5000
    ax.hist(v,bins=bins,weights=np.ones(len(v))/len(v),alpha=.52,label=label,color=col)
ax.axvline(0,color='#475569',lw=1,ls='--');ax.legend(frameon=False)
ax.set(title='One-year scenario returns | 5,000 paths per method',xlabel='12-month cumulative return',ylabel='Fraction of simulated paths')
ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y')
save(fig,'04_bootstrap')
s=pd.read_csv(OUT/'shap_summary.csv').head(10)
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
ax.barh(s.feature,s.mean_relative_importance,color=[blue]*3+['#94a3b8']*7);ax.invert_yaxis()
ax.set(title='SHAP | six year-end cross-sections, 2020–2025',xlabel='Mean within-year normalized mean absolute SHAP')
ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x')
save(fig,'05_shap')
print('Exported 5 PNG and 5 SVG figures.')
