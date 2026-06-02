#!/usr/bin/env python3
"""
Lighthouse Canton — Donchian Rotation Dashboard Generator
Runs in GitHub Actions. Outputs docs/index.html (Cloudflare Pages serves this).
Usage: python generate_dashboard.py
"""

import os, json, warnings
from itertools import combinations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Parameters ────────────────────────────────────────────────────────────────
N, CONFIRM, SHIFT, TC = 30, 2, 2, 3/10_000
BASE   = ['SPY','QQQ','GLD','DBMF']
ALL    = ['SPY','QQQ','GLD','DBMF','QLD','TQQQ']
NAMES  = {'SPY':'SPDR S&P 500','QQQ':'Nasdaq-100','GLD':'Gold',
          'DBMF':'Managed Futures','QLD':'QQQ 2× (QLD)','TQQQ':'QQQ 3× (TQQQ)'}
VARIANTS = {
    'base':  {'label':'Base QQQ',      'qqq_map':{'QQQ':1.0},                'color':'#2E86AB','badge':'1×'},
    'qld':   {'label':'QLD  (2×)',      'qqq_map':{'QLD':1.0},                'color':'#8E44AD','badge':'2×'},
    'tqqq':  {'label':'TQQQ (3×)',     'qqq_map':{'TQQQ':1.0},               'color':'#C0392B','badge':'3×'},
    'blend': {'label':'Blend (⅓ each)','qqq_map':{'QQQ':1/3,'QLD':1/3,'TQQQ':1/3},'color':'#E67E22','badge':'~2×'},
}
START = '2019-05-08'
OUT   = os.path.join('docs', 'index.html')

# ── Data ──────────────────────────────────────────────────────────────────────
print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] Downloading data...")
raw = yf.download(ALL, start=START, auto_adjust=False, progress=False)
pr  = raw['Adj Close'][ALL].dropna()
ret = pr.pct_change()
last_date = pr.index[-1].strftime('%d %b %Y')
print(f"  Data through: {last_date}  ({len(pr)} days)")

# ── Signal engine ─────────────────────────────────────────────────────────────
def don_signal(ratio):
    up=ratio.shift(1).rolling(N).max(); lo=ratio.shift(1).rolling(N).min()
    pos=0; s=[]
    for i in range(len(ratio)):
        if pd.isna(up.iloc[i]) or pd.isna(lo.iloc[i]): s.append(0); continue
        if ratio.iloc[i]>up.iloc[i]: pos=1
        elif ratio.iloc[i]<lo.iloc[i]: pos=-1
        s.append(pos)
    return pd.Series(s,index=ratio.index).replace(0,np.nan).ffill().fillna(0)

pairs = list(combinations(BASE,2))
sigs  = {f"{a}/{b}": don_signal(pr[a]/pr[b]) for a,b in pairs}
ups   = {k: pr[k.split('/')[0]].div(pr[k.split('/')[1]]).shift(1).rolling(N).max() for k in sigs}
los   = {k: pr[k.split('/')[0]].div(pr[k.split('/')[1]]).shift(1).rolling(N).min() for k in sigs}
ratios= {k: pr[k.split('/')[0]].div(pr[k.split('/')[1]]) for k in sigs}
sdf   = pd.DataFrame(sigs).dropna(); valid=sdf.index[0]

def vc(row):
    v={a:0 for a in BASE}
    for k,sig in row.items():
        a,b=k.split('/')
        if sig==1: v[a]+=1
        elif sig==-1: v[b]+=1
    return v
vdf = sdf.apply(lambda r: pd.Series(vc(r)), axis=1)

def h2h(tied, sr):
    if len(tied)==2:
        a,b=tied[0],tied[1]; k=f"{a}/{b}" if f"{a}/{b}" in sr.index else f"{b}/{a}"
        if k in sr.index:
            s=sr[k]; return a if (f"{a}/{b}"==k and s==1) or (f"{b}/{a}"==k and s==-1) else b
    return None

rh=[]
for idx in sdf.index:
    v=vdf.loc[idx]; mx=v.max(); tied=[a for a,vv in v.items() if vv==mx]
    rh.append(tied[0] if len(tied)==1 else (h2h(tied,sdf.loc[idx]) or tied[0]))
rs=pd.Series(rh,index=sdf.index); hold=rs.copy()
for i in range(1,len(hold)):
    if hold.iloc[i]!=hold.iloc[i-1]:
        stable=all(rs.iloc[i]==rs.iloc[max(0,i-j)] for j in range(1,CONFIRM))
        if not stable: hold.iloc[i]=hold.iloc[i-1]

# ── Build variants ────────────────────────────────────────────────────────────
def build_variant(qqq_map):
    tkrs = list(set(['SPY','GLD','DBMF']+list(qqq_map.keys())))
    h_lag= hold.shift(SHIFT)
    alloc= pd.DataFrame(0.0,index=sdf.index,columns=tkrs)
    for idx in sdf.index:
        sig=h_lag.loc[idx]
        if pd.isna(sig): continue
        if sig=='QQQ':
            for t,w in qqq_map.items():
                if t in alloc.columns: alloc.loc[idx,t]=w
        elif sig in ['SPY','GLD','DBMF']: alloc.loc[idx,sig]=1.0
    raw_r=(alloc*ret[tkrs].loc[sdf.index]).sum(axis=1)
    to_r =alloc.diff().abs().sum(axis=1).fillna(0)/2
    strat=(raw_r-to_r*TC*2).loc[valid:]
    active=alloc.loc[valid:].idxmax(axis=1); n_sw=int((alloc.diff().abs().sum(axis=1)>0).loc[valid:].sum())
    return strat, active, n_sw

def met(r):
    r=r.dropna(); yrs=len(r)/252
    cagr=(1+r).prod()**(1/yrs)-1; vol=r.std()*np.sqrt(252)
    sh=r.mean()*252/(r.std()*np.sqrt(252)) if r.std()>0 else 0
    cum=(1+r).cumprod(); dd=(cum/cum.cummax())-1; mdd=dd.min()
    sor_d=r[r<0].std()*np.sqrt(252); sor=r.mean()*252/sor_d if sor_d>0 else 0
    return dict(cagr=round(cagr*100,2),vol=round(vol*100,2),sharpe=round(sh,2),
                mdd=round(mdd*100,2),calmar=round(cagr/abs(mdd) if mdd!=0 else 0,2),
                sortino=round(sor,2),total=round(((1+r).prod()-1)*100,1))

def ts(s): return [[d.strftime('%Y-%m-%d'),round(v,6)] for d,v in s.dropna().items()]
def tr(r):  return round(((1+r.dropna()).prod()-1)*100,1)
def regime_blocks(series):
    blocks=[]; cur=series.iloc[0]; seg=series.index[0]
    for dt,h in series.items():
        if h!=cur:
            blocks.append({'asset':cur,'start':seg.strftime('%Y-%m-%d'),'end':dt.strftime('%Y-%m-%d')})
            cur=h; seg=dt
    blocks.append({'asset':cur,'start':seg.strftime('%Y-%m-%d'),'end':series.index[-1].strftime('%Y-%m-%d')})
    return blocks

variant_data={}
for vkey,vcfg in VARIANTS.items():
    strat,active,nsw = build_variant(vcfg['qqq_map'])
    m=met(strat)
    sw_dates=active[active!=active.shift(1)].index[1:]
    last_sw=sw_dates[-1] if len(sw_dates)>0 else valid
    days_in=(pr.index[-1]-last_sw).days
    sw_log=[]
    for dt in sw_dates:
        prev=active.loc[:dt].iloc[-2]; curr_h=active.loc[dt]
        v=vdf.loc[dt]; mx=v.max(); tied=[a for a,vv in v.items() if vv==mx]; is_tie=len(tied)>1
        trigs=[]
        for k in sdf.columns:
            a,b=k.split('/')
            if pd.notna(ups[k].loc[dt]):
                if ratios[k].loc[dt]>ups[k].loc[dt]: trigs.append(f'↑{a[:1]}{b[:1]}')
                elif ratios[k].loc[dt]<los[k].loc[dt]: trigs.append(f'↓{a[:1]}{b[:1]}')
        pv=[x for x in sw_dates if x<dt]
        sw_log.append({'date':dt.strftime('%d %b %Y'),'from_':str(prev),'to':str(curr_h),
                       'trigger':','.join(trigs[:3]) or ('H2H' if is_tie else 'Vote'),
                       'is_tie':is_tie,'days':(dt-pv[-1]).days if pv else 0})
    prev_hold=sw_log[-1]['from_'] if sw_log else 'N/A'
    variant_data[vkey]={
        'label':vcfg['label'],'color':vcfg['color'],'badge':vcfg['badge'],
        'instruments':list(vcfg['qqq_map'].keys()),
        'perf':m,'perf_spy':met(ret['SPY'].loc[valid:]),'perf_qqq':met(ret['QQQ'].loc[valid:]),
        'ytd_2025':tr(strat['2025-01-01':'2025-12-31']),'ytd_2026':tr(strat['2026-01-01':]),
        'ytd_since':tr(strat['2025-01-01':]),
        'spy_ytd25':tr(ret['SPY']['2025-01-01':'2025-12-31']),'spy_ytd26':tr(ret['SPY']['2026-01-01':]),
        'spy_since':tr(ret['SPY']['2025-01-01':]),
        'qqq_ytd25':tr(ret['QQQ']['2025-01-01':'2025-12-31']),'qqq_ytd26':tr(ret['QQQ']['2026-01-01':]),
        'n_switches':nsw,'curr_active':active.iloc[-1],'signal_hold':hold.iloc[-1],
        'days_in':int(days_in),'last_sw_date':last_sw.strftime('%d %b %Y'),'prev_hold':prev_hold,
        'ytd_equity':ts((1+strat.loc['2025-01-01':]).cumprod()*100),
        'ytd_blocks':regime_blocks(active.loc['2025-01-01':]),
        'sw_log':sw_log[-30:],'ytd_sw':[s for s in sw_log if '2025' in s['date'] or '2026' in s['date']],
        'annual':{str(yr):round(((1+strat[strat.index.year==yr]).prod()-1)*100,1)
                  for yr in sorted(strat.index.year.unique())},
    }
    print(f"  {vcfg['label']:22}: active={active.iloc[-1]}  days={days_in}  CAGR={m['cagr']:.1f}%")

# Shared data
curr_ratios={}
votes={a:0 for a in BASE}
for k in sdf.columns:
    a,b=k.split('/')
    rv=round(ratios[k].iloc[-1],4); uv=round(ups[k].iloc[-1],4); lv=round(los[k].iloc[-1],4)
    pct=(rv-lv)/(uv-lv)*100 if (uv-lv)>0 else 50
    sig_val=int(sigs[k].iloc[-1])
    curr_ratios[k]={'val':rv,'up':uv,'lo':lv,'sig':sig_val,
                    'above':bool(rv>uv),'below':bool(rv<lv),'pct':round(pct,0)}
    if sig_val==1: votes[a]+=1
    elif sig_val==-1: votes[b]+=1

shared={
    'curr_date':last_date,'generated_utc':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    'full_start':valid.strftime('%b %Y'),'full_end':pr.index[-1].strftime('%b %Y'),
    'curr_prices':{a:round(pr[a].iloc[-1],2) for a in ALL if a in pr.columns},
    'curr_ratios':curr_ratios,'votes':votes,
    'ratio_charts':{k:{'ratio':ts(ratios[k].loc['2025-01-01':]),
                       'up':ts(ups[k].loc['2025-01-01':]),
                       'lo':ts(los[k].loc['2025-01-01':])} for k in sdf.columns},
    'pairs':list(sdf.columns),
    'ytd_spy_eq':ts((1+ret['SPY'].loc['2025-01-01':]).cumprod()*100),
    'ytd_qqq_eq':ts((1+ret['QQQ'].loc['2025-01-01':]).cumprod()*100),
    'spy_annual':{str(yr):round(((1+ret['SPY'].loc[valid:][ret['SPY'].loc[valid:].index.year==yr]).prod()-1)*100,1)
                  for yr in sorted(ret['SPY'].loc[valid:].index.year.unique())},
    'qqq_annual':{str(yr):round(((1+ret['QQQ'].loc[valid:][ret['QQQ'].loc[valid:].index.year==yr]).prod()-1)*100,1)
                  for yr in sorted(ret['QQQ'].loc[valid:].index.year.unique())},
}

payload={'variants':variant_data,'shared':shared}

# ── Read HTML template and embed data ─────────────────────────────────────────
# Read the template (the dashboard HTML we built), swap in fresh data
with open('dashboard_template.html','r') as f:
    template = f.read()

json_str = json.dumps(payload).replace('</script>', '<\\/script>')
data_js = f"const P={json_str};"
# Replace the data block
import re
new_html = re.sub(r'<script>const P=\{.*?\};</script>', lambda m: f'<script>{data_js}</script>', template, flags=re.DOTALL)
                  f'<script>{data_js}</script>',
                  template, flags=re.DOTALL)

# Update "Generated" timestamp in footer
new_html = new_html.replace(
    'Lighthouse Canton Pte. Ltd.',
    f'Lighthouse Canton Pte. Ltd. · Updated {shared["generated_utc"]}'
)

os.makedirs('docs', exist_ok=True)
with open(OUT, 'w') as f:
    f.write(new_html)

print(f"\n✓ Dashboard written to {OUT}")
print(f"  Base signal: {variant_data['base']['curr_active']} "
      f"({variant_data['base']['days_in']} days)")
print(f"  Votes: {votes}")
