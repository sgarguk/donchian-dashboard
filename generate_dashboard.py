#!/usr/bin/env python3
"""
Lighthouse Canton — Donchian Rotation Dashboard Generator
Runs in GitHub Actions. Outputs docs/index.html (Cloudflare Pages serves this).
Usage: python generate_dashboard.py
"""

import os, json, warnings, re
from itertools import combinations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location('/tmp')  # disable SQLite cache — prevents locks in CI

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
if pr.empty:
    raise ValueError("Download returned empty data — market may be closed or yfinance unavailable")
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
s = template.find('<script>const P=')
e = template.find(';</script>', s) + len(';</script>')
new_html = template[:s] + '<script>' + data_js + '</script>' + template[e:]
new_html = re.sub(r'As of <strong style="color:#fff">[^<]*</strong>', f'As of <strong style="color:#fff">{last_date}</strong>', new_html)
# Rebuild pairwise signal table rows with live data
pair_colors = {'SPY':'#2E86AB','QQQ':'#2E86AB','GLD':'#D4A84B','DBMF':'#27AE60'}
pair_rows_html = ''
for k in sdf.columns:
    a, b = k.split('/')
    cr = curr_ratios[k]
    pct_display = f"{int(cr['pct'])}% of range"
    if cr['above']:
        status_cls, status_txt = 'status-above', '▲ ABOVE'
    elif cr['below']:
        status_cls, status_txt = 'status-below', '▼ BELOW'
    else:
        status_cls, status_txt = 'status-within', 'IN BAND'
    dir_txt = '▲ last up' if cr['sig'] == 1 else '▼ last down'
    vote_asset = a if cr['sig'] == 1 else b
    vote_color = pair_colors.get(vote_asset, '#2E86AB')
    pair_rows_html += (
        f'<div class="sr">'
        f'<div class="sl">{k}</div>'
        f'<div class="sv">{cr["val"]}<br><span class="sm">{pct_display}</span></div>'
        f'<div class="sb"><span class="bv g">{cr["up"]}</span><span class="bl">upper</span></div>'
        f'<div class="sb"><span class="bv r">{cr["lo"]}</span><span class="bl">lower</span></div>'
        f'<div class="ss {status_cls}">{status_txt}<br>'
        f'<span style="font-size:7px;opacity:0.85">{dir_txt}</span></div>'
        f'<div class="sv" style="font-weight:700;color:{vote_color}">+1 {vote_asset}<br>'
        f'<span class="sm" style="color:{vote_color}">vote</span></div>'
        f'</div>'
    )
sr_start = new_html.find('<div class="sr">')
sr_end = new_html.find('<div id="signalLogic"')
if sr_start != -1 and sr_end != -1:
    new_html = new_html[:sr_start] + pair_rows_html + new_html[sr_end:]
# Rebuild vote tally boxes with live data
winner_asset = max(votes, key=votes.get)
asset_rgb = {'SPY':'46,134,171','QQQ':'46,134,171','GLD':'212,168,75','DBMF':'39,174,96'}
asset_hex = {'SPY':'#2E86AB','QQQ':'#2E86AB','GLD':'#D4A84B','DBMF':'#27AE60'}
stars = {0:'·',1:'★',2:'★★',3:'★★★'}
vote_boxes = ''
for asset in ['SPY','QQQ','GLD','DBMF']:
    v = votes[asset]
    is_winner = (asset == winner_asset)
    bg = f'rgba({asset_rgb[asset]},0.12)' if is_winner else '#F7F9FC'
    border = asset_hex[asset] if is_winner else '#DDE3EA'
    col = asset_hex[asset] if is_winner else '#94a3b8'
    vote_boxes += (
        f'<div style="flex:1;text-align:center;padding:6px 4px;border-radius:4px;background:{bg};border:1.5px solid {border};">'
        f'<div style="font-size:8px;color:#7F8C8D;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:2px">{asset}</div>'
        f'<div style="font-size:20px;font-weight:700;font-family:Courier New,monospace;color:{col}">{v}</div>'
        f'<div style="font-size:9px;color:{col};margin-top:1px">{stars.get(v,"·")}</div>'
        f'</div>'
    )
vote_flex = f'<div style="display:flex;gap:6px">{vote_boxes}</div>'
vt_s = new_html.find('<div style="display:flex;gap:6px"><div style="flex:1;text-align:center;padding:6px 4px;border-radius:4px;background:#F7F9FC;border:1.5px solid #DDE3EA;">')
vt_e = new_html.find('</div><div style="margin-top:7px;font-size:9px', vt_s)
if vt_s != -1 and vt_e != -1:
    new_html = new_html[:vt_s] + vote_flex + new_html[vt_e:]

# Update winner line
winner_votes = votes[winner_asset]
winner_color = asset_hex[winner_asset]
new_html = re.sub(
    r'Winner: <strong style="color:[^"]*">[^<]*</strong>',
    f'Winner: <strong style="color:{winner_color}">{winner_asset} ({winner_votes} votes)</strong>',
    new_html
)
# Update current prices in signal card
cp = payload['shared']['curr_prices']
for ticker, price in cp.items():
          new_html = re.sub(
                rf'(<div class="pt"[^>]*>{ticker}</div>\s*<div class="pv"[^>]*>\$)[0-9.]+',
                rf'\g<1>{price:.2f}',
                new_html
            )
# Update "Generated" timestamp in footer
new_html = new_html.replace(
    'Lighthouse Canton Pte. Ltd.',
    f'Lighthouse Canton Pte. Ltd. · Updated {shared["generated_utc"]}'
)

# ── Signal Pipeline Card ──────────────────────────────────────────────────────
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
_cal = USFederalHolidayCalendar()
_good_fridays = pd.to_datetime(['2019-04-19','2020-04-10','2021-04-02','2022-04-15','2023-04-07','2024-03-29','2025-04-18','2026-04-03','2027-03-26'])
_base_holidays = _cal.holidays(start='2019-01-01', end='2030-01-01')
_all_holidays  = _base_holidays.union(_good_fridays)
NYSE_day = CustomBusinessDay(holidays=_all_holidays)
def _next_bdays(from_date, n):
    days = []; d = pd.Timestamp(from_date)
    for _ in range(n): d += NYSE_day; days.append(d)
    return days
_curr_hold  = variant_data['base']['curr_active']
_raw_winner = max(votes, key=votes.get)
_raw_votes  = votes[_raw_winner]
_consec = 0
for _i in range(len(rs)-1, -1, -1):
    if rs.iloc[_i] == _raw_winner: _consec += 1
    else: break
_trigger_date = pr.index[-1]
for _i in range(len(rs)-1, -1, -1):
    if rs.iloc[_i] != _raw_winner:
        _trigger_date = rs.index[_i+1] if _i+1 < len(rs) else pr.index[-1]
        break
if _raw_winner == _curr_hold:
    _state = 'stable'; _action_date = None
_confirm_date = _next_bdays(_trigger_date, CONFIRM - 1)[-1]
_action_date  = _next_bdays(_confirm_date, SHIFT)[-1]
if _consec >= CONFIRM:
    _state = 'confirmed'
else:
    _state = 'pending'
_acolor = {'SPY':'#2E86AB','QQQ':'#2E86AB','GLD':'#D4A84B','DBMF':'#27AE60'}
_cc = _acolor.get(_curr_hold,'#2E86AB'); _wc = _acolor.get(_raw_winner,'#2E86AB')
if _state == 'stable':
    _cb = '#DDE3EA'; _cbg = '#F7F9FC'
    _sh = (f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px"><span style="font-size:16px">✓</span><span style="font-size:11px;font-weight:700;color:#475569">Signal Stable · No Change Pending</span></div>'
           f'<div style="font-size:10px;color:#7F8C8D">Current hold <strong style="color:{_cc}">{_curr_hold}</strong> leads with <strong>{votes[_curr_hold]}</strong> votes · holding position</div>')
elif _state == 'pending':
    _cb = '#E67E22'; _cbg = '#FEFBF6'; _pct = int(_consec/CONFIRM*100)
    _sh = (f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px"><span style="font-size:14px">⏳</span><span style="font-size:11px;font-weight:700;color:#E67E22">Pending Confirmation</span></div>'
           f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">'
           f'<div style="background:#fff;border-radius:4px;padding:7px 9px;border:1px solid #DDE3EA"><div style="font-size:8px;color:#7F8C8D;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px">Current Hold</div><div style="font-size:16px;font-weight:700;font-family:Courier New,monospace;color:{_cc}">{_curr_hold}</div></div>'
           f'<div style="background:#fff;border-radius:4px;padding:7px 9px;border:1px solid {_wc}44"><div style="font-size:8px;color:#7F8C8D;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px">Challenger ({_raw_votes}v)</div><div style="font-size:16px;font-weight:700;font-family:Courier New,monospace;color:{_wc}">{_raw_winner}</div></div>'
           f'</div><div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;font-size:9px;color:#7F8C8D;margin-bottom:3px"><span>Confirmation</span><span>{_consec} of {CONFIRM} days</span></div>'
           f'<div style="background:#EEF0F3;border-radius:3px;height:6px;overflow:hidden"><div style="width:{_pct}%;background:#E67E22;height:100%;border-radius:3px"></div></div></div>'
           f'<div style="font-size:10px;color:#7F8C8D">Trigger: <strong>{_trigger_date.strftime("%d %b %Y")}</strong> &nbsp;·&nbsp; Est. action: <strong style="color:#E67E22">{_action_date.strftime("%d %b %Y")}</strong> &nbsp;(if confirmed)</div>')
else:
    _cb = '#27AE60'; _cbg = '#F4FCF7'
    _sh = (f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px"><span style="font-size:14px">⚡</span><span style="font-size:11px;font-weight:700;color:#27AE60">Change Confirmed · Awaiting Execution</span></div>'
           f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">'
           f'<div style="background:#fff;border-radius:4px;padding:7px 9px;border:1px solid #DDE3EA"><div style="font-size:8px;color:#7F8C8D;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px">Exiting</div><div style="font-size:16px;font-weight:700;font-family:Courier New,monospace;color:{_cc}">{_curr_hold}</div></div>'
           f'<div style="background:#fff;border-radius:4px;padding:7px 9px;border:1px solid {_wc}88"><div style="font-size:8px;color:#7F8C8D;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px">Entering ({_raw_votes}v)</div><div style="font-size:16px;font-weight:700;font-family:Courier New,monospace;color:{_wc}">{_raw_winner}</div></div>'
           f'</div><div style="font-size:10px;color:#475569">Confirmed: <strong>{_trigger_date.strftime("%d %b %Y")}</strong> &nbsp;·&nbsp; Consecutive days: <strong>{_consec}</strong> &nbsp;·&nbsp; Action date: <strong style="color:#27AE60">{_action_date.strftime("%d %b %Y")}</strong> &nbsp;·&nbsp; Execute at close</div>')
_pipeline_card = (f'<div class="card" style="margin-top:13px;border-top:3px solid {_cb};background:{_cbg}"><div class="ch" style="background:var(--nv2)"><span>Signal Pipeline</span><span>C={CONFIRM} days · Shift={SHIFT} days · NYSE calendar</span></div><div class="cb" style="padding:12px 13px">{_sh}</div></div>')
new_html = new_html.replace('<div id="pipelinePlaceholder"></div>', _pipeline_card)
os.makedirs('docs', exist_ok=True)
with open(OUT, 'w') as f:
    f.write(new_html)

print(f"\n✓ Dashboard written to {OUT}")
print(f"  Base signal: {variant_data['base']['curr_active']} "
      f"({variant_data['base']['days_in']} days)")
print(f"  Votes: {votes}")
