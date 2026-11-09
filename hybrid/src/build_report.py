import html
from pathlib import Path
import pandas as pd
import numpy as np

TITLE = 'Marketing Efficiency Report - Blended CAC Model'
COLS = [('channel','Channel','text'),('spend','Spend (local)','neutral'),('attributed_qs','Attributed QS','high'),('last_click_qs','Last-click QS','high'),('modelled_qs','Modelled QS','high'),('platform_cpqs','Platform CPQS','low'),('modelled_cpqs','Modelled CPQS','low'),('projected_op','Projected OP','high'),('projected_cohorted_cac','Projected CAC','low')]

def fmt(v):
    return '—' if pd.isna(v) or not np.isfinite(float(v)) else f'{float(v):,.2f}'

def ratio(a,b):
    return a/b if pd.notna(a) and pd.notna(b) and b>0 else np.nan

def colour(v, series, direction):
    if direction=='text' or pd.isna(v) or not np.isfinite(float(v)): return ''
    nums=pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    if nums.empty: return ''
    lo,hi=nums.min(),nums.max()
    t=0.5 if hi==lo else (float(v)-lo)/(hi-lo)
    if direction=='low': t=1-t
    if direction=='neutral': return f'background:rgba(91,116,154,{0.06+0.17*t:.3f})'
    red=np.array([254,226,226]); amber=np.array([255,244,205]); green=np.array([209,245,220])
    rgb=red*(1-2*t)+amber*(2*t) if t<=0.5 else amber*(2-2*t)+green*(2*t-1)
    return 'background:rgb({},{},{})'.format(*rgb.astype(int))

def table_for(sub):
    sub=sub.sort_values('modelled_qs',ascending=False)
    sums={key:sub[key].sum(min_count=1) for key in ['spend','attributed_qs','last_click_qs','modelled_qs','projected_op']}
    totals={'channel':'Total',**sums,'platform_cpqs':ratio(sums['spend'],sums['attributed_qs']),'modelled_cpqs':ratio(sums['spend'],sums['modelled_qs']),'projected_cohorted_cac':ratio(sums['spend'],sums['projected_op'])}
    head=''.join(f'<th>{html.escape(label)}</th>' for _,label,_ in COLS)
    rows=[]
    for _,r in sub.iterrows():
        cells=[]
        for key,_,direction in COLS:
            value=html.escape(str(r[key])) if key=='channel' else fmt(r[key])
            style=colour(r[key],sub[key],direction) if key!='channel' else ''
            cells.append(f'<td style="{style}">{value}</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    foot=''.join(f'<td>{html.escape(str(totals[key])) if key=="channel" else fmt(totals[key])}</td>' for key,_,_ in COLS)
    return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(rows)+'</tbody><tfoot><tr>'+foot+'</tr></tfoot></table></div>'

def build_report(allocation:pd.DataFrame,path:Path,settings=None,market_status=None):
    freq=(settings or {}).get('reporting',{}).get('frequency','weekly')
    tolerance=(settings or {}).get('minimum_qs_tolerance',0.10)
    tabs=[]
    if not allocation.empty:
        for (country,brand),group in allocation.groupby(['country_code','brand']):
            sections=[]
            for period in sorted(group['period_start'].dropna().unique(),reverse=True)[:3]:
                sub=group.loc[group['period_start']==period]
                summary=f"Actual QS: {fmt(sub['actual_qs'].iloc[0])} · Modelled channel QS: {fmt(sub['modelled_qs'].sum())} · Baseline QS: {fmt(sub['baseline_qs'].iloc[0])}"
                sections.append(f'<section><h2>{html.escape(str(pd.Timestamp(period).date()))}</h2><p>{summary}</p>{table_for(sub)}</section>')
            tabs.append((f'{country.upper()} · {brand}',''.join(sections)))
    if market_status is not None and not market_status.empty:
        for _,item in market_status.loc[market_status['status']!='valid'].iterrows():
            label=f"{str(item['country_code']).upper()} · {item['brand']}"
            reason=html.escape(str(item['reason']))
            tabs.append((label,f'<section><h2>Invalid data</h2><p>{reason}</p><p>No modelled QS, CPQS or CAC generated for this market.</p></section>'))
    if not tabs: tabs.append(('No eligible markets','<section><h2>No eligible markets</h2><p>Check market_status.csv and validation_report.csv.</p></section>'))
    buttons=''.join(f'<button onclick="show({i})">{html.escape(label)}</button>' for i,(label,_) in enumerate(tabs))
    panels=''.join(f'<div class="panel" id="panel{i}">{content}</div>' for i,(_,content) in enumerate(tabs))
    css='body{font:15px system-ui,Arial;margin:32px;background:#f7f8fa;color:#182333}h1{margin-bottom:4px}p{color:#526073}button{border:1px solid #ccd3dd;border-radius:8px;background:white;padding:10px;margin:4px;cursor:pointer}.panel{display:none}section{background:white;border:1px solid #e0e4ea;border-radius:12px;padding:18px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:11px;border-bottom:1px solid #e6e9ed}th:first-child,td:first-child{text-align:left}th{background:#f2f4f7}tfoot td{background:#e8edf5;font-weight:750;border-top:2px solid #9aaac1;border-bottom:0}'
    doc='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(TITLE)+'</title><style>'+css+'</style></head><body><h1>'+html.escape(TITLE)+'</h1>'+f'<p>{html.escape(str(freq).title())} reporting · latest 3 periods · {tolerance:.0%} minimum-QS tolerance · constrained allocation, not causal incrementality.</p><nav>'+buttons+'</nav>'+panels+'<script>function show(i){document.querySelectorAll(".panel").forEach((e,j)=>e.style.display=i===j?"block":"none")}show(0)</script></body></html>'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(doc,encoding='utf-8')
