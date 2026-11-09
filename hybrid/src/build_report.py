import html
from pathlib import Path
import pandas as pd

def fmt(x):
    return '—' if pd.isna(x) else f'{x:,.2f}'

def build_report(allocation:pd.DataFrame,path:Path,settings=None):
    freq=(settings or {}).get('reporting',{}).get('frequency','weekly')
    tabs=[]
    if not allocation.empty:
        for (country,brand),group in allocation.groupby(['country_code','brand']):
            sections=[]
            for period,sub in group.groupby('period_start',sort=True):
                sub=sub.sort_values('modelled_qs',ascending=False)
                rows=[]
                for _,r in sub.iterrows():
                    vals=[r['channel'],fmt(r['spend']),fmt(r['attributed_qs']),fmt(r['last_click_qs']),fmt(r['modelled_qs']),fmt(r['platform_cpqs']),fmt(r['modelled_cpqs']),fmt(r['projected_op']),fmt(r['projected_cohorted_cac'])]
                    rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in vals)+'</tr>')
                summary=f"Actual QS: {fmt(sub['actual_qs'].iloc[0])} · Modelled channel QS: {fmt(sub['modelled_qs'].sum())} · Baseline QS: {fmt(sub['baseline_qs'].iloc[0])}"
                heads=['Channel','Spend (local)','Attributed QS','Last-click QS','Modelled QS','Platform CPQS','Modelled CPQS','Projected OP','Projected CAC']
                header=''.join(f'<th>{html.escape(h)}</th>' for h in heads)
                sections.append(f'<section><h2>{html.escape(str(period.date()))}</h2><p>{summary}</p><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')
            tabs.append((f'{country.upper()} · {brand}',''.join(sections)))
    buttons=''.join(f'<button onclick="show({i})">{html.escape(label)}</button>' for i,(label,_) in enumerate(tabs))
    panels=''.join(f'<div class="panel" id="panel{i}">{content}</div>' for i,(_,content) in enumerate(tabs))
    doc='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MOE Hybrid</title><style>body{font:15px system-ui,Arial;margin:32px;background:#f7f8fa;color:#182333}h1{margin-bottom:4px}p{color:#526073}button{border:1px solid #ccd3dd;border-radius:8px;background:white;padding:10px;margin:4px;cursor:pointer}.panel{display:none}section{background:white;border:1px solid #e0e4ea;border-radius:12px;padding:18px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:right;padding:11px;border-bottom:1px solid #e6e9ed}th:first-child,td:first-child{text-align:left}th{background:#f2f4f7}</style></head><body><h1>MOE Hybrid · QS contribution</h1>'''+f'<p>{html.escape(str(freq).title())} reporting · historical constrained allocation, not causal incrementality. Platform conversions may overlap; projections use configured assumptions.</p><nav>'+buttons+'</nav>'+panels+'''<script>function show(i){document.querySelectorAll('.panel').forEach((e,j)=>e.style.display=i===j?'block':'none')}show(0)</script></body></html>'''
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(doc,encoding='utf-8')
