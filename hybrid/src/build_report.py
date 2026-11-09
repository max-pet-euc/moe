import html
import json
from pathlib import Path
import pandas as pd


def _num(x, money=False):
    if pd.isna(x): return '—'
    return ('£' if money else '') + f'{x:,.2f}'


def build_report(allocation: pd.DataFrame, path: Path):
    tabs=[]
    if not allocation.empty:
        for (country,brand,problem), group in allocation.groupby(['country_code','brand','problem_type']):
            months=[]
            for month, sub in group.groupby('month'):
                sub=sub.sort_values('modelled_qs',ascending=False)
                rows=[]
                for _,r in sub.iterrows():
                    rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['channel'],_num(r['spend'],True),_num(r['attributed_qs']),_num(r['last_click_qs']),_num(r['modelled_qs']),_num(r['platform_cpqs'],True),_num(r['modelled_cpqs'],True),_num(r['projected_op']),_num(r['projected_cohorted_cac'],True)])+'</tr>')
                summary=f"Actual QS: {_num(sub['actual_qs'].iloc[0])} · Allocated channel QS: {_num(sub['modelled_qs'].sum())} · Baseline QS: {_num(sub['baseline_qs'].iloc[0])}"
                header=''.join(f'<th>{h}</th>' for h in ['Channel','Spend','Attributed QS','Last-click QS','Modelled QS','Platform CPQS','Modelled CPQS','Projected OP','Projected CAC'])
                months.append(f'<section><h2>{html.escape(month)}</h2><p>{summary}</p><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')
            tabs.append((f'{country.upper()} · {brand} · {problem}', ''.join(months)))
    buttons=''.join(f'<button onclick="show({i})">{html.escape(label)}</button>' for i,(label,_) in enumerate(tabs))
    panels=''.join(f'<div class="panel" id="panel{i}">{content}</div>' for i,(_,content) in enumerate(tabs))
    doc='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MOE Hybrid</title><style>body{font:15px system-ui,Arial;margin:32px;background:#f7f8fa;color:#182333}h1{margin-bottom:4px}p{color:#526073}button{border:1px solid #ccd3dd;border-radius:8px;background:white;padding:10px;margin:4px;cursor:pointer}.panel{display:none}section{background:white;border:1px solid #e0e4ea;border-radius:12px;padding:18px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:right;padding:11px;border-bottom:1px solid #e6e9ed}th:first-child,td:first-child{text-align:left}th{background:#f2f4f7}</style></head><body><h1>MOE Hybrid · QS contribution</h1><p>Historical constrained allocation; not causal incrementality. Platform conversions may overlap. Projections use configured assumptions.</p><nav>'''+buttons+'</nav>'+panels+'''<script>function show(i){document.querySelectorAll('.panel').forEach((e,j)=>e.style.display=i===j?'block':'none')}show(0)</script></body></html>'''
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(doc,encoding='utf-8')
