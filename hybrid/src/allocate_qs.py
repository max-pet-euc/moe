import numpy as np
import pandas as pd
from scipy.optimize import minimize

GROUP = ['country_code','brand','problem_type']


def allocate_monthly(channels, constraints, targets, fit_daily, settings):
    ch = channels.merge(constraints[['channel','enabled','minimum_source','minimum_multiplier','constraint_type']], on='channel', how='inner')
    ch = ch.loc[ch['enabled']].copy()
    ch['month'] = ch['date_day'].dt.to_period('M').astype(str)
    ch['floor'] = np.where((ch['constraint_type'] == 'hard') & (ch['minimum_source'] == 'last_click_qs'),ch['last_click_qs'].fillna(0)*ch['minimum_multiplier'],0)
    t = targets.copy(); t['month'] = t['date_day'].dt.to_period('M').astype(str)
    f = fit_daily.copy(); f['month'] = f['date_day'].dt.to_period('M').astype(str)
    actual = t.groupby(GROUP+['month'],as_index=False)['actual_qs'].sum()
    totals = ch.groupby(GROUP+['month','channel'],as_index=False).agg(spend=('spend','sum'),attributed_qs=('attributed_qs',lambda x:x.sum(min_count=1)),last_click_qs=('last_click_qs',lambda x:x.sum(min_count=1)),floor=('floor','sum'))
    records=[]
    for market_month, a in actual.groupby(GROUP+['month'],dropna=False):
        a_total=float(a['actual_qs'].sum())
        sub=totals
        for key,value in zip(GROUP+['month'],market_month): sub=sub.loc[sub[key]==value]
        if sub.empty: continue
        fsub=f
        for key,value in zip(GROUP+['month'],market_month): fsub=fsub.loc[fsub[key]==value]
        names=sub['channel'].tolist()
        prior=[]
        for name in names:
            cols=[c for c in fsub if c.endswith('__'+name) and c.startswith('contribution__')]
            prior.append(float(fsub[cols].sum().sum()) if cols else 0.)
        prior=np.array(prior,dtype=float)
        floors=sub['floor'].to_numpy(dtype=float)
        if floors.sum()>a_total+1e-6:
            raise ValueError(f'infeasible floors for {market_month}: {floors.sum()} > {a_total}')
        # Model evidence provides the relative weights; residual QS goes to baseline.
        # No claim that this is a causal incremental allocation.
        desired_base=max(float(settings['baseline_prior_share'])*a_total, 0)
        desired_channels=prior.copy()
        if desired_channels.sum()<=0:
            desired_channels=floors.copy()
        if desired_channels.sum()>0:
            desired_channels=desired_channels/desired_channels.sum()*max(a_total-desired_base,0)
        target=np.r_[desired_channels,desired_base]
        lower=np.r_[floors,0.]
        weights=np.r_[np.ones(len(names))*float(settings['channel_signal_weight']),float(settings['baseline_weight'])]
        x0=lower.copy()
        x0[-1]+=a_total-lower.sum()
        def objective(x): return float(np.sum(weights*(x-target)**2))
        result=minimize(objective,x0,method='SLSQP',bounds=[(float(v),None) for v in lower],constraints=[{'type':'eq','fun':lambda x:np.sum(x)-a_total}],options={'maxiter':2000,'ftol':1e-9})
        if not result.success or abs(result.x.sum()-a_total)>1e-4 or (result.x<lower-1e-4).any():
            raise RuntimeError(f'QS allocation failed for {market_month}: {result.message}')
        for i,(_,r) in enumerate(sub.iterrows()):
            record={k:v for k,v in zip(GROUP+['month'],market_month)}
            record.update(channel=r['channel'],spend=float(r['spend']),attributed_qs=r['attributed_qs'],last_click_qs=r['last_click_qs'],floor=float(floors[i]),modelled_qs=float(result.x[i]),actual_qs=a_total,baseline_qs=float(result.x[-1]),raw_model_signal=float(prior[i]))
            records.append(record)
    result=pd.DataFrame(records)
    if not result.empty:
        result['platform_cpqs']=result['spend']/result['attributed_qs'].replace(0,np.nan)
        result['modelled_cpqs']=result['spend']/result['modelled_qs'].replace(0,np.nan)
    return result
