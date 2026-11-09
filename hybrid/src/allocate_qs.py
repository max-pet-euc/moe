import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .periods import frequency, period_start
from .load_data import MARKET, KEYS


def _sum_nullable(x): return x.sum(min_count=1)

def allocate_periods(channels,constraints,targets,fit_daily,settings):
    """Fit a constrained allocation per period, then distribute each channel across days.

    Daily channel contributions reconcile to daily actual QS, while hard floors are
    enforced over the configured constraint period. No causal interpretation.
    """
    cf=frequency(settings,'constraints')
    ch=channels.merge(constraints[['channel','enabled','minimum_source','minimum_multiplier','constraint_type']],on='channel',how='inner')
    ch=ch.loc[ch['enabled']].copy()
    ch['period_start']=period_start(ch['date_day'],cf)
    hard=ch['constraint_type'].eq('hard') & ch['minimum_source'].eq('minimum_qs')
    ch['floor']=np.where(hard,ch['last_click_qs'].fillna(0)*ch['minimum_multiplier'],0.)
    t=targets.copy();t['period_start']=period_start(t['date_day'],cf)
    f=fit_daily.copy()
    # Raw model signal by channel/day: sum both exposure and attribution components.
    contrib=[c for c in f if c.startswith('contribution__')]
    signal=[]
    for col in contrib:
        name=col.split('__')[-1]
        signal.append(f[KEYS+[col]].rename(columns={col:'raw_signal'}).assign(channel=name))
    if signal:
        sig=pd.concat(signal,ignore_index=True).groupby(KEYS+['channel'],as_index=False)['raw_signal'].sum()
    else:
        sig=pd.DataFrame(columns=KEYS+['channel','raw_signal'])
    ch=ch.merge(sig,on=KEYS+['channel'],how='left')
    ch['raw_signal']=ch['raw_signal'].fillna(0).clip(lower=0)
    records=[]
    for market_period, sub in ch.groupby(MARKET+['period_start'],dropna=False):
        country,brand,start=market_period
        actual=t.loc[(t['country_code']==country)&(t['brand']==brand)&(t['period_start']==start),['date_day','actual_qs']].copy()
        if actual.empty: continue
        actual=actual.sort_values('date_day')
        days=actual['date_day'].tolist()
        actual_by_day=actual.set_index('date_day')['actual_qs'].to_dict()
        names=sorted(sub['channel'].unique())
        n=len(names); d=len(days)
        floor=sub.groupby('channel')['floor'].sum().reindex(names,fill_value=0).to_numpy(float)
        total=float(actual['actual_qs'].sum())
        if floor.sum()>total+1e-6: raise ValueError(f'infeasible hard floors for {market_period}: {floor.sum()} > {total}')
        # Daily prior from fitted model. Allocate a modest baseline prior.
        prior=sub.pivot_table(index='date_day',columns='channel',values='raw_signal',aggfunc='sum').reindex(index=days,columns=names).fillna(0).to_numpy(float)
        actual_vec=actual['actual_qs'].to_numpy(float)
        baseline_share=float(settings.get('baseline_prior_share',0.15))
        desired=np.zeros((d,n+1))
        for j,a in enumerate(actual_vec):
            s=prior[j].sum()
            if s>0: desired[j,:n]=prior[j]/s*a*(1-baseline_share)
            desired[j,n]=a-desired[j,:n].sum()
        # Solve period-level constrained allocation. Daily rows must each sum to actual QS.
        # For speed and stability, optimize period channel totals, then distribute daily.
        desired_totals=desired.sum(axis=0)
        lower=np.r_[floor,0.]
        weights=np.r_[np.ones(n)*float(settings.get('channel_signal_weight',1)),float(settings.get('baseline_weight',0.5))]
        x0=lower.copy();x0[-1]+=total-lower.sum()
        def obj(x):return float(np.sum(weights*(x-desired_totals)**2))
        result=minimize(obj,x0,method='SLSQP',bounds=[(float(v),None) for v in lower],constraints=[{'type':'eq','fun':lambda x:float(np.sum(x)-total)}],options={'maxiter':1500,'ftol':1e-8})
        if not result.success or np.any(result.x<lower-1e-4) or abs(result.x.sum()-total)>1e-4:
            raise RuntimeError(f'QS allocation failed for {market_period}: {result.message}')
        # Distribute period totals to days with iterative proportional fitting.
        # A tiny positive prior allows floors to be distributed across all observed days.
        matrix=desired+1e-7
        col_target=np.maximum(result.x,0)
        for _ in range(1500):
            col_sums=matrix.sum(axis=0)
            matrix*=np.divide(col_target,col_sums,out=np.zeros_like(col_target),where=col_sums>0)[None,:]
            row_sums=matrix.sum(axis=1)
            matrix*=np.divide(actual_vec,row_sums,out=np.zeros_like(actual_vec),where=row_sums>0)[:,None]
            if np.max(np.abs(matrix.sum(axis=0)-col_target))<1e-5 and np.max(np.abs(matrix.sum(axis=1)-actual_vec))<1e-5: break
        if np.max(np.abs(matrix.sum(axis=0)-col_target))>1e-3:
            # More robust fallback: constrained joint optimisation for rare difficult periods.
            flat=matrix.ravel()
            row_constraints=[{'type':'eq','fun':lambda x,j=j:np.sum(x.reshape(d,n+1)[j,:])-actual_vec[j]} for j in range(d)]
            col_constraints=[{'type':'eq','fun':lambda x,i=i:np.sum(x.reshape(d,n+1)[:,i])-col_target[i]} for i in range(n)]
            opt=minimize(lambda x:float(np.sum((x-flat)**2)),flat,method='SLSQP',bounds=[(0,None)]*len(flat),constraints=row_constraints+col_constraints,options={'maxiter':1500,'ftol':1e-7})
            if not opt.success: raise RuntimeError(f'daily reconciliation failed for {market_period}: {opt.message}')
            matrix=opt.x.reshape(d,n+1)
        lookup=sub.set_index(['date_day','channel'])
        for j,day in enumerate(days):
            for i,name in enumerate(names):
                if (day,name) not in lookup.index: continue
                row=lookup.loc[(day,name)]
                if isinstance(row,pd.DataFrame): row=row.iloc[0]
                records.append({'date_day':day,'country_code':country,'brand':brand,'constraint_period_start':start,
                    'channel':name,'spend':row['spend'],'attributed_qs':row['attributed_qs'],
                    'last_click_qs':row['last_click_qs'],'floor':row['floor'],
                    'modelled_qs':float(matrix[j,i]),'actual_qs':float(actual_vec[j]),
                    'baseline_qs':float(matrix[j,-1]),'raw_model_signal':float(row['raw_signal'])})
    out=pd.DataFrame(records)
    if not out.empty:
        out['platform_cpqs']=out['spend']/out['attributed_qs'].replace(0,np.nan)
        out['modelled_cpqs']=out['spend']/out['modelled_qs'].replace(0,np.nan)
    return out

def aggregate_reporting(daily,settings):
    if daily.empty: return daily
    freq=frequency(settings,'reporting')
    df=daily.copy();df['period_start']=period_start(df['date_day'],freq)
    keys=MARKET+['period_start']
    # Daily actual/baseline appear on every channel row; deduplicate before sum.
    day_actual=df[KEYS+['actual_qs','baseline_qs']].drop_duplicates(KEYS)
    day_actual['period_start']=period_start(day_actual['date_day'],freq)
    totals=day_actual.groupby(keys,as_index=False).agg(actual_qs=('actual_qs','sum'),baseline_qs=('baseline_qs','sum'))
    grouped=df.groupby(keys+['channel'],as_index=False).agg(
        spend=('spend',_sum_nullable),attributed_qs=('attributed_qs',_sum_nullable),
        last_click_qs=('last_click_qs',_sum_nullable),floor=('floor','sum'),
        modelled_qs=('modelled_qs','sum'),raw_model_signal=('raw_model_signal','sum'))
    grouped=grouped.merge(totals,on=keys,how='left')
    grouped['platform_cpqs']=grouped['spend']/grouped['attributed_qs'].replace(0,np.nan)
    grouped['modelled_cpqs']=grouped['spend']/grouped['modelled_qs'].replace(0,np.nan)
    grouped['period']=grouped['period_start'].dt.strftime('%Y-%m-%d')
    return grouped
