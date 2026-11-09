import numpy as np
import pandas as pd
from scipy.optimize import linprog
from .periods import frequency, period_start
from .load_data import MARKET, KEYS


def _sum_nullable(x): return x.sum(min_count=1)

def _weighted_projection(desired, lower, weights, total):
    """Exact weighted least-squares projection onto a simplex with lower bounds.

    Minimises sum(weights * (x - desired)^2), with x >= lower and sum(x)=total.
    Bisection on the KKT multiplier avoids fragile SLSQP line searches.
    """
    desired=np.asarray(desired,dtype=float)
    lower=np.asarray(lower,dtype=float)
    weights=np.asarray(weights,dtype=float)
    if not (np.isfinite(desired).all() and np.isfinite(lower).all() and np.isfinite(weights).all()):
        raise ValueError('non-finite inputs to QS allocation')
    if (weights<=0).any() or (lower<0).any() or total<0:
        raise ValueError('invalid allocation weights, floors or total')
    if lower.sum()>total+1e-6:
        raise ValueError(f'infeasible hard floors: {lower.sum()} > {total}')
    if total-lower.sum()<=1e-9:
        return lower.copy()
    # x_i = max(lower_i, desired_i - lambda/(2*weight_i))
    def at(lam):
        return np.maximum(lower,desired-lam/(2*weights))
    lo=-1.0
    hi=1.0
    for _ in range(200):
        if at(lo).sum()>=total:break
        lo*=2
    for _ in range(200):
        if at(hi).sum()<=total:break
        hi*=2
    for _ in range(160):
        mid=(lo+hi)/2
        if at(mid).sum()>total:lo=mid
        else:hi=mid
    x=at((lo+hi)/2)
    # Correct floating point residual without breaching floors.
    residual=total-x.sum()
    if residual>0:
        x[np.argmax(weights)]+=residual
    elif residual<0:
        slack=x-lower
        idx=np.argmax(slack)
        if slack[idx]>=-residual-1e-7:x[idx]+=residual
    if abs(x.sum()-total)>1e-5 or (x<lower-1e-5).any():
        raise RuntimeError('numerical reconciliation failed in weighted projection')
    return x


def _daily_transport(desired, row_totals, column_totals):
    """Reconcile channel-period totals to daily actual QS with nonnegative entries."""
    d,k=desired.shape
    if not np.isclose(row_totals.sum(),column_totals.sum(),atol=1e-5):
        raise ValueError('daily and channel totals do not reconcile')
    if row_totals.sum()<=1e-12:return np.zeros_like(desired)
    matrix=np.maximum(desired,0)+1e-6
    # Sinkhorn / iterative proportional fitting; positive seed supports all cells.
    for _ in range(3000):
        col_sum=matrix.sum(axis=0)
        matrix*=np.divide(column_totals,col_sum,out=np.zeros_like(column_totals),where=col_sum>0)[None,:]
        row_sum=matrix.sum(axis=1)
        matrix*=np.divide(row_totals,row_sum,out=np.zeros_like(row_totals),where=row_sum>0)[:,None]
        if (np.max(np.abs(matrix.sum(axis=0)-column_totals))<1e-5 and
                np.max(np.abs(matrix.sum(axis=1)-row_totals))<1e-5):
            return matrix
    # Feasible transportation LP fallback (no nonlinear optimisation).
    from scipy.sparse import lil_matrix
    A=lil_matrix((d+k,d*k),dtype=float)
    for i in range(d):A[i,i*k:(i+1)*k]=1
    for j in range(k):A[d+j,j::k]=1
    b=np.r_[row_totals,column_totals]
    res=linprog(np.zeros(d*k),A_eq=A.tocsr(),b_eq=b,bounds=(0,None),method='highs')
    if not res.success:
        raise RuntimeError(f'daily transportation reconciliation failed: {res.message}')
    return res.x.reshape(d,k)


def allocate_periods(channels,constraints,targets,fit_daily,settings):
    """Fit a constrained allocation per period, then distribute each channel across days.

    Daily channel contributions reconcile to daily actual QS, while hard floors are
    enforced over the configured constraint period. No causal interpretation.
    """
    cf=frequency(settings,'constraints')
    ch=channels.merge(constraints[['channel','enabled','minimum_source','minimum_multiplier','constraint_type']],on='channel',how='inner')
    ch=ch.loc[ch['enabled']].copy()
    ch['period_start']=period_start(ch['date_day'],cf)
    hard=ch['constraint_type'].eq('hard') & ch['minimum_source'].ne('none')
    tolerance=float(settings.get('minimum_qs_tolerance',0.10))
    if not 0 <= tolerance < 1: raise ValueError('minimum_qs_tolerance must be in [0,1)')
    ch['floor']=np.where(hard,ch['minimum_qs'].fillna(0)*ch['minimum_multiplier']*(1-tolerance),0.)
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
        col_target=_weighted_projection(desired_totals,lower,weights,total)
        matrix=_daily_transport(desired,actual_vec,col_target)
        if np.max(np.abs(matrix.sum(axis=0)-col_target))>1e-3 or np.max(np.abs(matrix.sum(axis=1)-actual_vec))>1e-3:
            raise RuntimeError(f'daily reconciliation tolerance exceeded for {market_period}')
        lookup=sub.set_index(['date_day','channel'])
        for j,day in enumerate(days):
            for i,name in enumerate(names):
                if (day,name) not in lookup.index: continue
                row=lookup.loc[(day,name)]
                if isinstance(row,pd.DataFrame): row=row.iloc[0]
                records.append({'date_day':day,'country_code':country,'brand':brand,'constraint_period_start':start,
                    'channel':name,'spend':row['spend'],'attributed_qs':row['attributed_qs'],
                    'minimum_qs':row['minimum_qs'],'minimum_qs_source':row['minimum_qs_source'],'floor':row['floor'],
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
        minimum_qs=('minimum_qs',_sum_nullable),minimum_qs_source=('minimum_qs_source',lambda x: ', '.join(sorted(x.dropna().astype(str).unique()))),floor=('floor','sum'),
        modelled_qs=('modelled_qs','sum'),raw_model_signal=('raw_model_signal','sum'))
    grouped=grouped.merge(totals,on=keys,how='left')
    grouped['platform_cpqs']=grouped['spend']/grouped['attributed_qs'].replace(0,np.nan)
    grouped['modelled_cpqs']=grouped['spend']/grouped['modelled_qs'].replace(0,np.nan)
    grouped['period']=grouped['period_start'].dt.strftime('%Y-%m-%d')
    return grouped
