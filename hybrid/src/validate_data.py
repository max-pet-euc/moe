import numpy as np
import pandas as pd
from .load_data import KEYS, MARKET, METRICS
from .periods import frequency, period_start
from .coverage_report import filter_daily_coverage

AUDIT_COLS=['reason','country_code','brand','channel','rows','spend','first_date','last_date']

def prepare(channels, constraints, targets, settings):
    valid=set(constraints.loc[constraints['enabled'],'channel'].dropna())
    bad=channels['channel'].isna() | ~channels['channel'].isin(valid)
    excluded=channels.loc[bad].copy()
    audit=excluded.groupby(MARKET+['channel'],dropna=False).agg(
        rows=('date_day','size'),spend=('spend',lambda x:x.sum(min_count=1)),
        first_date=('date_day','min'),last_date=('date_day','max')).reset_index()
    audit.insert(0,'reason','missing_or_disabled_channel_mapping')
    channels=channels.loc[~bad].copy()
    channels,targets,daily_audit=filter_daily_coverage(channels,targets)
    audit=pd.concat([audit,daily_audit],ignore_index=True).reindex(columns=AUDIT_COLS)
    return channels,targets,audit

def validate(channels,constraints,targets,settings):
    errors,warnings=[],[]
    cf=frequency(settings,'constraints')
    frequency(settings,'reporting')
    if channels.duplicated(KEYS+['channel']).any():
        errors.append(f'duplicate channel daily keys: {int(channels.duplicated(KEYS+["channel"]).sum())}')
    if constraints['channel'].duplicated().any(): errors.append('duplicate channel names in constraints')
    for m in METRICS:
        if (channels[m].dropna()<0).any(): errors.append(f'negative values in {m}')
    if targets['actual_qs'].isna().any() or (targets['actual_qs']<0).any():
        errors.append('missing or negative actual QS in funnel')
    if not set(constraints['exposure_metric'].dropna()).issubset({'impressions','spend'}): errors.append('exposure_metric must be impressions or spend')
    if not set(constraints['minimum_source'].dropna()).issubset({'minimum_qs','none'}): errors.append('minimum_source must be minimum_qs or none')
    if not set(constraints['constraint_type'].dropna()).issubset({'hard','none'}): errors.append('constraint_type must be hard or none')
    if (constraints['minimum_multiplier']<0).any(): errors.append('negative minimum_multiplier')
    if channels.empty or targets.empty:
        errors.append('no eligible channel/target daily data after mapping and coverage')
        return errors,warnings,pd.DataFrame(columns=MARKET+['period_start','floor','actual_qs'])
    c=channels.merge(constraints.loc[constraints['enabled'],['channel','minimum_source','minimum_multiplier','constraint_type']],on='channel',how='inner')
    hard=c['constraint_type'].eq('hard') & c['minimum_source'].eq('minimum_qs')
    missing=c['last_click_qs'].isna() & hard
    if missing.any(): errors.append(f'{int(missing.sum())} hard minimum_qs values missing; cannot treat missing as zero')
    c['floor']=np.where(hard,c['last_click_qs'].fillna(0)*c['minimum_multiplier'],0.)
    c['period_start']=period_start(c['date_day'],cf)
    t=targets.copy();t['period_start']=period_start(t['date_day'],cf)
    floors=c.groupby(MARKET+['period_start'],as_index=False)['floor'].sum()
    actual=t.groupby(MARKET+['period_start'],as_index=False)['actual_qs'].sum()
    checks=floors.merge(actual,on=MARKET+['period_start'],how='left',validate='one_to_one')
    if checks['actual_qs'].isna().any(): errors.append('missing actual QS for a retained constraint period')
    impossible=checks.loc[checks['floor']>checks['actual_qs']+1e-6]
    if len(impossible): errors.append(f'{len(impossible)} {cf} periods have hard floors exceeding actual QS: {impossible.head(3).to_dict("records")}')
    for metric in ['attributed_qs','impressions','spend']:
        if channels[metric].isna().any(): warnings.append(f'{metric} contains nulls; null is not observed zero')
    return errors,warnings,checks
