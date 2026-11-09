import pandas as pd
from .load_data import KEYS, MARKET

def filter_daily_coverage(channels, targets):
    """Exclude channel rows without a daily funnel target; never invent actual QS."""
    target_keys = targets[KEYS].drop_duplicates().assign(_has_actual=True)
    tagged = channels.merge(target_keys, on=KEYS, how='left', validate='many_to_one')
    missing = tagged['_has_actual'].isna()
    excluded = tagged.loc[missing].drop(columns='_has_actual').copy()
    included = tagged.loc[~missing].drop(columns='_has_actual').copy()
    audit = excluded.groupby(MARKET+['channel'], dropna=False).agg(
        rows=('date_day','size'),
        spend=('spend',lambda x: x.sum(min_count=1)),
        first_date=('date_day','min'),last_date=('date_day','max')
    ).reset_index()
    audit.insert(0,'reason','missing_daily_actual_qs')
    valid_dates = included[KEYS].drop_duplicates()
    # Do not include target-only dates with no eligible channel rows in fitting.
    target_matched = targets.merge(valid_dates,on=KEYS,how='inner')
    return included,target_matched,audit

def coverage_summary(channels, targets, excluded):
    included = channels.groupby(MARKET,dropna=False).agg(
        included_rows=('date_day','size'), included_days=('date_day','nunique'),
        included_spend=('spend',lambda x:x.sum(min_count=1))
    ).reset_index()
    if excluded.empty:
        included['excluded_rows']=0
        included['excluded_spend']=0.
        return included
    ex=excluded.groupby(MARKET,dropna=False).agg(excluded_rows=('rows','sum'),excluded_spend=('spend','sum')).reset_index()
    return included.merge(ex,on=MARKET,how='outer').fillna({'included_rows':0,'included_days':0,'included_spend':0,'excluded_rows':0,'excluded_spend':0})
