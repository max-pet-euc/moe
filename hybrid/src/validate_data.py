import numpy as np
import pandas as pd
from .load_data import KEYS, MARKET, METRICS


def prepare(channels, constraints, targets, settings):
    """Exclude unconfigured/unmatched data explicitly and return audit rows."""
    audit = []
    valid = set(constraints.loc[constraints['enabled'], 'channel'].dropna())
    bad = channels['channel'].isna() | ~channels['channel'].isin(valid)
    excluded = channels.loc[bad].copy()
    for (country, brand, channel), part in excluded.groupby(['country_code', 'brand', 'channel'], dropna=False):
        audit.append({'reason': 'missing_or_disabled_channel_mapping', 'country_code': country,
                      'brand': brand, 'channel': str(channel), 'rows': len(part),
                      'spend': part['spend'].sum(min_count=1)})
    channels = channels.loc[~bad].copy()
    target_markets = targets[MARKET].drop_duplicates()
    tagged = channels.merge(target_markets.assign(_target_present=True), on=MARKET, how='left')
    no_target = tagged['_target_present'].isna()
    for (country, brand), part in tagged.loc[no_target].groupby(MARKET, dropna=False):
        audit.append({'reason': 'no_matching_funnel_target', 'country_code': country,
                      'brand': brand, 'channel': '(all)', 'rows': len(part),
                      'spend': part['spend'].sum(min_count=1)})
    channels = tagged.loc[~no_target].drop(columns=['_target_present']).copy()
    # Only model markets for which we have eligible channel data.
    targets = targets.merge(channels[MARKET].drop_duplicates(), on=MARKET, how='inner')
    return channels, targets, pd.DataFrame(audit, columns=['reason', 'country_code', 'brand', 'channel', 'rows', 'spend'])


def validate(channels, constraints, targets):
    errors, warnings = [], []
    key = KEYS + ['channel']
    if channels.duplicated(key).any():
        errors.append(f'duplicate channel daily keys: {int(channels.duplicated(key).sum())}')
    if constraints['channel'].duplicated().any():
        errors.append('duplicate channel names in constraints')
    for m in METRICS:
        if (channels[m].dropna() < 0).any():
            errors.append(f'negative values in {m}')
    if targets['actual_qs'].isna().any() or (targets['actual_qs'] < 0).any():
        errors.append('missing or negative actual QS in funnel')
    if not set(constraints['exposure_metric']).issubset({'impressions', 'spend'}):
        errors.append('exposure_metric must be impressions or spend')
    if not set(constraints['minimum_source']).issubset({'minimum_qs', 'none'}):
        errors.append('minimum_source must be minimum_qs or none')
    if not set(constraints['constraint_type']).issubset({'hard', 'none'}):
        errors.append('constraint_type must be hard or none')
    if not set(constraints['constraint_period']).issubset({'month'}):
        errors.append('only month constraint_period supported')
    if (constraints['minimum_multiplier'] < 0).any():
        errors.append('negative minimum_multiplier')
    if channels.empty or targets.empty:
        errors.append('no eligible channel/target data after mapping')
        return errors, warnings, pd.DataFrame()
    c = channels.merge(constraints.loc[constraints['enabled'], ['channel','minimum_source','minimum_multiplier','constraint_type']], on='channel', how='inner')
    missing_min = c['last_click_qs'].isna() & c['minimum_source'].eq('minimum_qs') & c['constraint_type'].eq('hard')
    if missing_min.any():
        errors.append(f'{int(missing_min.sum())} hard minimum_qs values missing; cannot treat missing as zero')
    c['floor'] = np.where((c['constraint_type'] == 'hard') & (c['minimum_source'] == 'minimum_qs'), c['last_click_qs'].fillna(0) * c['minimum_multiplier'], 0.0)
    c['month'] = c['date_day'].dt.to_period('M').astype(str)
    t = targets.copy(); t['month'] = t['date_day'].dt.to_period('M').astype(str)
    floors = c.groupby(MARKET + ['month'], as_index=False)['floor'].sum()
    actual = t.groupby(MARKET + ['month'], as_index=False)['actual_qs'].sum()
    check = floors.merge(actual, on=MARKET + ['month'], how='left')
    if check['actual_qs'].isna().any():
        errors.append(f'{int(check["actual_qs"].isna().sum())} channel market-months have no matching actual QS')
    impossible = check.loc[check['floor'] > check['actual_qs'] + 1e-6]
    if not impossible.empty:
        errors.append(f'monthly minimum QS exceeds actual QS in {len(impossible)} market-months: {impossible.head(3).to_dict("records")}')
    if channels['attributed_qs'].isna().any():
        warnings.append('attributed_qs has nulls (not interpreted as observed zero)')
    if channels['impressions'].isna().any():
        warnings.append('impressions has nulls; check exposure_metric configuration')
    if channels['spend'].isna().any():
        warnings.append('spend has nulls; reported cost metrics may be incomplete')
    return errors, warnings, check
