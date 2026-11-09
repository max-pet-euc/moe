import numpy as np
import pandas as pd
from .load_data import KEYS, METRICS


def validate(channels, constraints, targets):
    errors, warnings = [], []
    key = KEYS + ['channel']
    if channels.duplicated(key).any():
        errors.append(f'duplicate channel daily keys: {int(channels.duplicated(key).sum())}')
    if constraints['channel'].duplicated().any():
        errors.append('duplicate channel names in constraints')
    missing = set(channels['channel']) - set(constraints['channel'])
    if missing:
        errors.append(f'channels missing constraints: {sorted(missing)}')
    for m in METRICS:
        if (channels[m].dropna() < 0).any():
            errors.append(f'negative values in {m}')
    if (targets['actual_qs'] < 0).any():
        errors.append('negative actual QS')
    if targets.duplicated(KEYS).any():
        errors.append('duplicate target keys')
    if not set(constraints['exposure_metric']).issubset({'impressions','spend'}):
        errors.append('exposure_metric must be impressions or spend')
    if not set(constraints['minimum_source']).issubset({'last_click_qs','none'}):
        errors.append('minimum_source must be last_click_qs or none')
    if not set(constraints['constraint_type']).issubset({'hard','none'}):
        errors.append('constraint_type must be hard or none')
    if not set(constraints['constraint_period']).issubset({'monthly'}):
        errors.append('only monthly constraint_period supported')
    if (constraints['minimum_multiplier'] < 0).any():
        errors.append('negative minimum_multiplier')
    active = constraints.loc[constraints['enabled']].copy()
    c = channels.merge(active[['channel','minimum_source','minimum_multiplier','constraint_type']], on='channel', how='inner')
    if c.empty:
        errors.append('no enabled channels with matching data')
    if c['last_click_qs'].isna().any() and (c['constraint_type'] == 'hard').any():
        warnings.append('missing last_click_qs treated as zero for floor calculations; verify source completeness')
    c['floor'] = np.where((c['constraint_type'] == 'hard') & (c['minimum_source'] == 'last_click_qs'), c['last_click_qs'].fillna(0) * c['minimum_multiplier'], 0.0)
    c['month'] = c['date_day'].dt.to_period('M')
    t = targets.copy()
    t['month'] = t['date_day'].dt.to_period('M')
    floors = c.groupby(['country_code','brand','problem_type','month'], as_index=False)['floor'].sum()
    actual = t.groupby(['country_code','brand','problem_type','month'], as_index=False)['actual_qs'].sum()
    check = floors.merge(actual, on=['country_code','brand','problem_type','month'], how='left')
    missing_targets = check['actual_qs'].isna()
    if missing_targets.any():
        errors.append(f'{int(missing_targets.sum())} channel market-months lack actual QS targets')
    impossible = check.loc[check['floor'] > check['actual_qs'] + 1e-6]
    if not impossible.empty:
        examples = impossible.head(3).to_dict('records')
        errors.append(f'monthly last-click floors exceed actual QS in {len(impossible)} market-months: {examples}')
    if channels['attributed_qs'].isna().any():
        warnings.append('some attributed_qs values are missing; kept distinct from observed zero')
    if channels['impressions'].isna().any():
        warnings.append('some impressions are missing; exposure may use spend if configured')
    return errors, warnings, check
