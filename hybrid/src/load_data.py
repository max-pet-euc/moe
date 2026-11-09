from pathlib import Path
import json
import pandas as pd

KEYS = ['date_day', 'country_code', 'brand', 'problem_type']
METRICS = ['spend', 'impressions', 'attributed_qs', 'last_click_qs']


def load_settings(root: Path):
    return json.loads((root / 'hybrid/config/settings.json').read_text())


def load_inputs(root: Path, settings: dict):
    channels = pd.read_csv(root / settings['channel_file'])
    constraints = pd.read_csv(root / settings['constraint_file'])
    targets = pd.read_csv(root / settings['target_file'])
    for col in KEYS:
        if col not in channels:
            raise ValueError(f'channel file missing required column: {col}')
    if 'channel' not in channels:
        raise ValueError('channel file missing required column: channel')
    for col in METRICS:
        if col not in channels:
            channels[col] = float('nan')
        channels[col] = pd.to_numeric(channels[col], errors='coerce')
    channels['date_day'] = pd.to_datetime(channels['date_day'], errors='raise').dt.normalize()
    for col in ['country_code', 'brand', 'problem_type', 'channel']:
        channels[col] = channels[col].astype(str).str.strip().str.lower()
    constraints['channel'] = constraints['channel'].astype(str).str.strip().str.lower()
    defaults = {'exposure_metric': 'impressions', 'minimum_source': 'last_click_qs',
                'minimum_multiplier': 1.0, 'constraint_type': 'hard', 'enabled': True,
                'model_group': None, 'constraint_period': 'monthly', 'constraint_weight': 10.0}
    for col, default in defaults.items():
        if col not in constraints:
            constraints[col] = default
        else:
            constraints[col] = constraints[col].fillna(default) if default is not None else constraints[col]
    constraints['enabled'] = constraints['enabled'].astype(str).str.lower().isin(['true','1','yes','y'])
    for col in ['minimum_multiplier', 'constraint_weight']:
        constraints[col] = pd.to_numeric(constraints[col], errors='raise')
    date_col = settings['target_date_column']
    qs_col = settings['target_qs_column']
    for col in [date_col, qs_col]:
        if col not in targets:
            raise ValueError(f'target file missing {col!r}; update hybrid/config/settings.json')
    targets = targets.rename(columns={date_col: 'date_day', qs_col: 'actual_qs'})
    targets['date_day'] = pd.to_datetime(targets['date_day'], errors='raise').dt.normalize()
    targets['actual_qs'] = pd.to_numeric(targets['actual_qs'], errors='raise')
    for col in ['country_code','brand','problem_type']:
        if col not in targets:
            raise ValueError(f'target file missing {col!r}; supply a daily market-level QS target')
        targets[col] = targets[col].astype(str).str.strip().str.lower()
    for key, value in settings.get('target_filters', {}).items():
        if key not in targets:
            raise ValueError(f'unknown target filter column {key}')
        targets = targets.loc[targets[key].astype(str).str.lower() == str(value).lower()]
    targets = targets[KEYS + ['actual_qs']].groupby(KEYS, as_index=False)['actual_qs'].sum()
    return channels, constraints, targets
