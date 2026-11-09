from pathlib import Path
import json
import pandas as pd

# Source files have no problem_type dimension; never manufacture one.
KEYS = ['date_day', 'country_code', 'brand']
MARKET = ['country_code', 'brand']
METRICS = ['spend', 'impressions', 'attributed_qs', 'minimum_qs']
NA = ['', '-', '—', 'null', 'none', 'nan', 'n/a']


def load_settings(root: Path):
    return json.loads((root / 'hybrid/config/settings.json').read_text())


def load_inputs(root: Path, settings: dict):
    def read(key):
        path = root / settings[key]
        if not path.is_file():
            raise FileNotFoundError(f'Missing {key}: {path}. Check hybrid/config/settings.json')
        return pd.read_csv(path, na_values=NA, keep_default_na=True, low_memory=False)

    channels, constraints, targets = read('channel_file'), read('constraint_file'), read('target_file')
    for col in KEYS + ['channel', 'spend_local', 'impressions', 'attributed_qs', 'minimum_qs']:
        if col not in channels:
            raise ValueError(f'data_channel.csv missing column: {col}')
    channels = channels.rename(columns={'spend_local': 'spend'})
    channels['date_day'] = pd.to_datetime(channels['date_day'], errors='raise').dt.normalize()
    for col in MARKET + ['channel']:
        channels[col] = channels[col].astype('string').str.strip().str.lower().replace('', pd.NA)
    for col in METRICS:
        channels[col] = pd.to_numeric(channels[col], errors='coerce')
    if 'minimum_qs_source' not in channels:
        raise ValueError('data_channel.csv missing minimum_qs_source')
    channels['minimum_qs_source'] = channels['minimum_qs_source'].astype('string').str.strip().str.lower().replace({'-':pd.NA, '':pd.NA})

    if 'channel' not in constraints:
        raise ValueError('data_constraints.csv missing channel column')
    constraints['channel'] = constraints['channel'].astype('string').str.strip().str.lower()
    defaults = {'exposure_metric': 'impressions', 'minimum_source': 'minimum_qs',
                'minimum_multiplier': 1.0, 'constraint_type': 'hard', 'enabled': True,
                'model_group': None, 'constraint_period': 'month', 'constraint_weight': 10.0}
    for col, default in defaults.items():
        if col not in constraints:
            constraints[col] = default
        elif default is not None:
            constraints[col] = constraints[col].fillna(default)
    constraints['enabled'] = constraints['enabled'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
    constraints['constraint_period'] = constraints['constraint_period'].astype(str).str.lower().replace({'monthly': 'month'})
    constraints['minimum_source'] = constraints['minimum_source'].astype(str).str.lower().replace({'last_click_qs': 'minimum_qs'})
    constraints['minimum_source_label'] = constraints['minimum_source']
    for col in ['minimum_multiplier', 'constraint_weight']:
        constraints[col] = pd.to_numeric(constraints[col], errors='raise')

    date_col, qs_col = settings['target_date_column'], settings['target_qs_column']
    for col in [date_col, qs_col] + MARKET:
        if col not in targets:
            raise ValueError(f'data_funnel.csv missing column: {col}')
    targets = targets.rename(columns={date_col: 'date_day', qs_col: 'actual_qs'})
    targets['date_day'] = pd.to_datetime(targets['date_day'], errors='raise').dt.normalize()
    for col in MARKET:
        targets[col] = targets[col].astype('string').str.strip().str.lower()
    for col in ['actual_qs', 'cp', 'op']:
        if col not in targets:
            targets[col] = float('nan')
        targets[col] = pd.to_numeric(targets[col], errors='coerce')
    for key, value in settings.get('target_filters', {}).items():
        if key not in targets:
            raise ValueError(f'unknown target filter column: {key}')
        targets = targets.loc[targets[key].astype(str).str.lower() == str(value).lower()]
    targets = targets.groupby(KEYS, as_index=False).agg(
        actual_qs=('actual_qs', lambda x: x.sum(min_count=1)),
        actual_cp=('cp', lambda x: x.sum(min_count=1)),
        actual_op=('op', lambda x: x.sum(min_count=1)))
    return channels, constraints, targets
