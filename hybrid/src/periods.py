import pandas as pd

def frequency(settings, kind):
    obj = settings.get(kind, {})
    value = obj.get('frequency', 'weekly') if isinstance(obj, dict) else 'weekly'
    value = str(value).lower()
    value = {'day':'daily','week':'weekly','month':'monthly','d':'daily','w':'weekly','ms':'monthly'}.get(value,value)
    if value not in ('daily','weekly','monthly'):
        raise ValueError(f'{kind}.frequency must be daily, weekly or monthly; got {value}')
    return value

def period_start(dates, freq):
    dates = pd.to_datetime(dates)
    if freq == 'daily': return dates.dt.normalize()
    if freq == 'weekly': return (dates - pd.to_timedelta(dates.dt.dayofweek,unit='D')).dt.normalize()
    if freq == 'monthly': return dates.dt.to_period('M').dt.to_timestamp()
    raise ValueError(f'Unknown frequency {freq}')
