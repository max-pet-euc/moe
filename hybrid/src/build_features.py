import numpy as np
import pandas as pd
from .load_data import KEYS

def build_features(channels,constraints,targets,settings=None):
    enabled=constraints.loc[constraints['enabled'],['channel','exposure_metric']]
    data=channels.merge(enabled,on='channel',how='inner')
    data['exposure']=np.where(data['exposure_metric'].eq('spend'),data['spend'],data['impressions'])
    data['exposure']=pd.to_numeric(data['exposure'],errors='coerce')
    data['exposure_missing']=data['exposure'].isna().astype(float)
    data['attributed_missing']=data['attributed_qs'].isna().astype(float)
    data['exposure_signal']=np.log1p(data['exposure'].fillna(0).clip(lower=0))
    data['attributed_signal']=np.log1p(data['attributed_qs'].fillna(0).clip(lower=0))
    features=targets[KEYS].drop_duplicates().copy()
    for signal in ['exposure_signal','attributed_signal','exposure_missing','attributed_missing']:
        pivot=data.pivot_table(index=KEYS,columns='channel',values=signal,aggfunc='sum')
        pivot.columns=[f'{signal}__{c}' for c in pivot.columns]
        features=features.merge(pivot.reset_index(),on=KEYS,how='left')
    cols=[c for c in features if '__' in c]
    features[cols]=features[cols].fillna(0)
    features['dow']=features['date_day'].dt.dayofweek
    features['trend']=(features['date_day']-features['date_day'].min()).dt.days
    for k in range(1,7): features[f'dow_{k}']=(features['dow']==k).astype(float)
    features['year_sin']=np.sin(2*np.pi*features['date_day'].dt.dayofyear/365.25)
    features['year_cos']=np.cos(2*np.pi*features['date_day'].dt.dayofyear/365.25)
    return features.merge(targets,on=KEYS,how='inner'),data
