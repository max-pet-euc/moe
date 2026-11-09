import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def fit_markets(features, settings):
    rows, coefficients, statuses = [], [], []
    group_cols = ['country_code','brand']
    for market, frame in features.groupby(group_cols, dropna=False):
        frame = frame.sort_values('date_day').copy()
        channel_cols = sorted(c for c in frame if c.startswith(('exposure_signal__','attributed_signal__')))
        seasonal_cols = [c for c in frame if c.startswith('dow_')] + ['trend','year_sin','year_cos']
        cols = channel_cols + seasonal_cols
        x = frame[cols].fillna(0).to_numpy(dtype=float)
        y = frame['actual_qs'].to_numpy(dtype=float)
        minimum = int(settings['minimum_training_days'])
        if len(frame) < minimum:
            statuses.append({'country_code': market[0], 'brand': market[1],
                             'status': 'invalid_data', 'valid_days': len(frame),
                             'minimum_days': minimum,
                             'reason': f'Invalid data — only {len(frame)} valid days (minimum {minimum} required)'})
            continue
        statuses.append({'country_code': market[0], 'brand': market[1],
                         'status': 'valid', 'valid_days': len(frame),
                         'minimum_days': minimum, 'reason': ''})
        # Nonnegative coefficients on exposure and attributed QS; seasonal terms
        # are unconstrained via split positive/negative basis (Ridge positive=True).
        xc = x[:, :len(channel_cols)]
        xs = x[:, len(channel_cols):]
        xp = np.column_stack([xc, np.maximum(xs,0), np.maximum(-xs,0)])
        model = Ridge(alpha=float(settings['ridge_alpha']), positive=True)
        model.fit(xp, y)
        coefs = model.coef_[:len(channel_cols)]
        signal = np.maximum(xc @ coefs, 0)
        baseline = np.maximum(y - signal, 0)  # diagnostic only, not fitted baseline
        for i, (_, r) in enumerate(frame.iterrows()):
            record = {k: v for k,v in zip(group_cols,market)}
            record.update({'date_day':r['date_day'], 'actual_qs':float(y[i]),
                           'raw_channel_signal':float(signal[i]),
                           'raw_prediction':float(max(model.predict(xp[i:i+1])[0],0)),
                           'residual':float(y[i]-model.predict(xp[i:i+1])[0])})
            for j, col in enumerate(channel_cols):
                record[f'contribution__{col}'] = float(xc[i,j]*coefs[j])
            rows.append(record)
        for col, val in zip(channel_cols,coefs):
            coefficients.append(dict(zip(group_cols,market), feature=col, coefficient=float(val)))
    return pd.DataFrame(rows), pd.DataFrame(coefficients), pd.DataFrame(statuses)
