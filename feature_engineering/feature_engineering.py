import os
from contextlib import closing
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from utils.features import rain_probability, haversine_distance, unix_time, is_weekend, is_workhours, hour

def process_fold(train_df, test_df, x_num, x_cols):
    # Standardization
    scaler = StandardScaler()
    train_df[x_num] = scaler.fit_transform(train_df[x_num])
    test_df[x_num] = scaler.transform(test_df[x_num])
    # Fit Regressor
    params = {
        "n_estimators": 500,
        "objective": "reg:squarederror",
        "seed": 2025,
        "n_jobs": 2
    }

    reg = xgb.XGBRegressor(**params)
    reg.fit(train_df[x_cols], np.log(train_df['trip_duration']))
    # Predict
    y_test = test_df['trip_duration']
    y_pred = np.exp(reg.predict(test_df[x_cols]))
    mse = mean_squared_error(y_test, y_pred)
    return mse

base_path = os.path.dirname(os.path.abspath(__file__))
fpath = os.path.join(base_path, '../data', 'nyc-taxi-trip-duration', 'train_processed.csv')
data = pd.read_csv(fpath)
# data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'Y': 1, 'N': 0})  # Binarize the feature

baseline = lambda x : x['passenger_count']  # Placeholder for baseline performance
unix = partial(unix_time, col='pickup_datetime')
weekend = partial(is_weekend, col='pickup_datetime')
workhours = partial(is_workhours, col='pickup_datetime')
hour_of_day = partial(hour, col='pickup_datetime')
#rain = partial(rain_probability, lat='pickup_latitude', long='pickup_longitude', date='pickup_datetime')
haversine = partial(haversine_distance, lat1_col='pickup_latitude', lon1_col='pickup_longitude', lat2_col='dropoff_latitude', lon2_col='dropoff_longitude')

new_features = {
    'sin_hour~cos_hour': hour_of_day,
    'passenger_count': baseline,  # Identity mapping passenger_count to itself
    'unix_time': unix,
    'weekend': weekend,
    'workhours': workhours,
    #'rain': rain,
    'haversine': haversine,
}
is_numeric = {
    'sin_hour~cos_hour': True,
    'passenger_count': True,
    'unix_time': True,
    'weekend': False,
    'workhours': False,
    #'rain': True,
    'haversine': True,
}

feature_performance = {
    'feature': [],
    'mse_mean': [],
    'mse_std': []
}
for k,v in new_features.items():
    print(f"Processing feature: {k}")
    df = data.copy(deep=True)
    x_num = {
        'passenger_count', 'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
    }
    x_cat = {'vendor_id', 'passenger_count', 'store_and_fwd_flag'}
    if '~' in k:
        df[k.split('~')] = v(df)
        if is_numeric[k]:
            x_num.update(set(k.split('~')))
        else:
            x_cat.update(set(k.split('~')))
    else:
        df[k] = v(df)
        if is_numeric[k]:
            x_num.add(k)
        else:
            x_cat.add(k)
    x_cols = {*x_num, *x_cat}
    # Initialize 4-fold CV
    kf = KFold(n_splits=4, shuffle=True, random_state=2025)
    # Split generator (returns train/test indices)
    fold_func = partial(process_fold, x_num=list(x_num), x_cols=list(x_cols))
    with closing(Pool(processes=4)) as pool:
        scores = pool.starmap(
            fold_func,
            [(df.iloc[train_idx].copy(deep=True), df.iloc[test_idx].copy(deep=True)) for train_idx, test_idx in kf.split(df)]
        )
    feature_performance['feature'].append(k)
    feature_performance['mse_mean'].append(np.mean(scores))
    feature_performance['mse_std'].append(np.std(scores))
# Update the name of the placeholder feature
idx = feature_performance['feature'].index('passenger_count')
feature_performance['feature'][idx] = 'baseline'
feature_performance = pd.DataFrame(feature_performance)
print(feature_performance.to_markdown())
# print(feature_performance.to_latex())
