# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.read_cords import get_nyc_box

# ------------------------------------------------------------------------------
# Data Loading and Initial Preprocessing
# ------------------------------------------------------------------------------
base = os.path.dirname(os.path.abspath(__file__))
path_train = os.path.join(base, '..', 'data', 'nyc-taxi-trip-duration', 'train.csv')
path_test = os.path.join(base, '..', 'data', 'nyc-taxi-trip-duration', 'test.csv')

train = pd.read_csv(path_train)
test = pd.read_csv(path_test)
# Add an indicator column on each dataset
train["is_train"] = True
test["is_train"] = False

data = pd.concat([train, test], ignore_index=True)
print(f"Train data points: {train.shape[0]}")
print(f"Total data points: {data.shape[0]}")
print(f"Train ratio: {train.shape[0] / data.shape[0]:.2%}")

print(data.info())
print(data.describe().to_markdown())
print(data.isna().sum())

# ------------------------------------------------------------------------------
# Data Cleaning and Filtering
# ------------------------------------------------------------------------------
data = data.dropna()
print(f"Total datapoints after dropping nans: {data.shape[0]}")

for hour in range(1, 8):
    n = data[(1/3600 * data["trip_duration"]) > hour].shape[0]
    print(f"Rides over {hour} hours: {n}")

for minute in range(1, 6):
    minute /= 2.0
    n = data[(1/60 * data["trip_duration"]) < minute].shape[0]
    print(f"Rides under {minute} minutes: {n}")

dur_lb = (1/60 * data["trip_duration"]) >= 0.5
dur_ub = (1/3600 * data["trip_duration"]) <= 2.0

print(f"Total datapoints before filtering duration: {data.shape[0]}")
data = data[dur_lb & dur_ub].reset_index(drop=True)
print(f"Total datapoints after filtering duration: {data.shape[0]}")


n_passengers, count = np.unique(data['passenger_count'], return_counts=True)
for k, v in zip(n_passengers, count):
    print(f"Taxi rides with {k} passengers : {v}")

# See in more details taxi rides with unexpected number of passengers
for i in [0, 8, 9]:
    print(f"Taxi rides with {i} passengers :")
    print(data[data['passenger_count'] == i].to_markdown())

data = data[data['passenger_count'] < 8].reset_index(drop=True)
data.loc[data['passenger_count'] == 0, 'passenger_count'] = 1

# ------------------------------------------------------------------------------
# Outlier Detection and Analysis
# ------------------------------------------------------------------------------
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
data['pickup_time_unix'] = data['pickup_datetime'].apply(lambda x: x.timestamp())  # Convert time to numeric
data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'Y': 1, 'N': 0})  # Binarize the feature
use_xcols = [
    'pickup_time_unix', 'vendor_id', 'passenger_count', 'store_and_fwd_flag',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
    'trip_duration'
]

iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=2025, n_jobs=-1)
is_outlier = iso_forest.fit_predict(data[use_xcols])
data['is_outlier'] = np.where(is_outlier == -1, True, False)

# Do a plot of the outliers
cols_cont = [
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude',
    'pickup_time_unix', 'trip_duration'
]
cols_cat = [
    'vendor_id', 'passenger_count', 'store_and_fwd_flag'
]
with plt.ioff():
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    for i, column in enumerate(cols_cont):
        row = i // 2
        col = i % 2
        sns.boxplot(data=data, x="is_outlier", y=column, ax=ax[row, col])
    fig.suptitle('Continuous variables', fontsize=16)
    plt.savefig(
        os.path.join(base, '..', 'data', 'plots', 'outliers_continuous.png'),
        bbox_inches='tight'
    )
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    for i, column in enumerate(cols_cat):
        # sns.histplot(
        #     data=data, x="is_outlier", hue=column, multiple="dodge",
        #     stat='density', shrink=0.8, common_norm=False, ax=ax[i]
        # )
        sns.countplot(data=data, x="is_outlier", hue=column, stat="percent", ax=ax[i])
    fig.suptitle('Discrete variables', fontsize=16)
    plt.savefig(
        os.path.join(base, '..', 'data', 'plots', 'outliers_discrete.png'),
        bbox_inches='tight'
    )
# ------------------------------------------------------------------------------
# Geographical Filtering
# ------------------------------------------------------------------------------
cords_fpath = os.path.join(base, '..', 'data', 'DCM_StreetNameChanges_Points_20250622.csv')
min_lat, max_lat, min_long, max_long = get_nyc_box(
    pd.read_csv(cords_fpath)
)
in_nyc_pickup = data['pickup_longitude'].between(min_long, max_long) & data['pickup_latitude'].between(min_lat, max_lat)
in_nyc_dropoff = data['dropoff_longitude'].between(min_long, max_long) & data['dropoff_latitude'].between(min_lat, max_lat)

print(f"Datapoints before dropping rides outside NYC: {data.shape[0]}")
data = data[in_nyc_pickup & in_nyc_dropoff].reset_index(drop=True)
print(f"Datapoints after dropping rides outside NYC: {data.shape[0]}")
# ------------------------------------------------------------------------------
# Trip Duration Analysis
# ------------------------------------------------------------------------------
data['log_trip_duration'] = np.log(data['trip_duration'])
with plt.ioff():
    fig,ax = plt.subplots(1, 2, figsize=(12,12))
    sns.histplot(data=data, x="trip_duration", kde=True, ax=ax[0])
    sns.histplot(data=data, x="log_trip_duration", kde=True, ax=ax[1])
    fig.suptitle('Trip duration', fontsize=16)
    plt.savefig(
        os.path.join(base, '..', 'data', 'plots', 'trip_duration_distribution.png'),
        bbox_inches='tight'
    )
# ------------------------------------------------------------------------------
# Outlier Detection and Analysis (Cont'd)
# ------------------------------------------------------------------------------
xcols = [
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude',
    'pickup_time_unix',
]
df = data.copy()
df_std = df.copy()
df_std[[*xcols, 'log_trip_duration']] = StandardScaler().fit_transform(df_std[[*xcols, 'log_trip_duration']])
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=2025, n_jobs=-1)
oc_svm = SGDOneClassSVM(nu=0.05, shuffle=True, random_state=2025)
print('Fitting iso forest')
iso_pred = iso_forest.fit_predict(df[[*xcols, *cols_cat, 'trip_duration']])
print('Fitting oc_svm')
oc_pred = oc_svm.fit_predict(df_std[[*xcols, *cols_cat, 'log_trip_duration']])
common = np.where(iso_pred == -1, True, False) & np.where(oc_pred == -1, True, False)
print(f'Isolation forest outliers: {np.where(iso_pred == -1, True, False).sum()}')
print(f'SGD OneClass SVM outliers: {np.where(oc_pred == -1, True, False).sum()}')
print(f'Common outliers: {common.sum()}')

data = data[~common].reset_index(drop=True)
print(f'Data points after dropping common outliers: {data.shape[0]}')


path_train_out = os.path.join(base, '..', 'data', 'nyc-taxi-trip-duration', 'train_processed.csv')
path_test_out = os.path.join(base, '..', 'data', 'nyc-taxi-trip-duration', 'test_processed.csv')

df_train, df_test = train_test_split(data[train.columns], test_size=0.10, random_state=2025)
df_train.to_csv(path_train_out, index=False)
df_test.to_csv(path_test_out, index=False)