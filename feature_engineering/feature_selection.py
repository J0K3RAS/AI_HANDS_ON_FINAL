import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from utils.features import haversine_distance, unix_time, is_weekend, is_workhours, hour
from sklearn.feature_selection import SelectKBest, f_regression


def evaluate_feature_selection(df, all_features, method_name, selector_func):
    """
    Evaluates a given feature selection method using cross-validation.
    """
    scores = []
    selected_features_per_fold = []  # To store selected features for each fold

    kf = KFold(n_splits=4, shuffle=True, random_state=2025)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx].copy(deep=True)
        test_df = df.iloc[test_idx].copy(deep=True)

        X_train = train_df[all_features]
        y_train = np.log(train_df['trip_duration'])  # Log transform target for model fitting

        # Apply feature selection on the training data
        selector = selector_func()
        selector.fit(X_train, y_train)

        # Get selected features
        selected_cols = [col for col, selected in zip(all_features, selector.get_support()) if selected]
        selected_features_per_fold.append(selected_cols)

        if not selected_cols:
            print(f"Warning: No features selected by {method_name} in fold {fold_idx + 1}. Skipping fold.")
            continue

        # Define numerical and categorical columns for the selected features for standardization
        x_num_selected = [f for f in selected_cols if is_numeric.get(f, True)]  # Assume numeric if not specified
        x_cat_selected = [f for f in selected_cols if not is_numeric.get(f, False)]

        # Standardization on selected numerical features
        scaler = StandardScaler()
        train_df[x_num_selected] = scaler.fit_transform(train_df[x_num_selected])
        test_df[x_num_selected] = scaler.transform(test_df[x_num_selected])

        # Fit Regressor with selected features
        reg = xgb.XGBRegressor(enable_categorical=True, objective='reg:squarederror',
                               n_estimators=500, seed=2025)
        reg.fit(train_df[selected_cols], y_train)

        # Predict and evaluate
        y_test = test_df['trip_duration']
        y_pred = np.exp(reg.predict(test_df[selected_cols]))
        mse = mean_squared_error(y_test, y_pred)
        scores.append(mse)

    if not scores:
        return np.nan, np.nan, []  # Return NaN if no scores were collected

    # Count frequency of each feature across folds
    feature_freq = {}
    for fold_features in selected_features_per_fold:
        for feature in fold_features:
            feature_freq[feature] = feature_freq.get(feature, 0) + 1

    # Sort features by frequency and get top k (k = number of features selected in first fold)
    k = len(selected_features_per_fold[0]) if selected_features_per_fold else 0
    top_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)[:k]
    top_feature_names = [f[0] for f in top_features]

    print(f"\nTop {k} features selected by {method_name} (by frequency): {top_feature_names}")

    return np.mean(scores), np.std(scores), top_feature_names

def importance(df, all_features):
    """
    Calculates and visualizes the feature importance of a dataset based on a
    trained XGBoost regression model. The method excludes a predefined feature
    ('Fare') from the feature importance calculation. It uses the specified
    categorical and numerical features for training the model.

    The feature importance is computed based on two metrics, 'weight' and
    'gain', provided by the XGBoost library. The calculated importances are
    normalized as percentages. Finally, it generates and saves a bar plot
    representing the feature importances.

    :param self: Reference to the current instance of the class. Assumes the
        presence of specific attributes such as `x_num`, `x_cat`, `data`, and
        `out_dir`.
    :returns: None.
    :raises: This method may raise exceptions related to issues in data
        processing, model training, or file system operations during plot saving.
    """
    xgb_reg = xgb.XGBRegressor(enable_categorical=True, objective='reg:squarederror',
                               n_estimators=500, seed=2025)
    xgb_reg.fit(
        df[all_features],
        np.log(df['trip_duration'])
    )
    # print(f"RMSE: {np.sqrt(np.square(xgb_reg.predict(self.data[x_cols])-self.data['Fare']).mean())}")
    # print(f"MPE: {np.abs(((xgb_reg.predict(self.data[x_cols]) - self.data['Fare'])/self.data['Fare']*100)).mean()}")
    imp_df = pd.DataFrame()
    for imp in ['weight', 'gain']:
        f_importance = xgb_reg.get_booster().get_score(importance_type=imp)
        df = pd.DataFrame({
            'feature': list(f_importance.keys()),
            'importance': list(f_importance.values())
        })
        df['importance'] = df['importance'] / df['importance'].sum() * 100
        df['type'] = np.repeat(imp, len(df))
        imp_df = pd.concat([imp_df, df], ignore_index=True)
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', hue='type', data=imp_df)
    plt.title("Feature Importance [%]")
    plt.savefig(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../data',
            'plots',
            "feature_importance.png"
        ),
        bbox_inches="tight"
    )
    # self.xgb = xgb_reg
    return


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(base_path, '../data', 'nyc-taxi-trip-duration', 'train_processed.csv')
    data = pd.read_csv(fpath)

    # data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'Y': 1, 'N': 0})  # Binarize if not already

    unix = partial(unix_time, col='pickup_datetime')
    weekend = partial(is_weekend, col='pickup_datetime')
    workhours = partial(is_workhours, col='pickup_datetime')
    haversine = partial(haversine_distance, lat1_col='pickup_latitude', lon1_col='pickup_longitude',
                        lat2_col='dropoff_latitude', lon2_col='dropoff_longitude')
    hour_of_day = partial(hour, col='pickup_datetime')

    data['unix_time'] = unix(data)
    data['weekend'] = weekend(data)
    data['workhours'] = workhours(data)
    data['haversine'] = haversine(data)
    data[['sin_hour','cos_hour']] = hour_of_day(data)

    # Define all candidate features to be considered for selection
    # This should include existing features and newly engineered ones.
    all_features = [
        'vendor_id', 'passenger_count', 'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude', 'store_and_fwd_flag',
        'unix_time', 'weekend', 'workhours', 'haversine', 'sin_hour', 'cos_hour'
    ]


    is_numeric = {
        'vendor_id': False,
        'passenger_count': True,
        'pickup_latitude': True,
        'pickup_longitude': True,
        'dropoff_latitude': True,
        'dropoff_longitude': True,
        'store_and_fwd_flag': False,  # Binarized,
        'unix_time': True,
        'weekend': False,
        'workhours': False,
        'haversine': True,
        'sin_hour': True,
        'cos_hour': True,
    }
    print("\n--- Applying XGBoost Importance ---")
    importance(data, all_features)

    feature_selection_results = {
        'method': [],
        'mse_mean': [],
        'mse_std': [],
        'selected_features_example': []
    }

    # --- SelectKBest ---
    print("\n--- Applying SelectKBest for Feature Selection ---")
    k_best = 6
    select_k_best_func = lambda: SelectKBest(score_func=f_regression, k=k_best)
    mse_mean_skb, mse_std_skb, selected_feats_skb = evaluate_feature_selection(data, all_features,
                                                                               f"SelectKBest (k={k_best})",
                                                                               select_k_best_func)
    print(pd.DataFrame({
        'method': [f'SelectKBest (k={k_best})', ],
        'mse_mean': [mse_mean_skb, ],
        'mse_std': [mse_std_skb, ],
        'selected_features': [selected_feats_skb, ],
    }, index=[0,]).to_markdown(index=False))