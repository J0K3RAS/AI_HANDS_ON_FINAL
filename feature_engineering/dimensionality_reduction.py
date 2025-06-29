import os
from functools import partial
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from utils.features import haversine_distance, unix_time, is_weekend, is_workhours, hour
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# --- Feature Extraction Method (PCA) ---
def evaluate_pca(df, numeric_features, categorical_features, n_components):
    """
    Evaluates PCA as a dimensionality reduction method using cross-validation.
    """
    scores = []
    kf = KFold(n_splits=4, shuffle=True, random_state=2025)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx].copy(deep=True)
        test_df = df.iloc[test_idx].copy(deep=True)

        # Scale numeric features before PCA
        scaler = StandardScaler()
        X_train_numeric_scaled = scaler.fit_transform(train_df[numeric_features])
        X_test_numeric_scaled = scaler.transform(test_df[numeric_features])

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_numeric_scaled)
        X_test_pca = pca.transform(X_test_numeric_scaled)

        # Convert PCA results back to DataFrame for consistent handling
        pca_train_cols = [f'pca_comp_{i}' for i in range(n_components)]
        pca_test_cols = [f'pca_comp_{i}' for i in range(n_components)]
        train_df_pca = pd.DataFrame(data=X_train_pca, columns=pca_train_cols, index=train_df.index)
        test_df_pca = pd.DataFrame(data=X_test_pca, columns=pca_test_cols, index=test_df.index)

        # Merge PCA components with non-numeric (categorical) features
        final_train_cols = [*pca_train_cols, *categorical_features]
        final_test_cols = [*pca_test_cols, *categorical_features]

        # Combine PCA components with categorical features
        train_df_combined = pd.concat([train_df_pca, train_df], axis=1)
        test_df_combined = pd.concat([test_df_pca, test_df], axis=1)

        # Fit Regressor on combined features (PCA components + categorical)
        reg = xgb.XGBRegressor(
            enable_categorical=True, objective='reg:squarederror',
            n_estimators=500, seed=2025
        )
        reg.fit(train_df_combined[final_train_cols], np.log(train_df['trip_duration']))

        # Predict and evaluate
        y_test = test_df['trip_duration']
        y_pred = np.exp(reg.predict(test_df_combined[final_test_cols]))
        mse = mean_squared_error(y_test, y_pred)
        scores.append(mse)

    return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(base_path, '../data', 'nyc-taxi-trip-duration', 'train_processed.csv')
    data = pd.read_csv(fpath)

    # Apply all defined new features to the dataframe
    # data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'Y': 1, 'N': 0})

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

    # Numeric features to be passed to PCA
    numeric_features = [
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'unix_time', 'haversine', 'sin_hour', 'cos_hour'
    ]
    categorical_features = [
        'weekend', 'workhours'
    ]
    pca_results = {
        'n_components': [],
        'mse_mean': [],
        'mse_std': []
    }

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[numeric_features])

    # Apply PCA with all components
    pca_all = PCA(n_components=len(numeric_features))
    pca_all.fit(X_scaled)

    # Print explained variance ratio for each component
    print("\nExplained variance ratio by component:")
    for i, ratio in enumerate(pca_all.explained_variance_ratio_):
        print(f"Component {i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")
    print(f"Cumulative variance explained: {sum(pca_all.explained_variance_ratio_) * 100:.2f}%")

    # --- Applying PCA with different number of components ---
    for n_comp in [2, 3, 4, 5, 6, 7, 8]:
        print(f"\n--- Applying PCA with {n_comp} components ---")
        mse_mean_pca, mse_std_pca = evaluate_pca(data, numeric_features, categorical_features, n_comp)
        pca_results['n_components'].append(n_comp)
        pca_results['mse_mean'].append(mse_mean_pca)
        pca_results['mse_std'].append(mse_std_pca)

    print("\n--- PCA Results ---")
    pca_results_df = pd.DataFrame(pca_results)
    print(pca_results_df.to_markdown(index=False))

    