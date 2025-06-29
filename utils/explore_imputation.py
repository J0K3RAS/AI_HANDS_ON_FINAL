import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from  sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer

def get_cv_score(X, y, splits, regressor):
    scorer = make_scorer(mean_squared_log_error)
    full_scores = cross_val_score(
        regressor, X, y, scoring=scorer,
        cv=splits, n_jobs=-1
    )
    full_scores = np.sqrt(full_scores)
    return full_scores.mean(), full_scores.std()

def main(X, splits=4, regressor=None):
    if regressor is None:
        regressor = RandomForestRegressor(
            n_jobs=-1,
            random_state=2025
        )
    zero_imputer = SimpleImputer(
        missing_values=np.nan, strategy="constant", fill_value=0
    )
    mean_imputer = SimpleImputer(
        missing_values=np.nan, strategy="mean"
    )
    median_imputer = SimpleImputer(
        missing_values=np.nan, strategy="median"
    )
    iterative_imputer = IterativeImputer(
        random_state=2025
    )
    data = {
        'imputer': [zero_imputer, mean_imputer, median_imputer, iterative_imputer],
        'name': ['zero', 'mean', 'median', 'iterative'],
        'mean_score': [],
        'std_score': []
    }
    for imputer, name in zip(data['imputer'], data['name']):
        print(f"Running {name} imputer")
        Xi = X.copy()
        Xi = pd.DataFrame(data=imputer.fit_transform(Xi), columns=Xi.columns)
        yi = Xi['trip_duration']
        Xi.drop(columns=['trip_duration', ], inplace=True)
        mean, std = get_cv_score(Xi, yi, splits, regressor)
        print(f"Mean CV score: {mean:.4f} +/- {std:.4f}")
        data['mean_score'].append(mean)
        data['std_score'].append(std)

    return pd.DataFrame(data)