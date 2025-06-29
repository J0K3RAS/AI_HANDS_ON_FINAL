import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from models.xgb_reg import XGBModel
from sklearn.inspection import permutation_importance

# --- 1. LOAD MODEL ---
print("--- Loading Model ---")
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'hp_xgbreg.xz')
model = XGBModel().load_model(model_path)

# --- 2. BUILT-IN FEATURE IMPORTANCE PLOTS ---
print("--- Generating XGBoost Built-in Feature Importance Plots ---")
# 'weight': the number of times a feature is used to split the data
# 'gain': the average gain of splits which use the feature
with plt.ioff():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    fig.suptitle('XGBoost Built-in Feature Importance', fontsize=16, y=0.93)

    # Plot by 'gain'
    xgb.plot_importance(model.model, ax=ax1, importance_type='gain', title='Importance by Gain')

    # Plot by 'weight'
    xgb.plot_importance(model.model, ax=ax2, importance_type='weight', title='Importance by Weight (Frequency)')

    plt.savefig(
        os.path.join(base_path, 'plots', 'feature_importance.png'),
    )
print("Built-in feature importance plots displayed.\n")


# --- 3. RESIDUAL PLOTS ---
test_df = pd.read_csv(os.path.join(base_path, '../data/nyc-taxi-trip-duration/test_processed.csv'))
X_test = test_df[model.x_colnames]
y_test = test_df[model.y_colname]
print("--- Generating Residual Plots On Test Data ---")
# Calculate predictions and residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred
with plt.ioff():
    # Create a figure for residual plots
    plt.figure(figsize=(15, 6))
    plt.suptitle('Residual Analysis', fontsize=16)

    # Plot 1: Residuals vs. Predicted Values
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')

    # Plot 2: Histogram of Residuals
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')

    plt.savefig(
        os.path.join(base_path, 'plots', 'residuals.png'),
        bbox_inches='tight'
    )
print("Residual plots displayed.\n")


# --- 4. PERMUTATION IMPORTANCE ---
X_test_processed = model.preprocess(X_test)
feature_names = X_test_processed.columns
print("--- Generating Permutation Importance Plot ---")
perm_importance = permutation_importance(model.model, X_test_processed, np.log(y_test), n_repeats=10, random_state=2025)

# Sort features for plotting
sorted_idx = perm_importance.importances_mean.argsort()

with plt.ioff():
    plt.figure(figsize=(10, 8))
    plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx])
    plt.title("Permutation Importance (Test Set)")
    plt.xlabel("Importance Score Decrease")
    plt.tight_layout()
    plt.savefig(
        os.path.join(base_path, 'plots', 'permutation_importance.png'),
        bbox_inches='tight'
    )
print("Permutation importance plot displayed.\n")
