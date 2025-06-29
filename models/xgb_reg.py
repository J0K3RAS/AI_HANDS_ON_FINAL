"""
Created on 5 April 2025
@authors: Charalampos

This script implements the xgboost regressor for trip duration prediction.
"""
import os
import pickle
import tarfile
import tempfile
import numpy as np
import xgboost as xgb
from functools import partial
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from models.templates.ai_model_template import Model
from utils.features import haversine_distance, unix_time, is_weekend, is_workhours, hour

class XGBModel(Model):
    """
        Implements the Model abstract class, for
        trip duration XGBoost regression model.
    """
    def __init__(self):
        """ Constructor of class XGBModel. """
        super().__init__()
        self.y_colname = 'trip_duration'
        self.x_colnames = [
             'pickup_datetime', 'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude',
            #'unix_time', 'haversine', 'sin_hour', 'cos_hour'
        ]
        self.model_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'learning_rate': 0.3,
        }
        # Define the new features and their functions
        self._new_features = {
            'unix_time': partial(unix_time, col='pickup_datetime'),
            'weekend': partial(is_weekend, col='pickup_datetime'),
            'workhours': partial(is_workhours, col='pickup_datetime'),
            'haversine': partial(haversine_distance,
                                 lat1_col='pickup_latitude', lon1_col='pickup_longitude',
                                 lat2_col='dropoff_latitude', lon2_col='dropoff_longitude'
                                 ),
            'sin_hour~cos_hour': partial(hour, col='pickup_datetime'),
        }
        # Define the standard scaler and pca
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=8)

    def preprocess(self, x):
        x = x.copy(deep=True)
        for k, f in self._new_features.items():
            col = k.split('~') if '~' in k else k
            x[col] = f(x)
        x_cat = ['weekend', 'workhours',]
        x_num = [
            'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
            'dropoff_longitude', 'unix_time', 'haversine', 'sin_hour', 'cos_hour'
        ]
        x_pca = [f'component_{i+1}' for i in range(self.pca.n_components)]
        x[x_cat] = x[x_cat].astype('category')
        if not hasattr(self.scaler, "n_features_in_"):
            self.scaler.fit(x[x_num])
        x[x_num] = self.scaler.transform(x[x_num])
        if not hasattr(self.pca, "n_features_in_"):
            self.pca.fit(x[x_num])
        x[x_pca] = self.pca.transform(
            x[x_num]
        )
        return x[[*x_cat, *x_pca]]

    def load_model(self, path):
        """
            Load the model from model_file.
        """
        with tarfile.open(path, "r:gz") as tar, tempfile.TemporaryDirectory() as temp_dir:
            tar.extractall(
                path=temp_dir,
                members=('xgbreg.pkl', 'scaler.pkl', 'pca.pkl')
            )
            model_temp_file = os.path.join(temp_dir, 'xgbreg.pkl')
            scaler_temp_file = os.path.join(temp_dir, 'scaler.pkl')
            pca_temp_file = os.path.join(temp_dir, 'pca.pkl')

            model_pred =  pickle.load(open(model_temp_file, 'rb'))
            scaler = pickle.load(open(scaler_temp_file, 'rb'))
            pca = pickle.load(open(pca_temp_file, 'rb'))

        self.model = model_pred
        self.scaler = scaler
        self.pca = pca
        return self

    def save_model(self, path):
        """
            Save the model to the specified path.

            :param path: str, the save path
        """
        # Temporary filenames for the model and standardizer
        model_pred_temp_file = 'xgbreg.pkl'
        scaler_temp_file = 'scaler.pkl'
        pca_temp_file = 'pca.pkl'

        try:
            # Save the model to a temporary 'xgbreg.pkl'
            with open(model_pred_temp_file, 'wb') as f:
                pickle.dump(self.model, f)

            # Save the standardizer to a temporary 'scaler.pkl'
            with open(scaler_temp_file, 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save the pca to a temporary 'pca.pkl'
            with open(pca_temp_file, 'wb') as f:
                pickle.dump(self.pca, f)

            # Create a tar.gz archive and add all files
            with tarfile.open(path, "w:gz") as tar:
                tar.add(model_pred_temp_file, arcname='xgbreg.pkl')
                tar.add(scaler_temp_file, arcname='scaler.pkl')
                tar.add(pca_temp_file, arcname='pca.pkl')

        finally:
            # Clean up temporary files
            if os.path.exists(model_pred_temp_file):
                os.remove(model_pred_temp_file)
            if os.path.exists(scaler_temp_file):
                os.remove(scaler_temp_file)
            if os.path.exists(pca_temp_file):
                os.remove(pca_temp_file)

        return self

    def predict(self, x):
        """
            Make a prediction for x-data.

            :param x: pd.DataFrame, x-data
        """
        x_pred = x[self.x_colnames]
        x_pred = self.preprocess(x_pred)
        y_pred = self.model.predict(x_pred)
        return np.exp(y_pred)

    def train_model(self, x_train, y_train, x_val, y_val):
        """
            Using train data, train the
            XGBoost regressor model.

            :param x_train: pd.DataFrame, x train data
            :param y_train: pd.Series, y train data
            :param x_val: pd.DataFrame, x validation data
            :param y_val: pd.Series, y validation data
        """
        x_train = self.preprocess(x_train[self.x_colnames])
        x_val = self.preprocess(x_val[self.x_colnames])

        y_train = np.log(y_train)
        y_val = np.log(y_val)
        # Get the model using the corresponding method
        model = self.get_model()

        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

        self.model = model

        return self

    def get_model(self):
        """
            Define the xgboost regressor

            :return: XGBRegressor regressor
        """
        # Define the early stopping callback
        es = xgb.callback.EarlyStopping(
            rounds=10,
            min_delta=1e-4,
            save_best=True,
            maximize=True,
            data_name="validation_0",
            metric_name=r2_score.__name__,
        )

        # Define the XGBoost Regressor model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            enable_categorical=True,
            eval_metric=r2_score,
            callbacks=[es],
            n_estimators=int(self.model_params['n_estimators']),
            max_depth=int(self.model_params['max_depth']),
            learning_rate=self.model_params['learning_rate'],
            reg_alpha=self.model_params['reg_alpha'],
            reg_lambda=self.model_params['reg_lambda'],
            random_state=self.rnd_seed,
            n_jobs=-1
        )
        return model
