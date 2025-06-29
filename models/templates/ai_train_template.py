"""
Created on 5 April 2025
@authors: Charalampos

This script implements the abstract Trainer class, which provides
a common, basic functionality, across all model training scripts
that implement this class.
"""
import os
import sys
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split


class Trainer(ABC):
    """
        Create a train and test dataframe in order
        to train a new model.
    """
    def __init__(self, train_file, test_file, model_file):
        """
            Constructor of class Trainer

            :param train_file: str, filepath of train .csv
        """
        if not os.path.isfile(train_file):
            print(f"Data file {train_file} does not exist")
            sys.exit(1)
        if not os.path.isfile(train_file):
            print(f"Data file {test_file} does not exist")
            sys.exit(1)
        self._train_file = train_file
        self._test_file = test_file
        self.model_file = model_file
        self.df = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)

        self.rnd_seed = 2025

        self.train_df, self.val_df, self.test_df = self.get_train_test_dataframe(val_size=0.10)

    @property
    @abstractmethod
    def x_colnames(self):
        """
            Abstract property, it is a list of
             x-column names used by the model.
        """

    @property
    @abstractmethod
    def y_colname(self):
        """
            Abstract property, it is a string of
            y-column name, .i.e. the target column.
        """

    @abstractmethod
    def predict(self, x):
        """
            Abstract method, given x-data, do a
            prediction about y using the model.
        """

    @abstractmethod
    def train_model(self):
        """
            Abstract method, using train
            dataframe train the new model.
        """

    @abstractmethod
    def load_model(self):
        """
            Abstract method, load the model
            from existing model_file
        """

    def evaluate_model(self, x=None, y=None):
        """
            This method prints and returns various
            metrics regarding the model performance

            :param x: pd.DataFrame, x-validation data
            :param y: pd.Series, y-validation (true) labels
        """
        if x is not None and y is not None:
            return self.get_metrics(x, y)
        n_train = len(self.train_df)
        metr_train = self.get_metrics(
            self.train_df[self.x_colnames],
            self.train_df[self.y_colname]
        ).set_index(pd.Series(['train', ]))
        metr_val = self.get_metrics(
            self.val_df[self.x_colnames],
            self.val_df[self.y_colname]
        ).set_index(pd.Series(['val', ]))
        metr_test = self.get_metrics(
            self.test_df[self.x_colnames],
            self.test_df[self.y_colname]
        ).set_index(pd.Series(['test', ]))

        return pd.concat([metr_train, metr_val, metr_test])

    def get_metrics(self, x_data, y_data):
        """
            Does the actual metrics calculation for
            evaluate_model method.

            :param x_data: pd.DataFrame, x-validation data
            :param y_data: pd.Series, y-validation (true) labels
        """
        y_pred = self.predict(x_data)
        r2 = r2_score(y_data, y_pred)
        mape = mean_absolute_percentage_error(y_data, y_pred)
        rmse = root_mean_squared_error(y_data, y_pred)
        mse = mean_squared_error(y_data, y_pred)

        metrics = {
            'R2': [r2, ],
            'MAPE': [mape, ],
            'RMSE': [rmse, ],
            'MSE': [mse, ]
        }
        return pd.DataFrame(data=metrics)

    def get_train_test_dataframe(self, val_size):
        """
            Split the data in test, validation and train, and
            return the corresponding dataframes.
        """
        n_train = self.df.shape[0]
        n_test = self.df_test.shape[0]
        n = n_train + n_test
        val_ratio = n/n_train * val_size

        train_dataframe, val_dataframe = train_test_split(
            self.df,
            test_size=val_ratio,
            random_state=self.rnd_seed
        )

        return train_dataframe, val_dataframe, self.df_test
