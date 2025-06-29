"""
Created on 5 April 2025
@authors: Charalampos

This script provides the abstract class Model.
"""
from abc import ABC, abstractmethod


class Model(ABC):
    """
        Abstract class model, a high level
        description of what a model should
        implementation should look like.
    """
    def __init__(self):
        self.rnd_seed = 2025
        self._model = None
        self._x_colnames = []
        self._y_colname = None

    @property
    def x_colnames(self):
        """
            Abstract property, it is a list of
             x-column names used by the model.
        """
        return self._x_colnames

    @x_colnames.setter
    def x_colnames(self, x):
        self._x_colnames = x

    @property
    def y_colname(self):
        """
            Abstract property, it is a string of
            y-column name, .i.e. the target column.
        """
        return self._y_colname

    @y_colname.setter
    def y_colname(self, y):
        self._y_colname = y

    @property
    def model(self):
        """
            Model property.
        """
        return self._model

    @model.setter
    def model(self, value):
        """
            Model property setter.
        """
        self._model = value

    @abstractmethod
    def predict(self, x):
        """
            Abstract method, given x-data, do a
            prediction about y using the model.
        """

    @abstractmethod
    def train_model(self, x_train, y_train, x_val, y_val):
        """
            Abstract method, using train
            dataframe train the new model.
        """

    @abstractmethod
    def load_model(self, path):
        """
            Abstract method, load the model
            from existing model file path

            :param path: str, the model filepath
        """

    @abstractmethod
    def save_model(self, path):
        """
            Save the model to the specified path.

            :param path: str, the save path
        """
