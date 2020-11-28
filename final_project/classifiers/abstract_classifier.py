"""
Module for abstract classifier.
"""

from abc import ABC
from abc import abstractclassmethod


class AbstractClassifier(ABC):
    """Class for abstract classifier."""

    def __init__(self, pandas_bean, scores):
        """
        :param PandasBean pandas_bean:
        :param list[str] scores:
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.scores = scores

    @abstractclassmethod
    def grid_search(self):
        """
        Performs a grid search on the classifier
        """
        pass

    @abstractclassmethod
    def evaluate(self):
        """
        Evaluates the classifier.
        """
        pass

    @abstractclassmethod
    def _get_attributes(self):
        """
        Returns the attributes.
        :return dict:
        """
        pass
