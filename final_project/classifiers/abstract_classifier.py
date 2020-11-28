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
        self.x_train = pandas_bean.x_train
        self.x_test = pandas_bean.x_test
        self.y_train = pandas_bean.y_train
        self.y_test = pandas_bean.y_test
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

    @staticmethod
    def _print_scores(scores):
        """
        Prints the scores of the classifier.
        :param cross_val_score scores:
        """
        print("Accuracy: %.3f (%.3f)" % (scores.mean(), scores.std()))
