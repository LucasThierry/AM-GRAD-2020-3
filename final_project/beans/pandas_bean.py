"""
Module for pandas bean.
"""

class PandasBean:
    """Class for pandas bean."""

    def __init__(self, x_train, x_test, y_train, y_test):
        """
        Class constructor.
        :param x_train: pandas x_train.
        :param x_test: pandas x_test.
        :param y_train: pandas y_train.
        :param y_test: pandas y_test.
        """
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test

    @property
    def x_train(self):
        """
        Getter for x_train.
        """
        return self._x_train

    @property
    def x_test(self):
        """
        Getter for x_test.
        """
        return self._x_test

    @property
    def y_train(self):
        """
        Getter for y_train.
        """
        return self._y_train

    @property
    def y_test(self):
        """
        Getter for y_test.
        """
        return self._y_test
