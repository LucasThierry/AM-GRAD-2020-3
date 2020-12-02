"""
Module for pandas bean.
"""

class PandasBean:
    """Class for pandas bean."""

    def __init__(self, training_attributes, training_classes, test_attributes, test_classes):
        """
        Class constructor.
        :param training_attributes: pandas training_attributes.
        :param training_classes: pandas training_classes.
        :param test_attributes: pandas test_attributes.
        :param test_classes: pandas test_classes.
        """
        self._training_attributes = training_attributes
        self._training_classes = training_classes
        self._test_attributes = test_attributes
        self._test_classes = test_classes

    @property
    def training_attributes(self):
        """
        Getter for training_attributes.
        """
        return self._training_attributes

    @property
    def training_classes(self):
        """
        Getter for training_classes.
        """
        return self._training_classes

    @property
    def test_attributes(self):
        """
        Getter for test_attributes.
        """
        return self._test_attributes

    @property
    def test_classes(self):
        """
        Getter for test_classes.
        """
        return self._test_classes
