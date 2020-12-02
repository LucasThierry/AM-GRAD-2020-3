"""
Module for database manager.
"""

import csv
import os

import pandas


class DatabaseManager:
    """Class for managing the database."""

    def __init__(self, database_path, database_base_name):
        """
        Class constructor.
        :param str database_path: the path to the database.
        :param database_base_name: the base name of the database.
        """
        self._database_path = database_path
        self._database_base_name = database_base_name

    def get_training_set(self):
        """
        Gets the training set.
        :return tuple:
        """
        x_train = self._get_attributes('training')
        x_test = self._get_classes('training')
        return x_train, x_test

    def get_test_set(self):
        """
        Gets the training set.
        :return tuple:
        """
        y_train = self._get_attributes('test')
        y_test = self._get_classes('test')
        return y_train, y_test

    def _get_attribute_names(self):
        """
        Gets the names of the attributes.
        :return list[str]:
        """
        attributes_path = os.path.join(self._database_path, '{}_attribute_names.csv'.format(self._database_base_name))
        with open(attributes_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            return data[0]

    def _get_attributes(self, dataset_type):
        """
        Gets the attributes of the database.
        :param str dataset_type: Valid values are training or test.
        :return list:
        """
        filepath = os.path.join(self._database_path, '{}_{}_set_attributes.csv'.format(self._database_base_name, dataset_type))
        names = self._get_attribute_names()
        dataset = pandas.read_csv(filepath, names=names)
        modifiedDataset=dataset.fillna(dataset.mean())
        return modifiedDataset

    def _get_classes(self, dataset_type):
        """
        Gets the values of the classes of the database.
        :param str dataset_type: Valid values are training or test.
        :return list:
        """
        filepath = os.path.join(self._database_path, '{}_{}_set_classes.csv'.format(self._database_base_name, dataset_type))
        names = ['has problem on APS']
        return pandas.read_csv(filepath, names=names)
