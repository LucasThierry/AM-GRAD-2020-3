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

    def get_attribute_names(self):
        """
        Gets the names of the attributes.
        :return list[str]:
        """
        attributes_path = os.path.join(self._database_path, '{}_attribute_names.csv'.format(self._database_base_name))
        with open(attributes_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            return data[0]

    def get_attributes(self):
        """
        Gets the attributes of the database.
        """
        filepath = os.path.join(self._database_path, '{}_attributes_processed.csv'.format(self._database_base_name))
        names = self.get_attribute_names()
        dataset = pandas.read_csv(filepath, names=names)
        modifiedDataset=dataset.fillna(dataset.mean())
        print(modifiedDataset)
        return modifiedDataset

    def get_classes(self):
        """
        Gets the values of the classes of the database.
        """
        filepath = os.path.join(self._database_path, '{}_classes.csv'.format(self._database_base_name))
        names = ['has problem on APS']
        return pandas.read_csv(filepath, names=names)
