import argparse
import csv
import os

import pandas
from sklearn.model_selection import train_test_split

from classifiers import KNN
from classifiers import Tree
from classifiers import RandomForest
from classifiers import MLP
from classifiers import EnsembleMLP
from classifiers import Ensemble


def get_attribute_names():
    """
    Gets the names of the attributes.
    :return list[str]:
    """
    attributes_path = os.path.join('database', 'aps_failure_attribute_names.csv')
    with open(attributes_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        return data[0]


def get_attributes():
    """
    Gets the attributes of the database.
    """
    filepath = os.path.join('database', 'aps_failure_attributes_processed.csv')
    names = get_attribute_names()
    return pandas.read_csv(filepath, names=names)


def get_classes():
    """
    Gets the values of the classes of the database.
    """
    filepath = os.path.join('database', 'aps_failure_classes.csv')
    names = ['has problem on APS']
    return pandas.read_csv(filepath, names=names)


def arguments_definition():
    """
    Method for creating the possible parameters for execution.
    :return ArgumentParser:
    """
    parser = argparse.ArgumentParser(description='Runs the algorithms.')
    parser.add_argument(
        'algorithm',
        type=str,
        choices=['knn', 'tree', 'forest', 'mlp', 'ensemble-mlp', 'ensemble'],
        help='The algorithm to be executed.')
    parser.add_argument(
        'method',
        type=str,
        choices=['evaluate', 'grid_search'],
        help='The method to be executed.')

    return parser.parse_args()


def run():
    args = arguments_definition()

    pandas_attributes = get_attributes()
    pandas_classes = get_attributes()

    x_train, x_test, y_train, y_test = train_test_split(
        pandas_attributes, pandas_classes.values.ravel(), test_size=0.5, random_state=0)

    scores = ['precision', 'recall']

    algorithm = args.algorithm
    method = args.method

    if algorithm == 'knn':
        classifier = KNN(x_train, x_test, y_train, y_test, scores)
    elif algorithm == 'tree':
        classifier = Tree(x_train, x_test, y_train, y_test, scores)
    elif algorithm == 'forest':
        classifier = RandomForest(x_train, x_test, y_train, y_test, scores)
    elif algorithm == 'mlp':
        classifier = MLP(x_train, x_test, y_train, y_test, scores)
    elif algorithm == 'ensemble-mlp':
        classifier = EnsembleMLP(x_train, x_test, y_train, y_test, scores)
    elif algorithm == 'ensemble':
        classifier = Ensemble(x_train, x_test, y_train, y_test, scores)
    else:
        raise Exception('Invalid classifier: {}'.format(algorithm))

    if method == 'evaluate':
        classifier.evaluate()
    elif method == 'grid_search':
        classifier.grid_search()
    else:
        raise Exception('Invalid method: {}'.format(method))

if __name__ == '__main__':
    run()




