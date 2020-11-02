import os
import argparse

import pandas
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def get_attribute_names():
    """
    Gets the names of the attributes of the database.
    """
    filepath = os.path.join('database', 'glass_attributes_processed.csv')
    names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba','Fe']
    return pandas.read_csv(filepath, names=names)


def get_attributes():
    """
    Gets the values of the attributes of the database.
    """
    filepath = os.path.join('database', 'glass_classes.csv')
    names = ['Type of glass']
    return pandas.read_csv(filepath, names=names)


def get_ensemble_grid_parameters():
    """
    Gets the parameters for the grid search testing.
    :return dict:
    """
    return dict(
        n_estimators=[5, 25, 100],
        bootstrap_features=[False, True],
        bootstrap=[False, True])


def get_knn_ensemble_parameters():
    """
    Gets the parameters for the grid search testing.
    :return dict:
    """
    return dict()


def grid_parameter_estimation(classifier, score, x_train, y_train, grid_parameters):
    """
    Parameter estimation using grid search with cross-validation on SciKit-Learn
    :param classifier:
    :param str score:
    :param x_train:
    :param y_train:
    :param dict grid_parameters:
    """
    print("# Tuning hyper-parameters for {}.\n".format(score))

    clf = GridSearchCV(classifier, grid_parameters, scoring='%s_macro' % score)
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:\n{}".format(clf.best_params_))

    print("Grid scores on development set:\n")

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))


def arguments_definition():
    """
    Method for creating the possible parameters for execution.
    :return ArgumentParser:
    """
    parser = argparse.ArgumentParser(description='Runs the ensemble algorithms.')
    parser.add_argument('algorithm', type=str, choices=['knn', 'bagging'], help='The algorithm to be executed.')

    return parser.parse_args()


def run():
    args = arguments_definition()

    pandas_attribute_names = get_attribute_names()
    pandas_attributes = get_attributes()

    print(pandas_attribute_names)
    print(pandas_attributes)

    if args.algorithm == 'knn':
        classifier = KNeighborsClassifier()
        grid_parameters = get_knn_ensemble_parameters()

    elif args.algorithm == 'bagging':
        classifier = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
        grid_parameters = get_ensemble_grid_parameters()

    x_train, x_test, y_train, y_test = train_test_split(
        pandas_attribute_names, pandas_attributes.values.ravel(), test_size=0.5, random_state=0)

    scores = ['precision', 'recall']

    for score in scores:
        grid_parameter_estimation(classifier, score, x_train, y_train, grid_parameters)


if __name__ == '__main__':
    run()




