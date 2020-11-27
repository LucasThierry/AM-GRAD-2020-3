import os
import argparse

import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
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


def get_tree_ensemble_parameters():
    """
    Gets the parameters for the grid search testing with tree.
    :return dict:
    """
    return dict()


def get_bagging_grid_parameters():
    """
    Gets the parameters for the grid search testing with bagging.
    :return dict:
    """
    return dict(
        n_estimators=[5, 25, 100],
        bootstrap_features=[False, True],
        bootstrap=[False, True])


def get_boosting_grid_parameters():
    """
    Gets the parameters for the grid search testing with boosting.
    :return dict:
    """
    return dict(
        n_estimators=[5, 25, 100],
        learning_rate=[0.5, 1, 2])


def get_stacking_grid_parameters():
    """
    Gets the parameters for the grid search testing with stacking.
    :return dict:
    """
    return dict(
        stack_method=['auto', 'predict_proba', 'decision_function', 'predict'],
        passthrough=[False, True])


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
    parser.add_argument(
        'algorithm',
        type=str,
        choices=['tree', 'bagging', 'boosting', 'stacking'],
        help='The algorithm to be executed.')

    return parser.parse_args()


def run():
    args = arguments_definition()

    pandas_attribute_names = get_attribute_names()
    pandas_attributes = get_attributes()

    print(pandas_attribute_names)
    print(pandas_attributes)

    if args.algorithm == 'tree':
        classifier = DecisionTreeClassifier()
        grid_parameters = get_tree_ensemble_parameters()

    elif args.algorithm == 'bagging':
        classifier = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
        grid_parameters = get_bagging_grid_parameters()

    elif args.algorithm == 'boosting':
        classifier = AdaBoostClassifier(DecisionTreeClassifier())
        grid_parameters = get_boosting_grid_parameters()

    elif args.algorithm == 'stacking':
        classifier = StackingClassifier(estimators=[('dt', DecisionTreeClassifier())]
        grid_parameters = get_stacking_grid_parameters()

    x_train, x_test, y_train, y_test = train_test_split(
        pandas_attribute_names, pandas_attributes.values.ravel(), test_size=0.5, random_state=0)

    scores = ['precision', 'recall']

    for score in scores:
        grid_parameter_estimation(classifier, score, x_train, y_train, grid_parameters)


if __name__ == '__main__':
    run()




