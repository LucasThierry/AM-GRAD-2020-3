import os

import pandas
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def get_attribute_names():
    """
    Gets the names of the attributes of the database.
    """
    filepath = os.path.join('..','database', 'glass_attributes_processed.csv')
    names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba','Fe']
    return pandas.read_csv(filepath, names=names)


def get_attributes():
    """
    Gets the values of the attributes of the database.
    """
    filepath = os.path.join('database', 'glass_classes.csv')
    names = ['Type of glass']
    return pandas.read_csv(filepath, names=names)


def get_testing_parameters():
    """
    Gets the parameters for the gridsearch testing.
    :return dict:
    """
    return dict(n_estimators=[3, 5, 10, 15, 25])

def print_results(results):
    """
    Prints the results.
    :param Score results:
    """
    print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


def grid_parameter_estimation(classifier, score, x_train, y_train, tuned_parameters):
    """
    Parameter estimation using grid search with cross-validation on SciKit-Learn
    :param classifier:
    :param str score:
    :param x_train:
    :param y_train:
    :param dict tuned_parameters:
    """
    print("# Tuning hyper-parameters for {}.\n".format(score))

    clf = GridSearchCV(classifier, tuned_parameters, scoring='%s_macro' % score)
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:\n{}".format(clf.best_params_))

    print("Grid scores on development set:\n")

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))


def run():
    pandas_attribute_names = get_attribute_names()
    pandas_attributes = get_attributes()

    print(pandas_attribute_names)
    print(pandas_attributes)

    classifier = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

    x_train, x_test, y_train, y_test = train_test_split(
        pandas_attribute_names, pandas_attributes.values.ravel(), test_size=0.5, random_state=0)

    tuned_parameters = get_testing_parameters()

    scores = ['precision', 'recall']

    for score in scores:
        grid_parameter_estimation(classifier, score, x_train, y_train, tuned_parameters)


if __name__ == '__main__':
    run()




