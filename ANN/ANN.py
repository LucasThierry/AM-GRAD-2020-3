import os

import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


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


def print_results(results):
    """
    Prints the results.
    :param Score results:
    """
    print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


if __name__ == '__main__':

    pandas_attribute_names = get_attribute_names()
    pandas_attributes = get_attributes()

    print(pandas_attribute_names)
    print(pandas_attributes)

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1453)

    classifier = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(5, 2), learning_rate_init=0.001, max_iter=200, random_state=1)

    scores = cross_val_score(classifier, pandas_attribute_names, pandas_attributes.values.ravel(), scoring='accuracy', cv=cv)
    print_results(scores)
