import os

import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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

    classifier = MLPClassifier(activation='identity', alpha=1e-5, max_iter= 500, random_state=1) ## Parametros base

    X_train, X_test, y_train, y_test = train_test_split(
        pandas_attribute_names, pandas_attributes.values.ravel(), test_size=0.5, random_state=0)

    tuned_parameters = {'solver': ['lbfgs', 'sgd'], ##                                Grade de parâmetros
                        'learning_rate': ['constant', 'invscaling', 'adaptive'], 
                        'hidden_layer_sizes': [(5, 2), (10, 4), (20, 2), (50, 6)], 
    }

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score) ## Parameter estimation using grid search with cross-validation on SciKit-Learn
    print()

    clf = GridSearchCV(
        classifier, tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


