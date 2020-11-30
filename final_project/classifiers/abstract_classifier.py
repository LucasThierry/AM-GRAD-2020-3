"""
Module for abstract classifier.
"""

from abc import ABC
from abc import abstractclassmethod

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


class AbstractClassifier(ABC):
    """Class for abstract classifier."""

    def __init__(self, pandas_bean, scores):
        """
        :param beans.pandas_bean.PandasBean pandas_bean:
        :param list[str] scores:
        """
        self._pandas_bean = pandas_bean
        self._scores = scores
    
    @abstractclassmethod
    def grid_search(self, scores):
        """
        Runs the grid search of the classifier.
        :param list[str] scores:
        """
        pass

    @abstractclassmethod
    def evaluate(self):
        """
        Evaluates the classifier.
        """
        pass

    @abstractclassmethod
    def _grid_parameters(self):
        """
        Returns the grid parameters.
        :return dict:
        """
        pass

    @staticmethod
    def _print_scores(scores):
        """
        Prints the scores of the classifier.
        :param cross_val_score scores:
        """
        print("Accuracy: %.3f (%.3f)" % (scores.mean(), scores.std()))

    def _perform_grid_search(self, classifier, score):
        """
        Performs a grid search on the classifier.
        :param Classifier classifier: the classifier instance.
        :param str score:
        """
        print("# Tuning hyper-parameters for {}.\n".format(score))

        grid_parameters = self._grid_parameters()

        clf = GridSearchCV(classifier, grid_parameters, scoring='%s_macro' % score)
        clf.fit(self._pandas_bean.x_train, self._pandas_bean.y_train)
        print("Best parameters set found on development set:\n{}".format(clf.best_params_))

        print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))

    def _build_scores(self, classifier):
        """
        Buils the scores:
        :param classifier: A sklearn classifier.
        :return cross_val_score:
        """
        skf = StratifiedKFold(n_splits=10)
        return cross_val_score(classifier, self._pandas_bean.x_train, self._pandas_bean.y_train, scoring='accuracy', cv=skf)

    def _build_conf(self, classifier):
        """
        Buils the matrix
        """
        skf = StratifiedKFold(n_splits=10)
        y_pred = cross_val_predict(classifier, self._pandas_bean.x_train, self._pandas_bean.y_train, cv=skf)
        return confusion_matrix(self._pandas_bean.y_train, y_pred)
        


