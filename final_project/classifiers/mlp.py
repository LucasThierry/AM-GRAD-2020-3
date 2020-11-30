"""
Module for mlp classifier.
"""

from classifiers.abstract_classifier import AbstractClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score



class MLP(AbstractClassifier):
    """Class for mlp classifier."""

    def evaluate(self):
        """
        Evaluates the classifier.
        """
        classifier = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(10, 4), learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=1)
        scores = self._build_scores(classifier)
        self._print_scores(scores)

    def grid_search(self, scores):
        """
        Runs the grid search of the classifier.
        :param list[str] scores:
        """
        classifier = MLPClassifier(activation='relu', alpha=1e-5, learning_rate_init=0.001, max_iter=200, random_state=1)
        for score in scores:
            self._perform_grid_search(classifier, score)


    def _grid_parameters(self):
        """
        Returns the grid parameters.
        :return dict:
        """
        return dict(solver=['lbfgs', 'sgd', 'adam'], learning_rate=['constant', 'invscaling', 'adaptive'], hidden_layer_sizes=[(5, 2), (10, 4), (20, 2), (50, 6)])

    def build_mat(self):
        classifier = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(10, 4), learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=1)
        mat = self._build_conf(classifier)
        print(mat)