"""
Module for ensemble_mlp classifier.
"""

from classifiers.abstract_classifier import AbstractClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier



class EnsembleMLP(AbstractClassifier):
    """Class for ensemble_mlp classifier."""
    def evaluate(self):
        """
        Evaluates the classifier.
        """
        classifier = BaggingClassifier(MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(10, 4), learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=1), max_samples=0.5, max_features=0.5, n_estimators=5, bootstrap=False) 
        scores = self._build_scores(classifier)
        self._print_scores(scores)

    def grid_search(self, scores):
        """
        Runs the grid search of the classifier.
        :param list[str] scores:
        """
        classifier = BaggingClassifier(MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(10, 4), learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=1), max_samples=0.5, max_features=0.5)
        for score in scores:
            self._perform_grid_search(classifier, score)

    def _grid_parameters(self):
        """
        Returns the grid parameters.
        :return dict:
        """
        return dict(
            n_estimators=[5, 15],
            bootstrap=[False, True])