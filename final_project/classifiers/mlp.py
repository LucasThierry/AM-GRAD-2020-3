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
        print(self.x_train)
        print(self.y_train)
        skf = StratifiedKFold(n_splits=10)
        classifier = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(5, 2), learning_rate_init=0.001, max_iter=200, random_state=1)
        scores = cross_val_score(classifier, self.x_train, self.y_train, scoring='accuracy', cv=skf)
        self._print_scores(scores)

    def grid_search(self):
        pass

    def _get_attributes(self):
        pass