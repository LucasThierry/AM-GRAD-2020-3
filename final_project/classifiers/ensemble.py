"""
Module for ensemble classifier.
"""

from classifiers.abstract_classifier import AbstractClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score



class Ensemble(AbstractClassifier):
    """Class for ensemble classifier."""

    def evaluate(self):
        """
        Evaluates the classifier.
        """
        classifier = StackingClassifier(estimators=[('mlp-nn', MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(10, 4), learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=1))], 
                stack_method='auto',
                passthrough=False)
        scores = self._build_scores(classifier)
        self._print_scores(scores)

    def grid_search(self, scores):
        """
        Runs the grid search of the classifier.
        :param list[str] scores:
        """
        classifier = BaggingClassifier(estimators=[('mlp-nn', MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(10, 4), learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=1))])
        for score in scores:
            self._perform_grid_search(classifier, score)


    def _grid_parameters(self):
        """
        Returns the grid parameters.
        :return dict:
        """
        return dict(
                stack_method=['auto', 'predict_proba', 'decision_function', 'predict'],
                passthrough=[False, True])