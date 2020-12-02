"""
Module for random_forest classifier.
"""

from sklearn.ensemble import RandomForestClassifier

from classifiers.abstract_classifier import AbstractClassifier


class RandomForest(AbstractClassifier):
    """Class for random_forest classifier."""
    
    def evaluate(self):
        """
        Evaluates the classifier.
        """
        classifier = RandomForestClassifier(
            max_depth=50,
            min_samples_leaf=2,
            min_samples_split=5)
        scores = self._build_scores(classifier)
        self._print_scores(scores)
    
    def grid_search(self, scores):
        """
        Runs the grid search of the classifier.
        :param list[str] scores:
        """
        classifier = RandomForestClassifier()
        for score in scores:
            self._perform_grid_search(classifier, score)

    def _grid_parameters(self):
        """
        Returns the grid parameters.
        :return dict:
        """
        return dict(
            max_depth=[10, 50, 100],
            min_samples_leaf=[1, 2, 4],
            min_samples_split=[2, 5, 10]
        )