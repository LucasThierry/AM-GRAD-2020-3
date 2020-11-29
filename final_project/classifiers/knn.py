"""
Module for knn classifier.
"""

from sklearn.neighbors import KNeighborsClassifier


from classifiers.abstract_classifier import AbstractClassifier


class KNN(AbstractClassifier):
    """Class for knn classifier."""

    def evaluate(self):
        """
        Evaluates the classifier.
        """
        classifier = KNeighborsClassifier(algorithm='auto', n_neighbors='7', weights='distance')
        scores = self._build_scores(classifier)
        self._print_scores(scores)
    
    def grid_search(self, scores):
        """
        Runs the grid search of the classifier.
        :param list[str] scores:
        """
        classifier = KNeighborsClassifier()
        for score in scores:
            self._perform_grid_search(classifier, score)


    def _grid_parameters(self):
        """
        Returns the grid parameters.
        :return dict:
        """
        return dict(n_neighbors=[3, 5, 7], weights=['uniform', 'distance'], algorithm=['auto', 'brute'])