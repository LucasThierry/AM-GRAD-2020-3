"""
Module for knn classifier.
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


from classifiers.abstract_classifier import AbstractClassifier


class KNN(AbstractClassifier):
    """Class for knn classifier."""

    def grid_search(self):
        """
        Performs a grid search on the classifier
        """
        pass

    def evaluate(self):
        """
        Evaluates the classifier.
        """
        classifier = KNeighborsClassifier()
        skf = StratifiedKFold(n_splits=10)
        scores = cross_val_score(classifier, self.x_train, self.y_train, scoring='accuracy', cv=skf)
        self._print_scores(scores)

    def _get_attributes(self):
        """
        Returns the attributes.
        :return dict:
        """
        pass