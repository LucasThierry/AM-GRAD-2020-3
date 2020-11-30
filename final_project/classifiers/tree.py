"""
Module for decision tree classifier.
"""

from sklearn.tree import DecisionTreeClassifier

from classifiers.abstract_classifier import AbstractClassifier


class Tree(AbstractClassifier):
    """Class for decision tree classifier."""
    
    def evaluate(self):
        """
        Evaluates the classifier.
        """
        classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, splitter='random')
        scores = self._build_scores(classifier)
        self._print_scores(scores)
    
    def grid_search(self, scores):
        """
        Runs the grid search of the classifier.
        :param list[str] scores:
        """
        classifier = DecisionTreeClassifier()
        for score in scores:
            self._perform_grid_search(classifier, score)

    def build_mat(self):
        """
        Builds the confusion matrix.
        """
        classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, splitter='random')
        mat = self._build_conf(classifier)
        print(mat)

    def _grid_parameters(self):
        """
        Returns the grid parameters.
        :return dict:
        """
        return dict(splitter=['best', 'random'], min_samples_leaf=[1, 2, 3, 5, 10], max_depth=[None, 10, 50, 100])