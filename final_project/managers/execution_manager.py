"""
Module for the main manager.
"""

from sklearn.model_selection import train_test_split

from beans import PandasBean
from classifiers import KNN
from classifiers import Tree
from classifiers import RandomForest
from classifiers import MLP
from classifiers import EnsembleMLP
from classifiers import Ensemble
from managers.database_manager import DatabaseManager


class ExecutionManager:
    """Class for managing the project execution."""

    def __init__(self, database_path, database_base_name):
        """
        Class constructor.
        :param str database_path:
        :param str database_base_name:
        """
        self._dabase_manager = DatabaseManager(database_path, database_base_name)

    def run(self, algorithm, method):
        """
        Method for running the main function.
        :param str algorithm:
        :param str method:
        """
        print('Getting datasets.')
        pandas_training_set_train, pandas_training_set_test  = self._dabase_manager.get_training_set()
        pandas_test_set_train, pandas_test_set_test = self._dabase_manager.get_test_set()

        pandas_bean = self._build_pandas_bean(pandas_training_set_train, pandas_training_set_test, pandas_test_set_train, pandas_test_set_test)

        scores = ['precision', 'recall']

        print('Running classifier.')
        classifier = self._get_classifier(algorithm, pandas_bean, scores)
        self._run_classifier(method, classifier)

    @staticmethod
    def _build_pandas_bean(pandas_training_set_train, pandas_training_set_test, pandas_test_set_train, pandas_test_set_test):
        """
        Builds the pandas bean.
        :param pandas_training_set_train:
        :param pandas_training_set_test:
        :param pandas_test_set_train:
        :param pandas_test_set_test:
        """
        return PandasBean(pandas_training_set_train, pandas_training_set_test, pandas_test_set_train, pandas_test_set_test)

    @staticmethod
    def _get_classifier(algorithm, pandas_bean, scores):
        """
        Method for getting the classifier.
        :param str algorithm:
        :param PandasBean pandas_bean:
        :return Classifier:
        """
        if algorithm == 'knn':
            classifier = KNN(pandas_bean, scores)
        elif algorithm == 'tree':
            classifier = Tree(pandas_bean, scores)
        elif algorithm == 'forest':
            classifier = RandomForest(pandas_bean, scores)
        elif algorithm == 'mlp':
            classifier = MLP(pandas_bean, scores)
        elif algorithm == 'ensemble-mlp':
            classifier = EnsembleMLP(pandas_bean, scores)
        elif algorithm == 'ensemble':
            classifier = Ensemble(pandas_bean, scores)
        else:
            raise Exception('Invalid classifier: {}'.format(algorithm))
        return classifier

    @staticmethod
    def _run_classifier(method, classifier):
        """
        Method for running the classifier with the requested method.
        :param str method:
        :param Classifier classifier:
        """
        if method == 'evaluate':
            classifier.evaluate()
        elif method == 'grid_search':
            classifier.grid_search(['precision'])
        elif method == 'conf_mat':
            classifier.build_mat()
        else:
            raise Exception('Invalid method: {}'.format(method))