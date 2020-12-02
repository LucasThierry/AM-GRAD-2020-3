"""
Final project main script.
"""

import argparse
import os

from managers import ExecutionManager


def arguments_definition():
    """
    Method for creating the possible parameters for execution.
    :return ArgumentParser:
    """
    parser = argparse.ArgumentParser(description='Runs the algorithms.')
    parser.add_argument(
        'algorithm',
        type=str,
        choices=['knn', 'tree', 'forest', 'mlp', 'ensemble-mlp', 'ensemble'],
        help='The algorithm to be executed.')
    parser.add_argument(
        'method',
        type=str,
        choices=['evaluate', 'grid_search','conf_mat'],
        help='The method to be executed.')
    parser.add_argument(
        '--database_path',
        default=os.path.join('database', 'split'),
        type=str,
        help='Path to the database (Default is database/split).')
    parser.add_argument(
        '--database_base_name',
        default='aps_failure',
        type=str,
        help='The base name of the database (Default is aps_failure).')

    return parser.parse_args()


if __name__ == '__main__':
    """
    Main runner.
    """
    args = arguments_definition()

    algorithm = args.algorithm
    method = args.method

    database_path = args.database_path
    database_base_name = args.database_base_name

    exectution_manager = ExecutionManager(database_path, database_base_name)
    exectution_manager.run(algorithm, method)
