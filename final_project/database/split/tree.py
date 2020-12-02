import pandas
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

filename = 'aps_failure_training_set_attributes.csv'
X = pandas.read_csv(filename)

filename1 = 'aps_failure_training_set_classes.csv'
Y = pandas.read_csv(filename1)


classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, splitter='random')

classifier.fit(X, Y)
