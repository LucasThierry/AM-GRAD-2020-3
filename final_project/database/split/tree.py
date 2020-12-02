import pandas
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

train1 = 'aps_failure_training_set_attributes.csv'
Xp = pandas.read_csv(train1)
X=Xp.fillna(Xp.mean())


train2 = 'aps_failure_training_set_classes.csv'
Y = pandas.read_csv(train2)


test1 = 'aps_failure_test_set_attributes.csv'
Xtestp = pandas.read_csv(test1)
Xtest=Xtestp.fillna(Xtestp.mean())


test2 = 'aps_failure_test_set_classes.csv'
Ytest = pandas.read_csv(test2)

classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, splitter='random')

classifier.fit(X, Y)

print(classifier.predict(Xtest))