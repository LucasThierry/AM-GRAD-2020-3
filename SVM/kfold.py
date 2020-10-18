import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

filename = 'glass - atributos.csv'
names = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba','Fe']
X = pandas.read_csv(filename, names=names)
print(X)

filename1 = 'glass - classes.csv'
names1 = ['Type of glass']
Y = pandas.read_csv(filename1, names=names1)
print(Y)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1453)
clf = svm.SVC(kernel='linear', C = 1.0)

rkf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=2652124)
for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    clf.fit(X.iloc[train_index], Y.iloc[train_index].values.ravel())
    print(clf.predict(X.iloc[test_index]))

#scores = cross_val_score(clf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#print(clf.predict([[1,152.101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00]]))
#print(clf.predict([[22214,151.711,14.23,0.00,02.08,73.36,0.00,8.62,1.67,0.00]]))
