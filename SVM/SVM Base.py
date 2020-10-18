import pandas
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

filename = 'glass - atributos.csv' # Features
names = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba','Fe']
X = pandas.read_csv(filename, names=names)
print(X)

filename1 = 'glass - classes.csv' # Classes
names1 = ['Type of glass']
Y = pandas.read_csv(filename1, names=names1)
print(Y)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1453) # Parametros de Validação
clf = svm.SVC(kernel='linear', C = 1.0) # Parametros do Modelo

scores = cross_val_score(clf, X, Y.values.ravel(), scoring='accuracy', cv=cv) # Avaliação
print("Accuracy: %.3f (%.3f)" % (scores.mean(), scores.std()))