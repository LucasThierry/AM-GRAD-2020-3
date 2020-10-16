import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

filename = 'glass - atributos.csv'
names = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba','Fe']
X = pandas.read_csv(filename, names=names)
print(X)

filename1 = 'glass - classes.csv'
names1 = ['Type of glass']
Y = pandas.read_csv(filename1, names=names1)
print(Y)

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,Y.values.ravel())

print(clf.predict([[1,152.101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00]]))
print(clf.predict([[22214,151.711,14.23,0.00,02.08,73.36,0.00,8.62,1.67,0.00]]))
