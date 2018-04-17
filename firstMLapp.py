import sys
import scipy
import numpy
import matplotlib
import sklearn
import pandas as pd

import matplotlib
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col_names = ['age', 'workclass', 'fnl-wgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race','sex', 'capital-gain', 'capital-loss', 'hrs/week', 'native-country', 'avg-income']

dataset = pd.read_csv(url, names=col_names)


for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
'''
dataset = pd.DataFrame(url, columns=col_names)
print(dataset.shape)
print(dataset.head())
print("------------------------------------------------------------------\n")
#print(dataset.describe())
array = dataset.values
print(dataset.shape, "\n")
'''

test_size = 0.2
seed = 7
array = dataset.values
X = array[:, 0:15]
Y = array[:, 14]

print(dataset.groupby('avg-income').size())

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=seed)


models=[]

models.append(('Logistic Regresion', LogisticRegression()))
models.append(('KNeighbors Classifier', KNeighborsClassifier()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))

results=[]
names=[]


for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f " %(name,cv_results.mean())
    print(msg)

knaan = DecisionTreeClassifier()
knaan.fit(X_train, Y_train)
predictions = knaan.predict(X_validation)

print("\n Accuracy : ",accuracy_score(Y_validation, predictions))

print("\n Report : ",classification_report(Y_validation, predictions))
