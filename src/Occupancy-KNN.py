# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:42:28 2018

@author: mshahbaz
"""

import numpy as np
import pandas as pd

df = pd.read_csv("Occupancy-test.csv")

X = df.iloc[:, 1:-1].values
y = df.iloc[:,-1].values       
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#splitting the data Training and Test
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)

#fitting k-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
classifier.fit(X_train,y_train)



#Predicting the test set results
y_pred = classifier.predict(X_test)

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix: \n", cm)

#Accuracy

print("Accuracy : ", classifier.score(X_test,y_test))
#K-Fold Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())


#accuracy
print(classifier.score(X_test,y_test))


