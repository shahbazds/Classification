# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 13:04:19 2018

@author: Muhammad Shahbaz
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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=3)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#confusion Matrix and Score
from sklearn.metrics import confusion_matrix

print(classifier.score(X_test,y_test))
cm = confusion_matrix(y_test,y_pred)
print("confusion Matrix : \n",cm)

#Accuracy
print("Accuracy : ", classifier.score(X_test, y_test))

#K-Fold Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())

