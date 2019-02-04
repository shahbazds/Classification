# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 18:18:12 2018

@author: mshahbaz
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv("Occupancy-test.csv")

X = df.iloc[:, 1:-1].values
y = df.iloc[:,-1].values       

#splitting the data Training and Test
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train_sc =   sc.fit_transform(X_train)
#X_test_sc = sc.transform(X_test)

#SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,y_train)

#Pridiction
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : \n",cm)

#Score
print("Accuracry ; ", classifier.score(X_test,y_test))

#K-Fold Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())
