# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:07:42 2018

@author: mshahbaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("shuttle.txt",sep=" ",header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values       
#Splitting
#splitting the data Training and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#fitting Descision Tree
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0,max_depth=4)
classifier.fit(X_train_sc, y_train)

#Prediction
y_pred = classifier.predict(X_test_sc)

#confusion Matrix and Score
from sklearn.metrics import confusion_matrix

print(classifier.score(X_test_sc,y_test))
cm = confusion_matrix(y_test,y_pred)
print(cm)

#K-Fold Accuracy Check
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier,X,y,cv=10,scoring='accuracy')
print("Average K-Fold Accuracy: ", scores.mean())


