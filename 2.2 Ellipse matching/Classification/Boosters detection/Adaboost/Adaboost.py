import csv
import glob
import cv2
import numpy as np
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Getting back the objects:
with open('dataset_resized.pkl', 'rb') as g:
    ellipses, labels = pickle.load(g)

ellipses = ellipses.swapaxes(0,2)
ellipses2 = ellipses.reshape(5369,76800)

X_train, X_test, y_train, y_test = train_test_split(ellipses2, labels, test_size=0.3)

n_estimators = 70
lr = 1


clf = AdaBoostClassifier( n_estimators=n_estimators, learning_rate=lr, random_state=None)
clf.fit(X_train, y_train)

print("Learning rate: ", lr)
print("estimators :", n_estimators)
print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))
print("\n")
filename = 'AdaB' + str(n_estimators)  + str(lr)  +str(clf.score(X_test, y_test)) + "%" + '.sav'
if clf.score(X_test, y_test)>0.93:
    pickle.dump(clf, open(filename, 'wb'))

#clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(score)
print(confusion_matrix(y_test, y_predict))
