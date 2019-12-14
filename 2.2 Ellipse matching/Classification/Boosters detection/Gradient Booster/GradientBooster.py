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

X_train, X_test, y_train, y_test = train_test_split(ellipses2, labels, test_size=0.3, random_state = 42)


lr_list = [0.8] # current best = 0.8, but one can add learning rates to this list to test several configurations
n_estimators = 600
max_features =  2
warm_start =  True

for learning_rate in lr_list:
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_features = max_features, max_depth=2, random_state=0, warm_start = warm_start)
    clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))
    print("\n")
    filename = 'GraDB' +  str(n_estimators) + str(max_features) + str(learning_rate) + str(warm_start) + str(clf.score(X_test, y_test)) + "%" + '.sav'
    if clf.score(X_test, y_test)>0.9:
        pickle.dump(clf, open(filename, 'wb'))


y_predict = clf.predict(X_test)
score = accuracy_score(y_test, y_predict)
print("n_estimators = ", n_estimators)
print("max_features = ", max_features)
print("warm_start = ", warm_start)
print(confusion_matrix(y_test, y_predict))
print("\n \n")
target_names = ['class 0', 'class 1', 'class 2', 'class 3']
print(classification_report(y_test, y_predict, target_names=target_names))

# print(confusion_matrix(y_test, y_predict))

filename = 'model.sav'
pickle.dump(clf, open(filename, 'wb'))
