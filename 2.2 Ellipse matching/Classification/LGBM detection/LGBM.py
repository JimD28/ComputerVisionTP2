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
from sklearn.ensemble import HistGradientBoostingClassifier

# Getting back the objects:
with open('dataset_resized.pkl', 'rb') as g:
    ellipses, labels = pickle.load(g)

ellipses = ellipses.swapaxes(0,2)
ellipses2 = ellipses.reshape(5369,76800)

X_train, X_test, y_train, y_test = train_test_split(ellipses2, labels, test_size=0.3)


estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3]
max_depth = 2
max_features = 2
warm_start = False

for n_estimators in estimators:
    for lr in learning_rates:
        clf = HistGradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=n_estimators,
                                             criterion='friedman_mse', max_depth=max_depth,random_state=None,
                                             max_features=max_features, verbose=0, warm_start=warm_start,
                                              presort='deprecated')
        clf.fit(X_train, y_train)

        print("n_estimators : ", n_estimators)
        print("learning rate  :", lr)
        print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))
        print("\n")

        filename = 'LGBM' + str(n_estimators) + str(lr) +str(max_depth) + str(max_features) + str(warm_start)  +str(clf.score(X_test, y_test)) + "%" + '.sav'
        if clf.score(X_test, y_test)>0.93:
            pickle.dump(clf, open(filename, 'wb'))

y_predict = clf.predict(X_test)
score = accuracy_score(y_test, y_predict)
print("n_estimators = ", n_estimators)
print("max_features = ", max_features)
print("warm_start = ", warm_start)
print(score)
print(confusion_matrix(y_test, y_predict))

filename = 'model.sav'
pickle.dump(clf, open(filename, 'wb'))
