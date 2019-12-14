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
with open('BlurredCanny.pkl', 'rb') as g:
    ellipses, labels = pickle.load(g)

ellipses = ellipses.swapaxes(0,2)
ellipses2 = ellipses.reshape(5369,76800)

X_train, X_test, y_train, y_test = train_test_split(ellipses2, labels, test_size=0.3)


# estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
estimators = [100, 300, 500, 700, 1000]
max_depth = 30 #10
max_features = 100 # 12
warm_start = True

for n_estimators in estimators:

    clf =  RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=max_depth, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=max_features,
                                  max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=False, verbose=0,
                                  warm_start=warm_start, class_weight=None)
    clf.fit(X_train, y_train)

    print("n_estimators : ", n_estimators)
    print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))
    print("\n")

    filename = 'RF' + str(n_estimators) + str(max_depth) + str(max_features) + str(warm_start)  +str(clf.score(X_test, y_test)) + "%" + '.sav'
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
