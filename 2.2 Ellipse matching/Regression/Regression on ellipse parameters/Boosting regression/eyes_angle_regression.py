"""
This code loads a data set, currently set to eyes_dataset.pkl, and load its images to train
a gradient boosting model on the angle parameters of the ellipses contained in said images. It then saves the model
under a file name made of the main parameters of the model and computes its accuracy on
a testing set previously made and seeded to 42.
"""

import pickle
import cv2
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import multioutput
from math import sqrt

n_estimator = 300


def getAngles(Yval):
    angles = []
    for y in Yval:
        angle = y[4]
        angles.append(angle)
    return angles

with open('eyes_dataset.pkl', 'rb') as g:
    images, parameters = pickle.load(g)

parameters_regressor = []
for params in parameters :
    (x, y), (MA, ma), angle = params
    parameters_regressor.append([x,y,MA,ma,angle])
X_train, X_test, y_train, y_test = train_test_split(images, parameters_regressor, test_size=0.001, random_state = 42)

y_train_angles = getAngles(y_train)
y_test_angles = getAngles(y_test)


clf = GradientBoostingRegressor(loss='ls', learning_rate=0.5, n_estimators=n_estimator, verbose=2, warm_start=True)
clf.fit(X_train, y_train_angles)

filename = 'regression_seed42_gradientboost' + str(n_estimator)  + '_estimators_10.sav'
#pickle.dump(clf, open(filename, 'wb'))

clf = pickle.load(open(filename, 'rb'))

y_predict = clf.predict(X_test)
iterator = np.arange(0,len(y_predict),1)

angledif = 0
count = 0

for value in iterator:
    count += 1

    #predictions
    angle = y_predict[value]

    #real values
    angleReal = y_test_angles[value]

    # computing sum of errors
    angledif += sqrt( (angle - angleReal)**2)

# compute average
angledif /= count

filename = 'regression_seed42_gradientboost' + str(n_estimator)  + '_estimators_' + str(angledif) +'.sav'
pickle.dump(clf, open(filename, 'wb'))

print("\n  REPORT OF ERRORS FOR gradient boosting regressor ", n_estimator, " estimators")
print("angledif mean squared error :", angledif)
