"""
This code loads a data set, currently set to eyes_dataset.pkl, and load its images as well as
2  pre trained models, already saved in the same file, one which predicts all parameters of the
ellipses present in the image, and another one predicting only the angle of the ellipse.
It then computes the errors on a testing set, which is seeded as 42, just like the models built
 in other code files to avoid bias in the tests.

"""
import pickle
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import multioutput
from math import sqrt


def getAngles(Yval):
    angles = []
    for y in Yval:
        angle = y[4]
        angles.append(angle)
    return angles

n_estimator = 100

with open('eyes_dataset.pkl', 'rb') as g:
    images, parameters = pickle.load(g)

parameters_regressor = []
for params in parameters :
    (x, y), (MA, ma), angle = params
    parameters_regressor.append([x,y,MA,ma,angle])

X_train, X_test, y_train, y_test = train_test_split(images, parameters_regressor, test_size=0.3, random_state = 42)

filename = 'regression_seed42_' + str(n_estimator)  + '_estimators.sav'
filename_angles =  'regression_seed42_gradientboost300_estimators_9.996816865248428.sav'

clf = pickle.load(open(filename, 'rb'))
clf_angles = pickle.load(open(filename_angles, 'rb'))

y_predict = clf.predict(X_test)
y_predict_angles = clf_angles.predict(X_test)

y_test_angles = getAngles(y_test)

iterator = np.arange(0,len(y_predict),1)

Xdif = 0
Ydif = 0
MAdif = 0
madif = 0
angledif = 0
count = 0

for value in iterator:
    count += 1

    #predictions
    x = y_predict[value][0]
    y = y_predict[value][1]
    MA = y_predict[value][2]
    ma = y_predict[value][3]
    angle = y_predict_angles[value]

    #real values
    xReal = y_test[value][0]
    yReal = y_test[value][1]
    MAReal = y_test[value][2]
    maReal = y_test[value][3]
    angleReal = y_test_angles[value]

    # computing sum of errors
    Xdif += sqrt( (x - xReal)**2)
    Ydif += sqrt( (y - yReal)**2)
    MAdif += sqrt( (MA - MAReal)**2)
    madif += sqrt( (ma - maReal)**2)
    angledif += sqrt( (angle - angleReal)**2)

    param = (x, y), (MA, ma), angle
    img = X_test[value]
    img = img.reshape(240,320)
    image = cv2.ellipse(img, param, [255,0,0])
    path = 'results/1_sellipse_from_regression_' + str(value) + '.png'
    cv2.imwrite(path, image)

# compute average
Xdif /= count
Ydif /= count
MAdif /= count
madif /= count
angledif /= count

print("\n  REPORT OF ERRORS FOR RAND FOREST ", n_estimator, " estimators")
print("Xdif mean squared error :", Xdif)
print("Ydif mean squared error :", Ydif)
print("MAdif mean squared error :", MAdif)
print("madif mean squared error :", madif)
print("angledif mean squared error :", angledif)
