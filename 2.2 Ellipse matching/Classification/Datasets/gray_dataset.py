""" This code generates the dataset for the count of ellipses. This one is then used in random forest
for classification. This is done in 2 parts, first for images with ellipses and then for the images
with no ellipses. See report""" 

import numpy as np
import csv
import glob
import cv2
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

### 1. Ellipses ##

path = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\CV\\Dataset\\Ellipses.csv' # Only the ellipses lines
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    list_images = []
    for row in csv_reader: 
        list_images.append(row[0])
number_ellipses = Counter(list_images) # Count of the ellipses in images
list_ellipses = list(number_ellipses)
labels = np.fromiter(number_ellipses.values(), dtype=int)
 
files = sorted(glob.glob('C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\Ellipses\\*')) # Folder with only image lines
dim = (320, 240) 
	
del files[0] # Thumb file

first = True

for file in files:
	gray_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	if first == True:
		resized = cv2.resize(gray_image, dim)
		ellipses = resized
		first = False
	else :
		resized = cv2.resize(gray_image, dim)
		ellipses = np.dstack((ellipses,resized))

### 2. No Ellipses ###
	
	
files = sorted(glob.glob('C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\NoEllipses\\*'))
dim = (320, 240)
	
del files[0] # Thumb file
labelsNoEllipses = np.zeros(len(files))

first = True

for file in files:
	gray_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	if first == True:
		resized = cv2.resize(gray_image, dim)
		noEllipses = resized
		first = False
	else :
		resized = cv2.resize(gray_image, dim)
		noEllipses = np.dstack((noEllipses,resized))
	
images = np.dstack((ellipses, noEllipses))
labels_images = np.concatenate((labels, labelsNoEllipses), axis=0)

with open('dataset_resized.pkl', 'wb') as g:
    pickle.dump([images, labels_images], g)