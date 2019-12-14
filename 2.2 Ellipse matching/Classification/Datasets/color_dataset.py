""" This code generates the dataset for the count of ellipses. This one is then used in ResNet
for classification. This is done in 2 parts, first for images with ellipses and then for the images
with no ellipses. See report""" 

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

### 1. Ellipses ##

path = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\CV\\Dataset\\Ellipses.csv'
with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    list_images = []
    for row in csv_reader: 
        list_images.append(row[0])
number_ellipses = Counter(list_images) # Count of the ellipses
list_ellipses = list(number_ellipses)
labels = np.fromiter(number_ellipses.values(), dtype=int)
 
files = sorted(glob.glob('C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\Ellipses\\*'))
dim = (320, 240)
	
del files[0] # Thumb file

first = True

print("here")

for file in files:
	image = cv2.imread(file)
	if first == True:
		resized = cv2.resize(image, dim)
		resized = resized[None,:,:,:]
		ellipses = resized
		first = False
	else :
		resized = cv2.resize(image, dim)
		resized = resized[None,:,:,:]
		ellipses = np.concatenate((ellipses,resized))
		
### 2. No Ellipses ###
	
print("here")	
files = sorted(glob.glob('C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\NoEllipses\\*'))
dim = (320, 240)
	
del files[0] # Thumb file
labelsNoEllipses = np.zeros(len(files))

first = True

for file in files:
	image = cv2.imread(file)
	if first == True:
		resized = cv2.resize(image, dim)
		resized = resized[None,:,:,:]
		noEllipses = resized
		first = False
	else :
		resized = cv2.resize(image, dim)
		resized = resized[None,:,:,:]
		noEllipses = np.concatenate((noEllipses,resized))
		
print("here")	
		

## 3. Mix dataset

# Getting back the objects:
	
images = np.concatenate((ellipses, noEllipses))
labels_images = np.concatenate((labels, labelsNoEllipses), axis=0)

with open('dataset_color.pkl', 'wb') as g: 
    pickle.dump([images, labels_images], g)