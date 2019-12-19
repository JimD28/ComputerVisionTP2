""" This code generates the dataset for the eyes. This one is then used in random forest
for regression""" 

import pickle
import numpy as np
import cv2

# The first path is for the folder with eyes images. The second one for the annotations
copyPath = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\Eyes'
data = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\CV2019_Annots.csv'

# Read data file and preprocess : keep only correct matches between image folder and data file   
lines = []
with open(data,'r') as file:
    lines = file.readlines()

i = 0
while i < len(lines) :
    if lines[i].find("elps_eye") == -1:
        del lines[i]
    else:
        i = i +1
j = 0
while j < len(lines):
    sub = lines[j].split(',')
    img = cv2.imread(copyPath + '\\' + sub[0], cv2.IMREAD_GRAYSCALE)
    
    if isinstance(img,np.ndarray):
        j = j+1
    else:
        del lines[j]

## Isolate the coordinates from data file        
j = 0
tmp = []
parameters = []
images = []
for j in range(len(lines)):
    sub = lines[j].split(',')
    x = sub[2:len(sub):2]
    y = sub[3:len(sub):2]
    for k in range(len(x)):
        x[k]= float(x[k])
        y[k]= float(y[k])
    
    img = cv2.imread(copyPath + '\\' + sub[0], cv2.IMREAD_GRAYSCALE)
    img = img.reshape(76800)
    images.append(img)
    
    # Convert coordinates x-y system axis to row-column system axis
    var = x
    x = 240 - np.array(y)
    y = np.array(var)
    
    coordinates = np.stack((y, x))
    coordinates = np.transpose(coordinates)
    coordinates = coordinates.astype(int)
    tmp = cv2.fitEllipse(coordinates) # Obtain parameters from ellipse points
    parameters.append(tmp)

with open('eyes_dataset.pkl', 'wb') as g:  # Creation of the dataset
    pickle.dump([images, parameters], g)