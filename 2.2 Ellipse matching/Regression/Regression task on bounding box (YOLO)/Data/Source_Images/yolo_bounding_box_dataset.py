import numpy
import cv2
from os import walk
from shutil import copy

path = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\Soccer'
data = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Dataset\\CV2019_Annots_ElpsSoccer.csv'
    
lines = []
with open(data,'r') as file:
    lines = file.readlines()

i = 0
while i < len(lines) :
    if lines[i].find("elps_soccer") == -1:
        del lines[i]
    else:
        i = i +1
j = 0
"""while j < len(lines):
    sub = lines[j].split(',')
    img = cv2.imread(path + '\\' + sub[0])
    
    if isinstance(img,numpy.ndarray):
        j = j+1
    else:
        del lines[j]"""
        
points = numpy.zeros((len(lines),4))
total = []
j = 0
previous = ""

with open('Training_Images/vott-csv-export/Annotations-export.csv','a') as file:
    file.write('"image","xmin","ymin","xmax","ymax","label"\n')
     
    
for j in range(len(lines)):
    sub = lines[j].split(',')
    x = sub[3:len(sub):2]
    y = sub[4:len(sub):2]
    img = cv2.imread(path + '\\' + sub[0])
    original = img.shape   
    for k in range(len(x)):
        x[k]= float(x[k])
        y[k]= float(y[k])
        y[k] = original[0] - y[k]
    
    with open('Training_Images/vott-csv-export/Annotations-export.csv','a') as file:
        file.write('"' + sub[0] + '"')
        file.write(',%d,%d,%d,%d,"Ellipse"\n' % (min(x),min(y),max(x),max(y)))