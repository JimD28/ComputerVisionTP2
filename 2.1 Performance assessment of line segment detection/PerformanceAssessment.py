from itertools import chain
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import tools
import csv
directory = ''

dictResults = {}
def removeDuplicates(lst):
    return list(set(i for i in lst))
listName = []

# C:\\Users\\15143\\OneDrive\\Bureau\\Images+CSV\\Team03\\annot_road.csv
with open('annot_road.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            listName.append(row[0])
            if(row[0] == 'end'):
                break


sumTP = 0
sumTN = 0
sumFP = 0
sumFN = 0
totalImage = 0


listNoDupNames = removeDuplicates(listName)
f=0
for name in listNoDupNames:
    totalImage = totalImage + 1

    monSudoku = name
    filePath = 'C:/Users/15143/OneDrive/Bureau/Images+CSV/Team03/Road/' + name
    print(filePath)
    imageControl = cv2.imread(filePath)

    imageHough = cv2.imread(filePath)    # Image hough
    dimensions = imageControl.shape

    imageGray = cv2.cvtColor(imageControl, cv2.COLOR_BGR2GRAY)  # Image en gris

    blank_image = np.zeros(shape = [dimensions[0], dimensions[1], 3], dtype=np.uint8)#255 * np.ones(shape = [dimensions[0], dimensions[1], 3], dtype=np.uint8)
    blank_image2 = np.zeros(shape=[dimensions[0], dimensions[1], 3], dtype=np.uint8)


    omega = 10
    sigma_color = 20
    sigma_space = 50
    imgBilat = cv2.bilateralFilter(imageGray, omega, sigma_color, sigma_space)
    med = np.median(imageGray)
    sigma = 0.3
    loThreshold = 30
    hiThreshold = 90

    edges = cv2.Canny(imgBilat, loThreshold, hiThreshold,
                    apertureSize=3, L2gradient=False)
    kernel = np.ones((10, 10), np.uint8)
    edges1 = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((11, 11), np.uint8)
    edges2 = cv2.erode(edges1, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges2, 1, np.pi / 360,
                            threshold=67, minLineLength=70, maxLineGap=2)

#C:\\Users\\15143\\OneDrive\\Bureau\\Images+CSV\\Team03\\annot_road.csv
    with open('annot_road.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if(row[0] == monSudoku):
                    #listeSudoku.append((row[0], round(float(row[2]), round(float(row[3])), round(float(row[4])), round(float(row[5]))))
                    x1 = round(float(row[2]))
                    y1 = round(float(row[3]))
                    x2 = round(float(row[4]))
                    y2 = round(float(row[5]))
                    cv2.line(
                        blank_image, (x1, dimensions[0] - y1), (x2, dimensions[0] - y2), (100, 100, 100), 7)
                    line_count += 1
                    if(row[0] != monSudoku):
                        break






    for line in lines:
        x1, y1, x2, y2 = line[0]

        cv2.line(blank_image2, (x1, y1), (x2, y2), (155,155,155), 7)

    imageFinale = cv2.add(blank_image, blank_image2)

    listeHough = []
    listeTruthAndHough = []
    listeTruth = []
    listeZeroes = []

    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            b, g, r = imageFinale[x, y]
            if(b == 155):
                listeHough.append((x,y))
            if(b == 255):
                listeTruthAndHough.append((x,y)) 
            if(b == 100):
                listeTruth.append((x,y))
            if(b == 0):
                listeZeroes.append((x,y))       

    totPixelsHough = len(listeHough)
    totPixelsTruth = len(listeTruth)
    totPixelsHoughAndTruth = len(listeTruthAndHough)
    totPixelsZeroes = len(listeZeroes)
    totPixels = dimensions[0] * dimensions[1]


    #print('Liste hough seulement : ', totPixelsHough)
    #print('Liste truth seulement : ', totPixelsTruth)
    #print('Liste truth et hough : ', totPixelsHoughAndTruth)

    truePositive = totPixelsHoughAndTruth / totPixels #(totPixelsTruth + totPixelsHoughAndTruth)
    falsePositive = totPixelsHough /  totPixels# (totPixelsTruth + totPixelsHoughAndTruth)
    falseNegative = totPixelsTruth / totPixels#(totPixelsTruth + totPixelsHoughAndTruth)
    trueNegative = totPixelsZeroes / totPixels

#    tools.multiPlot(1, 1, (blank_image2, blank_image, imageFinale), ('Hough',
#                                                                     'Ground truth', 'Hough+GroundTruth'), cmap_tuple=(cm.gray, cm.gray, cm.gray))


    print(name)
    print('True Positive : ', truePositive * 100)
    print('False Positive : ', falsePositive * 100)
    print('True negative : ', trueNegative * 100)
    print('False negative : ', falseNegative * 100)

    sumTP = sumTP + (truePositive * 100)
    sumTN = sumTN + (trueNegative * 100)
    sumFN = sumFN + (falseNegative * 100)
    sumFP = sumFP + (falsePositive * 100)

    print('TP : ', sumTP/totalImage)
    print('TN : ', sumTN/totalImage)
    print('FN : ', sumFN/totalImage)
    print('FP: ', sumFP/totalImage)



    goodResults = (truePositive + trueNegative) * 100
    badResults = (falseNegative + falsePositive) * 100

    print('Good predictions : ', goodResults)
    print('Bad predictions : ',badResults)


    #Code to save results
    #dictResults[name] = (truePositive * 100, trueNegative *
    #                     100, falseNegative * 100, falsePositive * 100, goodResults, badResults)
    #line = str(name) + ' : ' + str(dictResults[name])
    #with open("C:/Users/15143/OneDrive/Bureau/Images+CSV/Team03/test.txt", "a") as myfile:
    #    myfile.write(line)
    #    myfile.close()
    








