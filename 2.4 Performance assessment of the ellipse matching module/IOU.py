"""This code is used to evaluate the metric 'Intersection over Union' (IOU) between real bounding boxes and 
   predicted bounding boxes. The mean, median, standard deviation and histogram of the IOUs are displayed 
   at the end of the execution"""

import numpy as np
from matplotlib import pyplot as plt


## From https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((float(boxA[2]) - float(boxA[0])) * (float(boxA[3]) - float(boxA[1])))
    boxBArea = abs((float(boxB[2]) - float(boxB[0])) * (float(boxB[3]) - float(boxB[1])))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# Path to annotations (bounding boxes of the dataset) + path to output from YOLO
path_annot = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Projet2_rendu\\Annotations.csv' 
path_test = 'C:\\Users\\Julien\\Documents\\School\\Tuyaux 2019-2020\\Q1\\CV\\Projet2_rendu\\Detection_Results_Test.csv' 

sub_annot =[]
sub_test = []

with open(path_annot,'r') as file:
    lines_annot = file.readlines()

with open(path_test,'r') as file:
    lines_test = file.readlines()
    
## List of images in each subset    
for i in range(len(lines_annot)):
    sub_annot.append(lines_annot[i].split(',')[0])
for j in range(len(lines_test)): 
    sub_test.append(lines_test[j].split(',')[0])

## When matching between the 2 list, evaluate IOU. If several possibility, keep the best
list_iou = np.zeros(len(sub_test))
for index, value in enumerate(sub_test) :
    match = [index2 for index2,value2 in enumerate(sub_annot) if value2 == value]
    if len(match) > 0:
        for i in match:
            sub_test2 = lines_test[index].split(',')
            sub_annot2 = lines_annot[i].split(',')
            boxA = [sub_test2[2], sub_test2[3], sub_test2[4], sub_test2[5]]
            boxB = [sub_annot2[1], sub_annot2[2], sub_annot2[3], sub_annot2[4]]
            list_iou[index] = max(list_iou[index], bb_intersection_over_union(boxA, boxB))

print("Mean of IOU : ", np.mean(list_iou))
print("Standard deviation of IOU : ", np.std(list_iou))
print("Median of IOU : ", np.median(list_iou))

plt.hist(list_iou)
plt.title("IOU distribution")
plt.ylabel("Number of images")
plt.xlabel("IOU")
plt.show()