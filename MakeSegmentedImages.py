import numpy as np
import matplotlib.pyplot as plt
import os





def seg_by_fixed_thresh(saliencyMap, threshold):
    map = np.zeros((224,224))
    for i in range(len(saliencyMap)):
        for j in range(len(saliencyMap[i])):
            if(saliencyMap[i][j] > threshold):
                map[i][j] = 1 
    
    return map

def seg_all_saliency_maps(allSM,threshold):
    allSegmented = []
    for sm in allSM:
        map = seg_by_fixed_thresh(sm, threshold)
        allSegmented.append(map)
    
    return allSegmented

def seg_all_saliency_maps_Achanta(allSM):
    allSegmented = []
    for sm in allSM:
        thre = np.mean(sm)
        map = seg_by_fixed_thresh(sm,2*thre)
        allSegmented.append(map)
        
    return allSegmented
    

def find_intersection_over_union(pred, groundT):
    FP = 0
    FN = 0
    TP = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if (pred[i][j] == 1 and groundT[i][j]== 1):
                TP = TP+1
            elif(pred[i][j] == 1 and groundT[i][j]== 0):
                FP = FP+1
            elif(pred[i][j] == 0 and groundT[i][j] == 1):
                FN = FN+1
    
    IOU =TP / (TP+FP+FN)
    
    return IOU

def find_number_of_hits(pred,groundT):
    TP = 0
    TOT = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if (pred[i][j] == 1 and groundT[i][j]== 1):
                TP = TP+1
            if(groundT[i][j] == 1):
                TOT = TOT+1
    return TP/TOT

                    