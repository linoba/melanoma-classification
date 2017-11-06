#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:54:47 2017

@author: bombus
"""


import pandas
import numpy as np
import shutil



#this is the path to your data folder
PATH_pictures = "/home/bombus/02456-deep-learning/Project/Data/ISIC_MSK-2_1/"
PATH_benign = "/home/bombus/02456-deep-learning/Project/Data/ISIC_MSK-2_1_sorted/Benign/"
PATH_malignant = "/home/bombus/02456-deep-learning/Project/Data/ISIC_MSK-2_1_sorted/Malignant/"


def filename_label(analysis_filename):
    #inputs: The file name where the analysis data is 
    #ouputs: an np.array with in col 0 the name of the file and in col 1 the label
    data = pandas.read_csv('analysis/'+analysis_filename).values.tolist()
    data_array=np.zeros((len(data)-1,2),dtype='object')
    for i in range(len(data))[1:]:
        data_array[i-1,:]=[data[i][8]+'.jpg',data[i][2]]
    return data_array

data_array=filename_label('ISIC_MSK-2_1.csv')
#data_array=data_array[10:15,:]

def sort_folder(data_array):
    notsorted=[]
    for row in range(data_array.shape[0]):
        data=data_array[row,:]
        if data[1]=='benign':
            shutil.copy2(PATH_pictures+data[0],PATH_benign)
        if data[1]=='malignant':
            shutil.copy2(PATH_pictures+data[0],PATH_malignant)
        else:
            print("different_label",row,data)
            notsorted.append(data)
            
sort_folder(data_array)        