import pandas
import numpy as np
import shutil
import os

#this is the path to your data folder
PATH_pictures = "/home/bombus/02456-deep-learning/Project/Data/ISIC_MSK-2_1/"
PATH_benign = "/home/bombus/Project/Project/melanoma-classification/Data/ISIC_MSK-2_1_sorted/train/Benign/"
PATH_malignant = "/home/bombus/Project/Project/melanoma-classification/Data/ISIC_MSK-2_1_sorted/train/Malignant/"
PATH_test = "/home/bombus/Project/Project/melanoma-classification/Data/ISIC_MSK-2_1_sorted/test/"
PATH_validation = "/home/bombus/Project/Project/melanoma-classification/Data/ISIC_MSK-2_1_sorted/validation/"
path_train= "/home/bombus/Project/Project/melanoma-classification/Data/ISIC_MSK-2_1_sorted/train/"

filesmalignant= os.listdir(PATH_malignant)
filesbenign = os.listdir(PATH_benign)

def split_data(path_test,path_validation):
    # approx 0.6 to training, 0.2 to validation, 0.2 to test
    filesmalignant= os.listdir(PATH_malignant)
    filesbenign = os.listdir(PATH_benign)
    for file in filesmalignant[:70]:
        shutil.move(path_train+'Malignant/'+file, PATH_validation+'Malignant')
    for file in filesmalignant[70:70+70]:
        shutil.move(path_train+'Malignant/'+file, PATH_test+'Malignant') 
    for file in filesbenign[:233]:
        shutil.move(path_train+'Benign/'+file, PATH_validation+'Benign')
    for file in filesbenign[233:233+234]:
        shutil.move(path_train+'Benign/'+file, PATH_test+'Benign')  

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
            
#sort_folder(data_array)        