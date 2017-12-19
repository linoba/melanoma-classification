# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:23:33 2017

@author: Georg
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img,array_to_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

from keras import backend as K
from keras.utils import plot_model
K.set_image_dim_ordering('th')
import os
import matplotlib.pyplot as plt
os.chdir("C:\\Users\\Georg\\Desktop")

from IPython.display import Image, display, clear_output
import datetime

import time
import cv2

img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'train'
validation_data_dir = 'validation'
test_dir = 'test'
nb_train_samples = 800
nb_validation_samples = 370
epochs = 20
batch_size = 10



def TakePictureAndSave():
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    time.sleep(0.3)  # If you don't wait, the image will be dark
    return_value, image = camera.read()
    cv2.imwrite("test.png", image)
    del(camera)  # so that others can use the camera as soon as possible

def predict_image_class(file):
    model = applications.VGG16(include_top=False, weights='imagenet')
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    model = Sequential()
    model.add(Flatten(input_shape=array.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights(top_model_weights_path)
    class_predicted = model.predict_classes(array)
    if class_predicted==1:
        print("malignant")
    else:
        print("benign")


TakePictureAndSave()
directory = 'opencv.png'

predict_image_class(directory)

