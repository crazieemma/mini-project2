#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 12:14:05 2018

@author: mac
"""

import os  
import tensorflow as tf  
from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt
import math
 
train_dir = '/home/ece-student/Desktop/image/input_data/'

truck = []
label_truck = []
bus = []
label_bus = []
bicycle = []
label_bicycle = []

def get_files(file_dir,ratio):
    for file in os.listdir(file_dir+'/bus'):
            bus.append(file_dir +'/bus'+'/'+file)
            label_bus.append(0)
    for file in os.listdir(file_dir+'/truck'):
            truck.append(file_dir +'/truck'+'/'+ file) 
            label_truck.append(1)
    for file in os.listdir(file_dir+'/bicycle'):
            bicycle.append(file_dir +'/bicycle'+'/'+file)
            label_bicycle.append(2)
    print("there are %d truck\nthere are %d bus\nthere are %d bicycle\n"%(len(truck),len(bus),len(bicycle)),end ="")       
    image_list = np.hstack((bus,truck,bicycle))
    label_list = np.hstack((label_bus, label_truck, label_bicycle))
    
    #shuffle the list
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    #take out the list from the shuffled tempimg和lab）
    #image_list = list(temp[:, 0])
    #label_list = list(temp[:, 1])
    #label_list = [int(i) for i in label_list]
    #return image_list, label_list

    #transfer all the img and lab to list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
 
    #divide the list to two parts,one part for train,the other for test.ratio is the percentage for the test
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))   #the num of test sample
    n_train = n_sample - n_val   #the num of train sample
 
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
 
    return tra_images, tra_labels, val_images, val_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
   
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0]) #read img from a queue
    
    image = tf.image.decode_jpeg(image_contents, channels=3)
   
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 32, 
                                                capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch 



