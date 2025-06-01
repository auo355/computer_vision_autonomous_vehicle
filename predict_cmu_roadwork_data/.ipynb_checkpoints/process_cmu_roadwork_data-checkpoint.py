#!/usr/bin/env python
# coding: utf-8

import 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow.keras as keras
from tensorflow.keras.utils import load_img, img_to_array, array_to_img, to_categorical
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pathlib
from argparse import ArgumentParser
from PIL import Image


def modify_address(x, val_or_train):
        x1 = x.split("sem_seg_labels/gtFine/"+val_or_train+"/")
        x2 = x1[1].split("_Ids")
        y = x1[0] + 'images/' + x2[0] + '.jpg'
        return y



def shuffle_directories(train_input_dir, filt_type, seed, shuffle, chunkstart, chunkend, val_or_train, is_it_label_address):
    train_input_img_paths = [os.path.join(train_input_dir, fname) 
         for fname in os.listdir(train_input_dir) 
             if fname.endswith(filt_type)]

    if (is_it_label_address==False):
        train_input_img_paths = [modify_address(x, val_or_train) for x in train_input_img_paths]
    else:
        pass
        
    if shuffle == True:
        random.Random(seed).shuffle(train_input_img_paths)
    else:
        pass
    return train_input_img_paths[chunkstart : chunkend]

    
def convert_color_to_mask_class (x, img_breadth, img_length, label_list ): 
    x = tf.convert_to_tensor(x)
    y = tf.Variable(np.zeros((img_breadth, img_length, 1)))
    for label in label_list:
        y[:,:,0].assign(tf.where(  ((x[:,:,0]==label[0])   & (x[:,:,1]==label[1])   & (x[:,:,2]==label[2]))  , 1 , y[:,:,0]))
    return y.numpy()

def path_to_target (path, img_breadth, img_length, label):
    mask = img_to_array(load_img(path, target_size=(img_breadth, img_length)  ))
    mask = convert_color_to_mask_class (mask, img_breadth, img_length, label)
    return mask

def get_numpy_array_of_all_targets_in_folder(train_target_paths, num_images, img_breadth, img_length, label):
    train_targets_updated = np.zeros( (num_images, img_breadth, img_length, 1), dtype='int')
    for i in range(num_images):
        train_targets_updated[i] = path_to_target(train_target_paths[i], img_breadth, img_length, label)
    return train_targets_updated


def get_number_of_distinct_items(img):
    img_adjust = img.reshape((np.size(img,0)*np.size(img,1),1))
    img_pandas = pd.DataFrame(img_adjust, columns =['class'])
    return img_pandas['class'].value_counts()


def path_to_raw_data(path, img_breadth, img_length ):
    return img_to_array(load_img(path, target_size=(img_breadth, img_length) ))


def get_numpy_array_of_all_images_in_folder(train_input_img_paths, num_images, img_breadth, img_length):
    train_input_imgs = np.zeros( (num_images, img_breadth, img_length, 3), dtype='float32')
    for i in range(num_images):
        train_input_imgs[i] = path_to_raw_data(train_input_img_paths[i],img_breadth, img_length)
    return train_input_imgs


def is_subarray_present(array_2d, array_1d):
    answer = tf.reduce_any(tf.reduce_all(tf.equal(array_2d, array_1d), axis=1))
    return answer.numpy()


def get_name_of_image_with_label(label, input_img_paths, img_breadth, img_length):
    label = tf.convert_to_tensor(label)
    address = []
    count = 0
    
    for x in input_img_paths:
        image_x = img_to_array(load_img(x, target_size=(img_breadth, img_length) ))
        img_x_adjust = image_x.reshape((np.size(image_x,0)*np.size(image_x,1),3))

        if (count >= 20):
            break
        else:
            pass

        img_x_adjust = tf.convert_to_tensor(img_x_adjust)
        

        label_present = is_subarray_present(img_x_adjust, label)
        if (label_present == True):
            address.append(x)
            count = count + 1
        else:
            pass

    # get address of target with highest label/object frequency
    array_of_labels = get_numpy_array_of_all_targets_in_folder(address, len(address), img_breadth, img_length, label)
    array_of_frequency = np.zeros((len(address)))
    for i in range(len(address)):
        array_of_frequency[i] = get_number_of_distinct_items(array_of_labels[i])[1]
    maximum_index = np.argmax(array_of_frequency)
               
    return address[maximum_index]




def get_plots_for_a_class(label, address_val_target, img_breadth, img_length, val_or_train):
    target_address = get_name_of_image_with_label(label, address_val_target, img_breadth, img_length)
    image_address = modify_address(target_address, val_or_train)
    
    image = img_to_array(load_img(image_address, target_size=(img_breadth, img_length) ))
    plt.imshow(array_to_img(image))
    plt.show()
    
    target = img_to_array(load_img(target_address, target_size=(img_breadth, img_length) ))
    
    modified_target = convert_color_to_mask_class (target, img_breadth, img_length, label )
    plt.imshow(array_to_img(modified_target))
    plt.show()
    

def get_batch_of_image_and_label(input_path, target_path, input_filter_type, target_filter_type, img_length, img_breadth, chunkstart, chunkend,  label_list, val_or_train, seed=50, shuffle=False):
    train_input_img_paths = shuffle_directories(input_path, input_filter_type, seed, shuffle, chunkstart, chunkend, val_or_train, is_it_label_address=False)
    train_target_paths = shuffle_directories(target_path, target_filter_type, seed, shuffle, chunkstart, chunkend, val_or_train, is_it_label_address=True  )
    img_length = img_length
    img_breadth = img_breadth
    num_images = len(train_target_paths)
    train_targets_updated = get_numpy_array_of_all_targets_in_folder(train_target_paths, num_images, img_breadth, img_length,  label_list)
    train_input_imgs = get_numpy_array_of_all_images_in_folder(train_input_img_paths, num_images, img_breadth, img_length)
    return [train_input_imgs, train_targets_updated]
    
def 

