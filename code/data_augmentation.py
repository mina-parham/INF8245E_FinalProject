import numpy as np
import pandas as pd
import os
import tensorflow as tf

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []    
    for X_img in X_imgs:
        noise = tf.random.normal(shape=tf.shape(X_img), mean=0.0, stddev=0.1, dtype=tf.float32)
        noise_img = X_img + noise
        noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
        gaussian_noise_imgs.append(noise_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs

def rotate(X_imgs, rotation_range=90):
    rotated_imgs = []
    for X_img in X_imgs:
        rotated_img = tf.keras.preprocessing.image.random_rotation(
            X_img, rotation_range, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest',
            cval=0.0, interpolation_order=1)
        rotated_imgs.append(rotated_img)
    return rotated_imgs

def zoom(X_imgs, zoom_range=(0.5,1.3)):
    zoomed_imgs = []
    for X_img in X_imgs:
        zoomed_img = tf.keras.preprocessing.image.random_zoom(
            X_img, zoom_range, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
        zoomed_imgs.append(zoomed_img)
    return zoomed_imgs

def horizontal_flip(X_imgs):
    flipped_imgs = []
    for X_img in X_imgs:
        flipped_img = tf.image.flip_left_right(X_img)
        flipped_imgs.append(X_img)
    return flipped_imgs
  
def vertical_flip(X_imgs):
    flipped_imgs = []
    for X_img in X_imgs:
        flipped_img = tf.image.flip_up_down(X_img)
        flipped_imgs.append(X_img)
    return flipped_imgs


def selective_augment(X_imgs, y_labels, times=2):
    augmented_data_X = np.concatenate((X_imgs, horizontal_flip(X_imgs)))
    augmented_data_Y = np.concatenate((y_labels, y_labels))
    if times==3:
        vertical_flipped_imgs = vertical_flip(X_imgs)
        augmented_data_X = np.concatenate((augmented_data_X, vertical_flipped_imgs))
        augmented_data_Y = np.concatenate((augmented_data_Y, y_labels))
    return (augmented_data_X, augmented_data_Y)

def get_image_data_generator(rotation_range=30, zoom_range=[0.8, 1.2], brightness_range=[0.8,1.0]):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
		rotation_range=rotation_range, zoom_range=zoom_range,
		brightness_range=brightness_range)
    print('Returning image data generator')
    return data_generator
