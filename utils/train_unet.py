import os
import argparse
import logging
import glob

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

import numpy as np
from sklearn.model_selection import train_test_split
from skimage import filters, transform

from nii_preprocess import niftiToFlowArray



def getFileList(image_path, mask_path):
    # Index available Nifti files
    if image_path[-1] != '/':
        image_path = image_path + '/'

    if mask_path[-1] != '/':
        mask_path = mask_path + '/'

    image_list = glob.glob(image_path +'*.nii')
    mask_list = glob.glob(mask_path + '*.nii')

    file_list = list(zip(image_list, mask_list))
    print(f'Found {len(file_list)} Nii files')
    
    return file_list


def resizeDataArrays(image_data_array, mask_data_array, output_height=512, output_width=512):
    # Create temporary arrays
    resized_image_array = np.ndarray(shape=(image_data_array.shape[0], 512, 512, 1), dtype='float32')
    resized_mask_array = np.ndarray(shape=(mask_data_array.shape[0], 512, 512, 1), dtype='float32')
    
    for i in range(image_data_array.shape[0]):
        # Resize to 512 x 512
        resized_image_array[i,:,:,0] = transform.resize(image_data_array[i,:,:,0], (output_height,output_width), anti_aliasing=True)
        resized_mask_array[i,:,:,0] = transform.resize(mask_data_array[i,:,:,0], (output_height,output_width), anti_aliasing=True)

        # Median filter to clean up images
        resized_image_array[i,:,:,0] = filters.median(resized_image_array[i,:,:,0])
        
    return resized_image_array, resized_mask_array


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', help='Path to model', required=True)
    parser.add_argument('-i', '--image_path', help='Path to images', required=True)
    parser.add_argument('-c', '--class_path', help='Path to masks', required=True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=10)
    parser.add_argument('-s', '--epoch_steps', help='Number of steps per epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', help='Training batch size', type=int, default=8)
    #parser.add_argument('-o', '--optimizer', help='Define optimizer (Default: Adam)', default='adam')
    args = parser.parse_args()
    
    model_path = args.model_path
    image_path = args.image_path
    mask_path = args.class_path
    print('Populating file list')
    file_list = getFileList(image_path, mask_path)

    # Convert Nii stacks to rank 4 numpy array (batch size, height, width, channels) to use
    # Keras ImageDataGenerator.flow() method
    print('Converting Nii files to Numpy array')
    image_data_array, mask_data_array = niftiToFlowArray(file_list, image_height=630, image_width=630)
    
    # Resize images to 512 x 512
    print('Resizing images to 512 x 512')
    image_data_array, mask_data_array = resizeDataArrays(image_data_array, mask_data_array, output_height=512, output_width=512)

    # Split into training/testing sets
    print('Splitting image sets into train/test')
    x_train, x_test, y_train, y_test = train_test_split(image_data_array, mask_data_array, test_size=0.33, random_state=42)

    # Instantiate data generators
    print('Creating data generators')
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)

    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    # Provide the same seed and keyword arguments to the fit and flow methods
    # for data and masks so they stay paired
    seed = 1
    batch_size = args.batch_size # This will depend on how much GPU memory, default=8
    image_datagen.fit(x_train, augment=False, seed=seed)
    mask_datagen.fit(y_train, augment=False, seed=seed)

    image_generator = image_datagen.flow(x_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    print('Loading model')
    with strategy.scope():
        # Instantiate model
        model = tf.keras.models.load_model(model_path)

    # Train the model
    print('Training model')
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(train_generator, steps_per_epoch=args.epoch_steps, epochs=args.epochs, callbacks=model_checkpoint) # Defaults: epoch_steps=100, epochs=10
    
    # Save loss history
    loss_history = np.array(history.history['loss'])
    np.savetxt("loss_history.csv", loss_history, delimiter=",")

    # Test the model
    test_datagen = ImageDataGenerator()
    test_datagen.fit(x_test)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=4)

    test_results = model.evaluate(test_generator)

    # Evaluate the model
    # TODO: replace x_test (Covid images) with COPDGene images, move to separate script
    # predict_results = model.predict(x_test, batch_size=10, verbose=1)