import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np


# For more information about directory tree setup, visit:
# https://keras.io/api/preprocessing/image/#flowfromdirectory-method
def generateDataFromDirectory(data_generator, path, batch_size, seed=None):
    generator = data_generator.flow_from_directory(
	    directory=path,
	    target_size=(512, 512),
	    color_mode="greyscale",
	    classes=None,
	    class_mode="categorical",
	    batch_size=batch_size,
	    shuffle=True,
	    seed=seed,
	    save_to_dir=None,
	    save_prefix="",
	    save_format="png",
	    follow_links=False,
	    subset=None,
	    interpolation="nearest",
	)

    return generator


# To maintain image/mask pair, provide same seed and keyword args to generator for each
def generateDataFromArray(data_generator, dataset, batch_size, seed=None, augment=False)
    datagen = data_generator.fit(dataset, augment=augment, seed=seed)
    generator = datagen.flow(dataset, batch_size=batch_size, seed=seed)

    return generator


def trainModel(model_path, train_generator, epochs, epoch_steps, output_path=None, strategy=None, loss_history_path='loss_history.csv')
    # If no strategy is specified, create a mirrored strategy by default
    if not strategy:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # If no output path is given, overwrite the input model file with trained model
    if not output_path:
        output_path = model_path

    # Open a strategy scope and load the model
    with strategy.scope():
        model = tf.keras.models.load_model(model_path, compile=True)

    # Train the model
    model_checkpoint = ModelCheckpoint(output_path, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(train_generator, steps_per_epoch=epoch_steps, epochs=epochs, callbacks=model_checkpoint) # Defaults: epoch_steps=100, epochs=10
    
    # Save loss history
    loss_history = np.array(history.history['loss'])
    np.savetxt(loss_history_path, loss_history, delimiter=",")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', help='Path to model file', required=True)
    parser.add_argument('-i', '--image_path', help='Path to images', required=True)
    parser.add_argument('-c', '--class_path', help='Path to masks', required=True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=10)
    parser.add_argument('-s', '--epoch_steps', help='Number of steps per epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', help='Training batch size', type=int, default=8)
    parser.add_argument('-o', '--output_path', help='Output path', default=None)
    parser.add_argument('-l', '--loss_history_path', help='Path to save loss history', default='loss_history.csv')
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path
    class_path = args.class_path
    epochs = args.epochs
    epoch_steps = args.epoch_steps
    batch_size = args.batch_size
    output_path = args.output_path

    # Set a seed number to keep images and masks paired
    seed = 1 

    image_datagen = ImageDataGenerator()
    image_generator = generateDataFromDirectory(data_generator=image_datagen, path=image_path, batch_size=batch_size, seed=seed)

    mask_datagen = ImageDataGenerator()
    mask_generator = generateDataFromDirectory(data_generator=mask_datagen, path=mask_path, batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)

    trainModel(model_path=model_path, train_generator=train_generator, epochs=epochs, epoch_steps=epoch_steps, output_path=output_path, loss_history_path=loss_histroy_path)
