# LeNet Keras implementation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *

import argparse

def lenet(
    metrics,
    input_height=512, 
    input_width=512, 
    input_channels=1,  
    num_classes=2, 
    dropout=None, 
    activation='relu', 
    weights_path=None,
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.10, momentum=0.99)
):
    # Initialize model
    model = Sequential()

    input_shape = (input_height, input_width, input_channels)

    # Model layers
    model.add(Conv2D(16, 5, padding='same', input_shape=input_shape, activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(32, 5, padding='same', input_shape=input_shape, activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    # If netwrok was pretrained
    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--input_height', help='Input height', type=int, default=512)
    parser.add_argument('-W', '--input_width', help='Input shape', type=int, default=512)
    parser.add_argument('-C', '--input_channels', help='Input channels', type=int, default=1)
    parser.add_argument('-o', '--output_path', help='Output path', default='lenet.h5')
    parser.add_argument('-A', '--activation', help='Activation', default='relu')
    parser.add_argument('-w', '--weights_path', help='Load weights from file', default=None)
    parser.add_argument('-c', '--num_classes', help='Number of classes', type=int, default=2)
    parser.add_argument('-l', '--loss', help='Loss function', default='categorical_crossentropy')
    parser.add_argument('-d', '--dropout', help='Dropout rate', type=float, default=0.5)
    # TODO:
    # parser.add_argument('-O', '--optimizer', help='Optimizer')
    # parser.add_argument('-M', '--metrics', help='Metrics')
    args = parser.parse_args()

    input_height = args.input_height
    input_width = args.input_width
    input_channels = args.input_channels
    num_classes = args.num_classes
    dropout = args.dropout
    activation = args.activation
    weights_path = args.weights_path
    loss = args.loss
    output_path = args.output_path
    optimizer = SGD(lr=0.01, momentum=0.90)

    print(f'Args: {vars(args)}')

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of GPUS: {strategy.num_replicas_in_sync}')

    with strategy.scope():
        metrics = ['acc']
        model = lenet(metrics, input_height, input_width, input_channels, num_classes, dropout, activation, weights_path, loss, optimizer)

    model.save(output_path)