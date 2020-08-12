import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Cropping2D
from tensorflow.keras.backend import int_shape
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

import argparse
import os
import logging

# UNet Keras implementation
def unet(
    input_height,
    input_width,
    input_channels,
    metrics,
    num_classes,
    dropout, 
    filters,
    output_activation, # 'sigmoid' or 'softmax'
    weights_path,
    loss,
    optimizer,
    num_layers,
): 
    # Build U-Net model
    input_shape = (input_height, input_width, input_channels)
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters)
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2) (x)
        filters = filters*2 # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters)

    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer 
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)       
        ch, cw = get_crop_shape(int_shape(conv), int_shape(x))
        conv = Cropping2D(cropping=(ch, cw))(conv)
        x = Concatenate()([x, conv])
        x = conv2d_block(inputs=x, filters=filters)
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)    
    model = Model(inputs=[inputs], outputs=[outputs])

    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def conv2d_block(
    inputs,
    use_batch_norm=False,
    dropout=0.0,
    filters=64,
    kernel_size=(3,3),
    activation='relu',
    kernel_initializer='he_normal',
    padding='same'
):
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm
    )(inputs)

    if use_batch_norm:
        c = BatchNormalization()(c)

    if dropout > 0.0:
        c = Dropout(dropout)(c)

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm
    )(c)

    if use_batch_norm:
        c = BatchNormalization()(c)

    return c


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--input_height', help='Input height', type=int, default=512)
    parser.add_argument('-W', '--input_width', help='Input shape', type=int, default=512)
    parser.add_argument('-C', '--input_channels', help='Input channels', type=int, default=1)
    parser.add_argument('-o', '--output_path', help='Output path', default='unet.hdf5')
    parser.add_argument('-F', '--filters', help='Number of filters', type=int, default=64)
    parser.add_argument('-A', '--activation', help='Output activation', default='sigmoid')
    parser.add_argument('-w', '--weights_path', help='Load weights from file', default=None)
    parser.add_argument('-c', '--num_classes', help='Number of classes', type=int, default=1)
    parser.add_argument('-l', '--loss', help='Loss function', default='binary_crossentropy')
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
    filters = args.filters
    output_activation = args.activation
    weights_path = args.weights_path
    loss = args.loss
    output_path = args.output_path
    optimizer = SGD(lr=0.01, momentum=0.99)
    num_layers = 4

    logger.debug(f'Args: {args}')
    print(f'Args: {vars(args)}')

    strategy = tf.distribute.MirroredStrategy()
    logger.debug(f'Number of GPUS: {strategy.num_replicas_in_sync}')

    with strategy.scope():
        metrics = [MeanIoU(num_classes=args.num_classes+1)]
        model = unet(input_height, input_width, input_channels, metrics, num_classes, dropout, filters, output_activation, weights_path, loss, optimizer, num_layers)

    model.save(output_path)