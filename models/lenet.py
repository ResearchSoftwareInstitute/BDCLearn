# LeNet Keras implementation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras import backend as K 

class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):
        # Initialize model
        model = Sequential()

        # Check if "channels first" is selected
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
        else:
            inputShape = (imgRows, imgCols, numChannels)

        # Model layers
        model.add(Conv2D(20, 5, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides(2,2)))

        model.add(Conv2D(50, 5, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides(2,2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        # If netwrok was pretrained
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

