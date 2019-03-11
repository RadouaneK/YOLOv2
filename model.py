from keras.layers import Input, Add, Activation, BatchNormalization, Reshape, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Layer
from keras.layers.merge import concatenate
from keras import backend as K

def DarkNet_deep(input_shape, trainable=True):

    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    # Layer 1
    x = Conv_layer(input_shape, 32, (3, 3), strides=(1, 1), num='1', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv_layer(x, 64, (3, 3), strides=(1, 1), num='2', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv_layer(x, 128, (3, 3), strides=(1, 1), num='3', trainable=trainable)
    x = Conv_layer(x, 64, (1, 1), strides=(1, 1), num='4', trainable=trainable)
    x = Conv_layer(x, 128, (3, 3), strides=(1, 1), num='5', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv_layer(x, 256, (3, 3), strides=(1, 1), num='6', trainable=trainable)
    x = Conv_layer(x, 128, (1, 1), strides=(1, 1), num='7', trainable=trainable)
    x = Conv_layer(x, 256, (3, 3), strides=(1, 1), num='8', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv_layer(x, 512, (3, 3), strides=(1, 1), num='9', trainable=trainable)
    x = Conv_layer(x, 256, (1, 1), strides=(1, 1), num='10', trainable=trainable)
    x = Conv_layer(x, 512, (3, 3), strides=(1, 1), num='11', trainable=trainable)
    x = Conv_layer(x, 256, (1, 1), strides=(1, 1), num='12', trainable=trainable)
    x = Conv_layer(x, 512, (3, 3), strides=(1, 1), num='13', trainable=trainable)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='14', trainable=trainable)
    x = Conv_layer(x, 512, (1, 1), strides=(1, 1), num='15', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='16', trainable=trainable)
    x = Conv_layer(x, 512, (1, 1), strides=(1, 1), num='17', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='18', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='19', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='20', trainable=trainable)

    # Layer 21
    skip_connection = Conv_layer(skip_connection, 64, (1, 1), strides=(1, 1), num='21', trainable=trainable)
    skip_connection = Lambda(space_to_depth_x2, name='space_to_depth')(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='22', trainable=trainable)

    return x
