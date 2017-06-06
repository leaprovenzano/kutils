from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Input, Dropout


def dense_bn(units, activation='relu', bn=True, **kwargs):

    def block(inp):
        x = Dense(size)(inp)
        #hacky exclude batchnorm if elu
        if bn & activation is not 'elu':
            x = BatchNormalization(scale=False)(x)
        #hacky check for advanced activation layers like prelu
        if type(activation) is not str:
            x = activation()(x)
        else:
            x = Activation(activation)(x)
        return x
    return block


def dense_block(size, n_layers=1, act='relu', drop=.0, decrease_by=None, bn=True, **kwargs):
    """an n_layer dense block w/ each layer followed by dropout (if drop is povided)"""
    
    def block(inp):
        x = inp
        s=size
        d = drop
        for i in range(n_layers): 
            x = dense_bn(s, activation=act, bn=bn, **kwargs)(x)
            x = Dropout(d)(x)
            if decrease_by:
                s = s // decrease_by
                d = d / decrease_by
        return x
    return block


def conv2d_bn(filters, kernal_size, padding='same', strides= 1, activation='relu', **kwargs):
    def block(inp):
        x = Conv2D(filters, kernal_size, padding=padding, strides= strides, **kwargs)(inp)
        # if activation is elu skip bn
        if activation is not 'elu':
            x = BatchNormalization(scale=False)(x)
        #hacky check for advanced activation layers like prelu
        if type(activation) is not str:
            x = activation()(x)
        else:
            x = Activation(activation)(x)
        return x
    return block


def factorized_conv(filters, kernel_size=3, padding='same', strides= 1, activation='relu', **kwargs):
    """a  factorized convolution "layer" equivilant to size"""
    def block(inp):
        x = conv2d_bn(filters, (1, kernel_size), padding=padding, strides= strides, activation=activation, **kwargs)(inp)
        x = conv2d_bn(filters, (kernel_size, 1), padding=padding, strides= strides, activation=activation, **kwargs)(x)
        return x
    return block



def conv_block(filters, kernel_size=3, n_layers=2, padding='same', strides=1, activation='relu',\
    drop=0., pool=(2, 2), factorized=False, **kwargs):
    """a block of n_layer convolutions followed by a pool layer & dropout (if drop is povided)"""
    if factorized:
        builder = factorized_conv
    else:
        builder = conv2d_bn
    def block(inp):
        x = builder(filters, (kernel_size, kernel_size), padding=padding, strides=strides, activation=activation, **kwargs)(inp)
        for n in range(n_layers - 1):
            x = builder(filters, (kernel_size, kernel_size), padding=padding, strides=strides, activation=activation, **kwargs)(x)
        if pool:
            x = MaxPooling2D(pool)(x)
        if drop:
            x = Dropout(drop)(x)
        return x
    return block


# inception blocks ------

def block_inceptionv4c(d, act='relu', **kwargs):
    def block(inp):
        b1 = conv2d_bn(d, size=(1, 1),  activation=act, **kwargs)(inp)

        b2 = conv2d_bn(round(d * 1.5), size=(1, 1), activation=act, **kwargs)(inp)
        b2_1 = conv2d_bn(d, size=(1, 3), activation=act, **kwargs)(b2)
        b2_2 = conv2d_bn(d, size=(3, 1), activation=act, **kwargs)(b2)
        b2 = Concatenate(axis=-1)([b2_1, b2_2])

        b3 = conv2d_bn(round(d * 1.5), size=(1, 1), activation=act, **kwargs)(inp)
        b3 = conv2d_bn(round(d * 1.75), size=(3, 1), activation=act, **kwargs)(b3)
        b3 = conv2d_bn(d*2, size=(1, 3), activation=act, **kwargs)(b3)
        b3_1 = conv2d_bn(d, size=(1, 3), activation=act, **kwargs)(b3)
        b3_2 = conv2d_bn(d, size=(3, 1), activation=act, **kwargs)(b3)
        b3 = Concatenate( axis=-1)([b3_1, b3_2])

        b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inp)
        b4 = conv2d_bn(d, size=(1, 1), activation=act, **kwargs)(b4)

        x = Concatenate( axis=-1)([b1, b2, b3, b4])
        return x
    return block


def block_inceptionv4a(d, act='relu', **kwargs):
    
    def block(inp):
        a1 = conv2d_bn(round(d * 1.5), size=(1, 1), activation=act, **kwargs)(inp)

        a2 = conv2d_bn(d, size=(1, 1), activation=act, **kwargs)(inp)
        a2 = conv2d_bn(round(d * 1.5), size=(3, 3), activation=act, **kwargs)(a2)

        a3 = conv2d_bn(d, size=(1, 1), activation=act, **kwargs)(inp)
        a3 = conv2d_bn(round(d * 1.5), size=(3, 3), activation=act, **kwargs)(a3)
        a3 = conv2d_bn(round(d * 1.5), size=(3, 3), activation=act, **kwargs)(a3)

        a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(inp)
        a4 = conv2d_bn(round(d * 1.5), size=(1, 1), activation=act, **kwargs)(a4)

        x = Concatenate( axis=-1)([a1, a2, a3, a4])
        return x
    return block

