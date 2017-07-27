from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Input, Dropout, AlphaDropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten
from keras.layers.merge import Concatenate, Add
from keras import backend as K
import numpy as np



class Block(object):
    """building block base object"""

    # check backend for batch normalization axis.
    channel_axis = -1 if K.image_dim_ordering() is 'tf' else 1

    def __init__(self, activation='relu', batch_norm=True):
        self._activation = activation
        self.batch_norm = batch_norm and (activation is not 'selu')
        self.dropout = AlphaDropout if activation is 'selu' else Dropout
        self.kernal_init = 'lecun_normal' if activation is 'selu' else 'glorot_uniform'

    def activation(self):
        # a hacky check for advanced activations
        if type(self._activation) is not str:
            return self._activation()
        return Activation(self._activation)


class ConvBlock(Block):
    """basic convolutional block, inherits from 
    Block base class
    """

    def __init__(self, activation='relu', batch_norm=True):
        Block.__init__(self, activation, batch_norm)
        self.pool_types = {'max': MaxPooling2D,
                           'strided': self.strided_conv,
                           'depthwise': self.conv_depth_pool}



    def conv2d_unit(self, filters, kernal_size=(3, 3), padding='same', strides=1, skip_activation=False, drop=0.,**kwargs):
        if type(kernal_size) is int:
            kernal_size = (kernal_size, kernal_size)
        def block(inp):
            x = Conv2D(filters, kernal_size, padding=padding,
                       strides=strides, kernel_initializer=self.kernal_init, **kwargs )(inp)

            if skip_activation:
                return x
            if self.batch_norm:
                x = BatchNormalization(scale=False, axis=self.channel_axis)(x)
            x = self.activation()(x)
            if drop:
                x= self.dropout(drop)(x)
            return x
        return block

    def factorized_conv(self, filters, kernel_size=3, padding='same', strides=1, drop=0., **kwargs):
        """a  factorized convolution "layer" equivilant to size"""
        def block(inp):
            x = self.conv2d_unit(filters, (1, kernel_size),
                                 padding=padding, strides=strides, **kwargs)(inp)
            x = self.conv2d_unit(filters, (kernel_size, 1),
                                 padding=padding, strides=strides, drop=drop, **kwargs)(x)
            return x
        return block

    def strided_conv(self, pool_stride=2, kernal_size=3, drop=0., **kwargs):
        def block(inp):
            filters = inp.shape.as_list()[self.channel_axis]
            x = self.conv2d_unit(filters, kernal_size=kernal_size,
                                 padding='same', strides=pool_stride, drop=drop, **kwargs)(inp)
            return x
        return block

    def conv_depth_pool(self, pool_filters=2, drop=0., **kwargs):
        """reduce depth by using a 1x1 convolution, pool_filters must  be
        an integer to be divided by current depth dimension (ie pool_filters= 2 
        on a tensor of shape (None, 64, 64, 128) will result in a tensor of
        shape (None, 64, 64, 64) floor division is used."""
        def block(inp):
            filters = int(inp.shape.as_list()[self.channel_axis] // pool_filters)
            x = self.conv2d_unit(filters, kernal_size=1,
                                 padding='valid', strides=1, drop=drop, **kwargs)(inp)
            return x
        return block



    def conv_block(self, filters, kernel_size=3, n_layers=2, padding='same', strides=1, drop=0., factorized=False, **kwargs):
        """a block of n_layer convolutions followed by a pool layer & dropout (if drop is povided)
        note that if you want if pool is "depthwise" the pool argument should represent ether number of filters or
        """
        if factorized:
            builder = self.factorized_conv
        else:
            builder = self.conv2d_unit

        def block(inp):
            x = inp
            for n in range(n_layers):
                x = builder(filters, kernel_size, padding=padding,
                            strides=strides, drop=drop, **kwargs)(x)
            return x

        return block



    def residual_block(self, filters, kernel_size=3, n_layers=3, padding='same', strides=1, drop=0., factorized=False, **kwargs):

        def block(inp):
            if inp.shape.as_list()[self.channel_axis] == filters:
                skip, x = inp, inp

            else:
                skip, x = self.conv2d_unit(filters, kernel_size, padding=padding,
                            strides=strides, drop=drop, **kwargs)(inp), inp

            x = self.conv_block(filters, kernel_size, n_layers-1, padding=padding, 
                                 strides=strides, drop=drop, factorized=factorized, **kwargs)(x)

            x = self.conv2d_unit(filters, kernel_size, padding=padding,
                            strides=strides, skip_activation=True, drop=drop, **kwargs)(x)

            x = Add()([skip, x])

            if self.batch_norm:
                x = BatchNormalization(axis=self.channel_axis)(x)

            x = self.activation()(x)
            return x

        return block

    # inception blocks ------

    def block_inceptionish_c(self, grow=False, **kwargs):

        def block(inp):
            d = inp.shape.as_list()[self.channel_axis]

            fr = np.array([4, 8, 3.425, 3], 'float32')
            if grow:
                fr /=2

            b1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inp)
            b1 = self.conv_depth_pool(pool_filters=fr[0], **kwargs)(b1)

            b2 = self.conv_depth_pool(pool_filters=fr[0], **kwargs)(inp)

            b3 = self.conv_depth_pool(pool_filters=fr[0], **kwargs)(inp)
            b3_1 = self.conv2d_unit(int(d // fr[1]), (1, 3), **kwargs)(b3)
            b3_2 = self.conv2d_unit(int(d // fr[1]), (3, 1), **kwargs)(b3)
            b3 = Concatenate(axis=self.channel_axis)([b3_1, b3_2])

            b4 = self.conv_depth_pool(pool_filters=fr[0])(inp)
            b4 = self.conv2d_unit(int(d // fr[2]), (1, 3), **kwargs)(b4)
            b4 = self.conv2d_unit(int(d // fr[3]), (3, 1), **kwargs)(b4)
            b4_1 = self.conv2d_unit(int(d // fr[1]), (1, 3), **kwargs)(b4)
            b4_2 = self.conv2d_unit(int(d // fr[1]), (3, 1), **kwargs)(b4)
            b4 = Concatenate(axis=self.channel_axis)([b4_1, b4_2])

            x = Concatenate(axis=self.channel_axis)([b1, b2, b3, b4])
            return x
        return block

    def block_inceptionish_b(self, grow=False, **kwargs):

        def block(inp):
            d = inp.shape.as_list()[self.channel_axis]
            fr = np.array([4, 6, 5.3333, 4.5666], 'float32')
            if grow:
                fr /=2

            b1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inp)
            b1 = self.conv_depth_pool(pool_filters=fr[0], **kwargs)(b1)

            b2 = self.conv_depth_pool(pool_filters=fr[0], **kwargs)(inp)

            b3 = self.conv_depth_pool(pool_filters=fr[2], **kwargs)(inp)
            b3 = self.conv2d_unit(int(d // fr[3]), (1, 7), **kwargs)(b3)
            b3 = self.conv2d_unit(int(d // fr[0]), (7, 1), **kwargs)(b3)

            b4 = self.conv_depth_pool(pool_filters=fr[2], **kwargs)(inp)
            b4 = self.conv2d_unit(int(d // fr[2]), (1, 7), **kwargs)(b4)
            b4 = self.conv2d_unit(int(d // fr[3]), (7, 1), **kwargs)(b4)

            b4 = self.conv2d_unit(int(d // fr[3]), (1, 7), **kwargs)(b4)
            b4 = self.conv2d_unit(int(d // fr[0]), (7, 1), **kwargs)(b4)

            x = Concatenate(axis=self.channel_axis)([b1, b2, b3, b4])
            return x
        return block


    def block_inceptionish_a(self, grow=False, **kwargs):

        def block(inp):
            d = inp.shape.as_list()[self.channel_axis]
            fr = np.array([4, 6], 'float32')
            if grow:
                fr /=2

            b1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inp)
            b1 = self.conv_depth_pool(pool_filters=fr[0], **kwargs)(b1)

            b2 = self.conv_depth_pool(pool_filters=fr[0], **kwargs)(inp)

            b3 = self.conv_depth_pool(pool_filters=fr[1], **kwargs)(inp)
            b3 = self.conv2d_unit(int(d// fr[0]), (3, 3), **kwargs)(b3)

            b4 = self.conv_depth_pool(pool_filters=fr[1])(inp)
            b4 = self.conv2d_unit(int(d // fr[0]), (3, 3), **kwargs)(b4)
            b4 = self.conv2d_unit(int(d // fr[0]), (3, 3), **kwargs)(b4)

            x = Concatenate(axis=self.channel_axis)([b1, b2, b3, b4])

            return x
        return block


    def inceptionish_stem(self, **kwargs):
        """a version of the inception v4 stem, but
        without the valid padding."""
        def block(inp):

            x = self.conv2d_unit(32, 3, strides=2, **kwargs )(inp)
            x = self.conv2d_unit(32, 3, **kwargs)(x)
            x = self.conv2d_unit(64, 3, **kwargs)(x)

            x1 = MaxPooling2D((2, 2), strides=2)(x)
            x2 = self.conv2d_unit(96, 3, strides=2, **kwargs)(x)

            x = Concatenate(axis=self.channel_axis)([x1, x2])

            x1 = self.conv_block(64, 1, **kwargs)(x)
            x1 = self.conv2d_unit( 96, 3, **kwargs)(x1)

            x2 = self.conv2d_unit(64, 1, **kwargs)(x)
            x2 = self.factorized_conv(64, 7, **kwargs)(x2)
            x2 = self.conv2d_unit(96, 3, **kwargs)(x2)

            x = Concatenate(axis=self.channel_axis)([x1, x2])

            x1 = self.conv2d_unit(192, 3, strides=2, **kwargs)(x)
            x2 = MaxPooling2D((2, 2), strides=2)(x)

            x = Concatenate(axis=self.channel_axis)([x1, x2])
            return x
        return block


    def inceptionish_reduction_b(self, **kwargs):
        """version of inception v4 reduction block B"""
        def block(inp):
            x1 = self.conv2d_unit(128, 1, **kwargs)(inp)
            x1 = self.conv2d_unit(128, 3, strides=2, **kwargs)(x1)

            x2 = self.conv2d_unit(256, 1, **kwargs)(inp)
            x2 = self.factorized_conv(320, 7, **kwargs)(x2)
            x2 = self.conv2d_unit(256, 1, strides=2, **kwargs)(x2)

            x3 = MaxPooling2D((2, 2), strides=2, padding='valid')(inp)

            x = Concatenate(axis=self.channel_axis)([x1, x2, x3])
            return x
        return block



class DenseBlock(Block):
    """basic convolutional block, inherits from 
    Block base class
    """
    spatial_transformation = {'flatten': Flatten,
                              'gap': GlobalAveragePooling2D,
                              'gmp': GlobalMaxPooling2D}

    def __init__(self, activation='relu', batch_norm=True, spatial_transformation_layer='flatten'):
        Block.__init__(self, activation, batch_norm)
        self.spatial_transformation_layer = self.spatial_transformation[
            spatial_transformation_layer]

    def dense_unit(self, units, **kwargs):
        """only units need be specified, other other params
        inherited from Block class directly, also takes keyword
        args passed directly to keras Dense.__init__ """
        def block(inp):
            x = Dense(units, kernel_initializer=self.kernal_init,  **kwargs)(inp)
            if self.batch_norm:
                x = BatchNormalization(scale=False, axis=self.channel_axis)(x)
            x = self.activation()(x)
            return x
        return block

    def dense_block(self, units, drop=[.0], n_layers=1, **kwargs):
        """an n_layer dense block w/ each layer followed by dropout (if drop is povided)"""

        try:
            if len(units) < n_layers:
                n_layers = len(units)
        except TypeError:
            units = [units] * n_layers
            pass

        try:
            if len(drop) < n_layers:
                drop = [drop[0]] * n_layers
        except TypeError:
            drop = [drop] * n_layers
            pass

        def block(inp):
            in_shape = inp.shape.as_list()
            x = inp
            if len(in_shape) > 2:
                x = self.spatial_transformation_layer()(x)

            for i in range(n_layers):
                x = self.dense_unit(units[i], **kwargs)(x)
                x = self.dropout(drop[i])(x)

            return x
        return block


