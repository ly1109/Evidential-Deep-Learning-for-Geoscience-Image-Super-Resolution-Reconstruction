from tensorflow.keras.layers import Input, Lambda, SpatialDropout2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

from common import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def build_model(input_shape, scale, num_filters, num_group, num_block, method):
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(inputs)
    x_1 = x

    for _ in range(num_group):
        x = residual_group(x, num_filters, num_block, method)
        if method == 'dropout':
            x = SpatialDropout2D(0.1)(x)

    x = Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x_1, x])

    if method == 'evidential':
        mu = Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
        mu = Lambda(subpixel_conv2d(scale))(mu)
        mu = Conv2D(input_shape[-1], kernel_size=3, strides=1, padding='same')(mu)

        y = UpSampling2D(size=(scale, scale))(x)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        y = Conv2D(input_shape[-1] * 3, kernel_size=3, strides=1, padding='same', activation=tf.nn.softplus)(y)
        v, alpha, beta = tf.split(y, 3, axis=-1)
        outputs = Concatenate()([mu, v, alpha + 1.0, beta])

    elif method == 'dropout':
        mu = Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
        mu = Lambda(subpixel_conv2d(scale))(mu)
        mu = Conv2D(input_shape[-1], kernel_size=3, strides=1, padding='same')(mu)

        y = UpSampling2D(size=(scale, scale))(x)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        y = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(y)
        logvar = Conv2D(input_shape[-1], kernel_size=3, strides=1, padding='same')(y)
        outputs = Concatenate()([mu, logvar])

    elif method is None:
        x = Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(subpixel_conv2d(scale))(x)
        outputs = Conv2D(input_shape[-1], kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=outputs, name='rcan')