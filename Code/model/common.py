import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, GlobalAveragePooling2D, Multiply, Reshape

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def channel_attention(input_tensor, num_filters, reduction_ratio=16):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, num_filters))(x)
    x = Conv2D(num_filters // reduction_ratio, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=1, padding='same', activation='sigmoid')(x)
    output = Multiply()([x, input_tensor])
    return output


def residual_channel_attention_block(x, num_filters):
    shortcut = x
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = channel_attention(x, num_filters)
    x = Add()([shortcut, x])
    return x


def residual_group(input_tensor, num_filters, num_block=8, method=None):
    x = input_tensor
    for _ in range(num_block):
        x = residual_channel_attention_block(x, num_filters, method)
    x = Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])
    return x