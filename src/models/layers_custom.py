import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.backend import variable


class Conv2DNormalization(Layer):

    def __init__(self, scale, n_channels = 3, **kwargs):
        super(Conv2DNormalization,self).__init__(**kwargs)
        self.scale = scale
        self.n_channels = n_channels
        self.axis = 1

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = variable(init_gamma, name='{}_gamma'.format(self.name))

    def call(self, x, mask=None):
        output = l2_normalize(x, self.axis)
        output = output * self.gamma[tf.newaxis, :, tf.newaxis, tf.newaxis]
        return output







