
import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np

from lib.Layer import Layer
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length
from lib.init_tensor import init_filters

def quantize_weights(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (7. - (-8.))
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, -8, 7)
  return w, scale

class Convolution(Layer):

    def __init__(self, input_shape, filter_sizes, init, strides=[1,1,1,1], padding='SAME', bias=0., use_bias=True, name=None, load=None, train=True):
        self.input_shape = input_shape
        self.filter_sizes = filter_sizes
        self.batch_size, self.h, self.w, self.fin = self.input_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        self.init = init
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        self.padding = padding
        self.use_bias = use_bias
        assert(use_bias == True)
        self.name = name
        self.train_flag = train
        
        filters = init_filters(size=self.filter_sizes, init=self.init)
        bias = np.ones(shape=self.fout) * bias

        self.filters = tf.Variable(filters, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        # return [(self.name, self.filters), (self.name + "_bias", self.bias)]
        return [(self.name, quantize_weights(self.filters)), (self.name + "_bias", quantize_weights(self.bias))]

    def output_shape(self):
        oh = conv_output_length(self.h, self.fh, self.padding.lower(), self.sh)
        ow = conv_output_length(self.w, self.fw, self.padding.lower(), self.sw)
        od = self.fout
        return [oh, oh, od]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size

    ###################################################################

    def forward(self, X):
        qw, sw = quantize_weights(self.filters)
        qb, sb = quantize_weights(self.bias) 
        Z = tf.nn.conv2d(X, (qw * sw), self.strides, self.padding) + (qb * sb)
        return Z, (Z,)
        
    def forward1(self, X):
        qw, sw = quantize_weights(self.filters)
        qb, sb = quantize_weights(self.bias) 
        Z = tf.nn.conv2d(X, qw, self.strides, self.padding) + qb
        return Z, (tf.reduce_max(Z),)
        
    def forward2(self, X):
        return self.forward1(X)

    ###################################################################
    
    def bp(self, AI, AO, DO, cache):    
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        # DI = tf.nn.conv2d_transpose(value=DO, filter=self.filters, output_shape=self.input_shape, strides=self.strides, padding=self.padding)
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        if self.use_bias:
            return DI, [(DF, self.filters), (DB, self.bias)]
        else:
            return DI, [(DF, self.filters)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ################################################################### 
        
        
