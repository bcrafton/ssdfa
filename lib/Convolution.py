
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Linear

from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

from lib.init_tensor import init_filters

class Convolution(Layer):

    def __init__(self, input_shape, filter_sizes, init, strides=[1,1,1,1], padding='SAME', activation=None, bias=0., use_bias=False, name=None, load=None, train=True):
        self.input_shape = input_shape
        self.filter_sizes = filter_sizes
        self.batch_size, self.h, self.w, self.fin = self.input_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        self.init = init
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        self.padding = padding
        self.activation = Linear() if activation == None else activation
        self.use_bias = use_bias
        self.name = name
        self.train_flag = train
        
        filters = init_filters(size=self.filter_sizes, init=self.init)
        bias = np.ones(shape=self.fout) * bias

        self.filters = tf.Variable(filters, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def output_shape(self):
        oh = conv_output_length(self.h, self.fh, self.padding.lower(), self.sh)
        ow = conv_output_length(self.w, self.fw, self.padding.lower(), self.sw)
        od = self.fout
        return [oh, oh, od]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size

    def forward(self, X):
        Z = tf.nn.conv2d(X, self.filters, self.strides, self.padding)
        if self.use_bias:
            Z = Z + tf.reshape(self.bias, (1, 1, 1, self.fout))

        A = self.activation.forward(Z)
        return {'aout':A, 'cache':{}}

    ###################################################################
    
    def backward(self, AI, AO, DO, cache):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache):    
        if not self.train_flag:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO, cache):
        return self.backward(AI, AO, DO, cache)
        
    def dfa_gv(self, AI, AO, E, DO, cache):
        return self.gv(AI, AO, DO, cache)
        
    ###################################################################    
        
    def lel_backward(self, AI, AO, DO, Y, cache):
        return self.backward(AI, AO, DO, cache)

    def lel_gv(self, AI, AO, DO, Y, cache):
        return self.gv(AI, AO, DO, cache)

    ################################################################### 
        
        
