
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
        
        filters = np.absolute(init_filters(size=self.filter_sizes, init=self.init))
        sign = np.array([1.] * (self.fout // 2) + [-1.] * (self.fout // 2))

        self.filters = tf.Variable(filters, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        self.sign = tf.constant(sign, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters)]

    def output_shape(self):
        oh = conv_output_length(self.h, self.fh, self.padding.lower(), self.sh)
        ow = conv_output_length(self.w, self.fw, self.padding.lower(), self.sw)
        od = self.fout
        return [oh, oh, od]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        return filter_weights_size 

    def forward(self, X):
        Z = tf.nn.conv2d(X, self.filters, self.strides, self.padding)
        A = self.activation.forward(Z) * self.sign
        return {'aout':A, 'cache':{}}

    ###################################################################
    
    def bp(self, AI, AO, DO, cache):
        DO = DO * self.activation.gradient(tf.abs(AO)) * self.sign
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        
        return {'dout':DI, 'cache':{}}, [(DF, self.filters)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ################################################################### 
        
        
