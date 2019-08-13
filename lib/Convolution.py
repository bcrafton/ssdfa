
import tensorflow as tf
import numpy as np

from lib.Layer import Layer
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length
from lib.init_tensor import init_filters

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
        self.name = name
        self.load = load
        self.train_flag = train
        
        if self.load:
            print ('Loading: %s' % (self.name))
            weights = np.load(self.load, allow_pickle=True).item()
            filters = weights[self.name]
            if self.use_bias:
                bias = weights[self.name + '_bias']
        else:
            filters = init_filters(size=self.filter_sizes, init=self.init)
            if self.use_bias:
                bias = np.ones(shape=self.fout) * bias

        self.filters = tf.Variable(filters, dtype=tf.float32)
        if self.use_bias:
            self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        if self.use_bias:
            return [(self.name, self.filters), (self.name + "_bias", self.bias)]
        else:
            return [(self.name, self.filters)]

    def output_shape(self):
        oh = conv_output_length(self.h, self.fh, self.padding.lower(), self.sh)
        ow = conv_output_length(self.w, self.fw, self.padding.lower(), self.sw)
        od = self.fout
        return [oh, oh, od]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        if self.use_bias:
            return filter_weights_size + bias_weights_size
        else:
            return filter_weights_size

    def forward(self, X):
        Z = tf.nn.conv2d(X, self.filters, self.strides, self.padding)
        if self.use_bias:
            Z = Z + self.bias
        return {'aout':Z, 'cache':{}}

    ###################################################################
    
    def bp(self, AI, AO, DO, cache):    
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        if self.use_bias:
            return {'dout':DI, 'cache':{}}, [(DF, self.filters), (DB, self.bias)]
        else:
            return {'dout':DI, 'cache':{}}, [(DF, self.filters)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ################################################################### 
        
        
