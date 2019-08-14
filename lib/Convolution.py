
import tensorflow as tf
import numpy as np

from lib.Layer import Layer
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length
from lib.init_tensor import init_filters

class Convolution(Layer):

    def __init__(self, input_shape, filter_sizes, init, strides=[1,1,1,1], padding='SAME', bias=0., use_bias=False, name=None, load=None, train=True):
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
        self.train_flag = train
        
        filters = np.absolute(init_filters(size=self.filter_sizes, init=self.init))
        ss = np.ones_like(filters) * np.average(filters)
        bias = np.ones(shape=self.fout) * bias

        self.filters = tf.Variable(filters, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        self.ss = tf.constant(ss, dtype=tf.float32)
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
        return filter_weights_size + bias_weights_size

    def forward(self, X):
        Z = tf.nn.conv2d(X, self.filters, self.strides, self.padding)
        if self.use_bias:
            Z = Z + self.bias
        return {'aout':Z, 'cache':{}}

    ###################################################################
    
    def bp(self, AI, AO, DO, cache):    
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        
        # DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.ss, out_backprop=DO, strides=self.strides, padding=self.padding)
        
        '''
        DI_bp = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        DI_ss = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.ss, out_backprop=DO, strides=self.strides, padding=self.padding)
        DI = DI_ss
        DI = DI - tf.reduce_mean(DI_ss)     + tf.reduce_mean(DI_bp)
        DI = DI / tf.math.reduce_std(DI_ss) * tf.math.reduce_std(DI_bp)
        '''

        '''
        FB = tf.cast(tf.greater(self.filters, np.ones_like(self.filters) * tf.reduce_mean(self.filters)), dtype=tf.float32) * tf.reduce_mean(self.filters)
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=FB, out_backprop=DO, strides=self.strides, padding=self.padding)
        '''

        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # DF = tf.Print(DF, [tf.count_nonzero(DF), tf.count_nonzero(self.filters), tf.shape(DF)], message='', summarize=1000)

        if self.use_bias:
            return {'dout':DI, 'cache':{}}, [(DF, self.filters), (DB, self.bias)]
        else:
            return {'dout':DI, 'cache':{}}, [(DF, self.filters)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ################################################################### 
        
        
