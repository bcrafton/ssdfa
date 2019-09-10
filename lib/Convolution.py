
import tensorflow as tf
import numpy as np

from lib.Layer import Layer
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length
from lib.init_tensor import init_filters

class Convolution(Layer):

    def __init__(self, input_shape, filter_sizes, init, strides=[1,1,1,1], padding='SAME', bias=0., use_bias=True, name=None, load=None, train=True, fb='f'):
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
        self.fb = fb
        
        filters = np.absolute(init_filters(size=self.filter_sizes, init=self.init))
        bias = np.ones(shape=self.fout) * bias
        mask = np.random.choice([0., 1.], size=[self.fin, self.fout], replace=True)

        self.filters = tf.Variable(filters, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        if self.use_bias:
            self.bias = tf.Variable(bias, dtype=tf.float32)
        self.mask = tf.Variable(mask, dtype=tf.float32)

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
        if self.fb   == 'ud01f':
            mask = tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0,1], keep_dims=True)), dtype=tf.float32)
        elif self.fb == 'ud012f':
            mask = tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0,1,2], keep_dims=True)), dtype=tf.float32)
        elif self.fb == 'ud0123f':
            mask = tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0,1,2,3], keep_dims=True)), dtype=tf.float32)
        elif self.fb == 'udc01f':
            mask = self.mask
        else:
            mask = tf.ones_like(self.filters)

        Z = tf.nn.conv2d(X, self.filters * mask, self.strides, self.padding)
        if self.use_bias:
            Z = Z + self.bias

        return {'aout':Z, 'cache':{}}

    ###################################################################
    
    def bp(self, AI, AO, DO, cache):    
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        if self.fb == 'udc01f':
            DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters * self.mask, out_backprop=DO, strides=self.strides, padding=self.padding)
        else:
            DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)

        if self.use_bias:
            return DI, [(DF, self.filters), (DB, self.bias)]
        else:
            return DI, [(DF, self.filters)]

    def ss(self, AI, AO, DO, cache):    
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        if   self.fb == 'f':
            ss = self.filters
        elif self.fb == 'u01':
            ss = tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1], keep_dims=True)
        elif self.fb == 'u012':
            ss = tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1, 2], keep_dims=True)
        elif self.fb == 'u0123':
            ss = tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1, 2, 3], keep_dims=True)
        elif self.fb == 'ud01' or self.fb == 'ud01f':
            mask = tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1], keep_dims=True)), dtype=tf.float32)
            ss = mask * 2. * tf.reduce_mean(self.filters, axis=[0, 1], keep_dims=True)
        elif self.fb == 'ud012' or self.fb == 'ud012f':
            mask = tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1, 2], keep_dims=True)), dtype=tf.float32)
            ss = mask * 2. * tf.reduce_mean(self.filters, axis=[0, 1, 2], keep_dims=True)
        elif self.fb == 'ud0123' or self.fb == 'ud0123f':
            mask = tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1, 2, 3], keep_dims=True)), dtype=tf.float32)
            ss = mask * 2. * tf.reduce_mean(self.filters, axis=[0, 1, 2, 3], keep_dims=True)
        elif self.fb == 'udc01f':
            mask = self.mask * tf.ones_like(self.filters)
            ss = mask * 2. * tf.reduce_mean(self.filters, axis=[0, 1], keep_dims=True)
        else:
            assert(False)
            
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=ss, out_backprop=DO, strides=self.strides, padding=self.padding)

        if self.use_bias:
            return DI, [(DF, self.filters), (DB, self.bias)]
        else:
            return DI, [(DF, self.filters)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ################################################################### 
        
        
