
import tensorflow as tf
import numpy as np

from lib.Layer import Layer
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length
from lib.init_tensor import init_filters

class Convolution(Layer):

    def __init__(self, input_shape, filter_sizes, init, strides=[1,1,1,1], padding='SAME', bias=0., use_bias=True, name=None, load=None, train=True, fb='f_f', rate=0.5):
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
        self.rate = rate

        [self.fb_mask, self.fb_kernel] = self.fb.split('_')

        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1', allow_pickle=True).item()
            filters = weight_dict[self.name]
            if self.use_bias:
                bias = weight_dict[self.name + '_bias']
            mask = 1.0 * (filters > np.mean(filters, axis=(0, 1), keepdims=False))

            self.train_flag = False
        else:
            filters = np.absolute(init_filters(size=self.filter_sizes, init=self.init))
            bias = np.ones(shape=self.fout) * bias    
            mask = np.random.choice([0., 1.], size=[self.fin, self.fout], replace=True, p=[1. - self.rate, self.rate])

        # filters = np.absolute(filters)
        assert (np.all(filters >= 0.))

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

    ###################################################################

    def get_mask(self):
        if   self.fb_mask == 'f':
            return tf.ones_like(self.filters)
        elif self.fb_mask == 'mask01':
            return tf.ones_like(self.filters) * self.mask
        elif self.fb_mask == 'mean01':
            return tf.ones_like(self.filters) * tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1], keep_dims=True)), dtype=tf.float32) 
        elif self.fb_mask == 'mean012':
            return tf.ones_like(self.filters) * tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1, 2], keep_dims=True)), dtype=tf.float32)
        elif self.fb_mask == 'mean0123':
            return tf.ones_like(self.filters) * tf.cast(tf.greater(self.filters, tf.ones_like(self.filters) * tf.reduce_mean(self.filters, axis=[0, 1, 2, 3], keep_dims=True)), dtype=tf.float32)
        else:
            assert (False)

    def get_ss(self):
        mask = self.get_mask()

        if   self.fb_kernel == 'f':
            return self.filters
        elif self.fb_kernel == 'mean01':
            return mask * tf.reduce_mean(self.filters * mask, axis=[0, 1], keep_dims=True)
        elif self.fb_kernel == 'mean012':
            return mask * tf.reduce_mean(self.filters * mask, axis=[0, 1, 2], keep_dims=True)
        elif self.fb_kernel == 'mean0123':
            return mask * tf.reduce_mean(self.filters * mask, axis=[0, 1, 2, 3], keep_dims=True)
        else:
            assert (False)

    def forward(self, X):

        mask = self.get_mask()

        Z = tf.nn.conv2d(X, self.filters * mask, self.strides, self.padding)
        if self.use_bias:
            Z = Z + self.bias
        return Z, None

    def bp(self, AI, AO, DO, cache): 

        # assert (False)

        mask = self.get_mask()
   
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=self.filters * mask, out_backprop=DO, strides=self.strides, padding=self.padding)
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding) * mask
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        if not self.train_flag:
            return DI, [DI], []
        elif self.use_bias:
            return DI, [DI], [(DF, self.filters), (DB, self.bias)]
        else:
            return DI, [DI], [(DF, self.filters)]

    def ss(self, AI, AO, DO, cache):

        ss = self.get_ss()
        mask = self.get_mask()
            
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_shape, filter=ss, out_backprop=DO, strides=self.strides, padding=self.padding)
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding) * mask
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        if not self.train_flag:
            return DI, [DI], []
        if self.use_bias:
            return DI, [DI], [(DF, self.filters), (DB, self.bias)]
        else:
            return DI, [DI], [(DF, self.filters)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ################################################################### 
        
        
