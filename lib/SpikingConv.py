
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Linear

from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

class SpikingConv(Layer):

    def __init__(self, input_shape, filter_sizes, init, strides, padding, alpha=0., activation=None, bias=0., name=None, load=None, train=True):
        self.input_shape = input_shape
        self.filter_sizes = filter_sizes
        self.batch, self.times, self.h, self.w, self.fin = self.input_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        self.padding = padding
        self.oh = conv_output_length(self.h, self.fh, self.padding.lower(), self.sh)
        self.ow = conv_output_length(self.w, self.fw, self.padding.lower(), self.sw)
        self.od = self.fout

        self.alpha = alpha
        self.activation = Linear() if activation == None else activation
        self.name = name
        self._train = train
        
        sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
        filters = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.filter_sizes)
        self.filters = tf.Variable(filters, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters)]

    def output_shape(self):
        return [self.oh, self.oh, self.od]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        return filter_weights_size
                
    def forward(self, X):
        X = tf.reshape(X, (self.batch * self.times, self.h, self.w, self.fin))
        Z = tf.nn.conv2d(X, self.filters, self.strides, self.padding)
        Z = tf.reshape(Z, (self.batch, self.times, self.oh, self.ow, self.od))
        A = self.activation.forward(Z)
        return A

    def backward(self, AI, AO, DO):    
        input_sizes = (self.batch * self.times, self.h, self.w, self.fin)
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (self.batch * self.times, self.oh, self.ow, self.od))
        
        DI = tf.nn.conv2d_backprop_input(input_sizes=input_sizes, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        DI = tf.reshape(DI, (self.batch, self.times, self.h, self.w, self.fin))
        return DI

    def gv(self, AI, AO, DO):    
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (self.batch * self.times, self.oh, self.ow, self.od))
        AI = tf.reshape(AI, (self.batch * self.times, self.h, self.w, self.fin))
        
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        return [(DF, self.filters)]
        
    def train(self, AI, AO, DO): 
        assert(False)
        return []
        
    ###################################################################

        
        
