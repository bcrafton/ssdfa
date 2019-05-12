
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Linear

from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

class SpikingSum(Layer):

    def __init__(self, input_shape, init, alpha=0., activation=None, bias=0., name=None, load=None, train=True):
        self.input_shape = input_shape
        self.batch, self.times, self.input_size = self.input_shape
        # need to make sure we are convolving time, not neuron!!!
        self.filter_sizes = [1, 1, self.times, 1]
        
        self.strides = 1
        self.padding = 'SAME'
        self.alpha = alpha
        self.activation = Linear() if activation == None else activation
        self.name = name
        self._train = train
        
        filters = np.random.normal(loc=1.0, scale=0.0, size=self.filter_sizes)
        self.filters = tf.Variable(filters, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters)]

    def output_shape(self):
        return [self.batch, self.input_size]

    def num_params(self):
        filter_weights_size = self.times
        return filter_weights_size 
                
    def forward(self, X):
        X = tf.reshape(X, (self.batch, 1, self.times, self.input_size))
        X = tf.transpose(X, (0, 1, 3, 2))
        
        Z = tf.nn.conv2d(input=X, filter=self.filters, strides=[1,1,1,1], padding=self.padding)
        Z = tf.reshape(Z, (self.batch, self.input_size))
        
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO): 
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (self.batch, 1, self.input_size, 1))
        
        _input_shape = [self.batch, 1, self.input_size, self.times]
        
        DI = tf.nn.conv2d_backprop_input(input_sizes=_input_shape, filter=self.filters, out_backprop=DO, strides=[1,1,1,1], padding=self.padding)
        DI = tf.reshape(DI, (self.batch, self.input_size, self.times))
        DI = tf.transpose(DI, (0, 2, 1))
        
        return DI

    def gv(self, AI, AO, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (self.batch, 1, self.input_size, 1))
        
        AI = tf.reshape(AI, (self.batch, 1, self.times, self.input_size))
        AI = tf.transpose(AI, (0, 1, 3, 2))
        
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=[1,1,1,1], padding=self.padding)
        DF = tf.reshape(DF, (1, 1, self.times, 1))
        
        return [(DF, self.filters)]
        
    def train(self, AI, AO, DO): 
        assert(False)
        return []
        
    ###################################################################

