
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Linear

from lib.init_tensor import init_matrix

class FullyConnected(Layer):

    def __init__(self, input_shape, size, init, activation=None, bias=0., use_bias=False, name=None, load=None, train=True):
        self.input_size = input_shape
        self.output_size = size
        self.init = init
        self.activation = Linear() if activation == None else activation
        self.name = name
        self.train_flag = train
        self.use_bias = use_bias
        
        bias = np.ones(shape=self.output_size) * bias
        weights = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        
        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    ###################################################################

    def forward(self, X):
        Z = tf.matmul(X, self.weights) 
        if self.use_bias:
            Z = Z + self.bias
        A = self.activation.forward(Z)
        return {'aout':A, 'cache':{}}
            
    def backward(self, AI, AO, DO, cache=None):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights))
        return {'dout':DI, 'cache':{}}
        
    def gv(self, AI, AO, DO, cache=None):
        if not self.train_flag:
            return []
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones_like(AI)
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self.train_flag:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)
        
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
        
    def lel_backward(self, AI, AO, DO, Y, cache):
        return self.backward(AI, AO, DO, cache)

    def lel_gv(self, AI, AO, DO, Y, cache):
        return self.gv(AI, AO, DO, cache)
        

        
