
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.init_tensor import init_matrix

def quantize_weights(w):
  # scale = (7. - (-8.)) / (tfp.stats.percentile(w, 95) - tfp.stats.percentile(w, 5))
  scale = (7. - (-8.)) / (tf.reduce_max(w) - tf.reduce_min(w))
  # scale = (7. - (-8.)) / (2 * tf.math.reduce_std(w))

  w = scale * w
  w = tf.floor(w)
  w = tf.clip_by_value(w, -8, 7)
  return w, scale

class FullyConnected(Layer):

    def __init__(self, input_shape, size, init, bias=0., use_bias=True, name=None, load=None, train=True):
        self.input_size = input_shape
        self.output_size = size
        self.init = init
        self.name = name
        self.train_flag = train
        self.use_bias = use_bias
        
        bias = np.ones(shape=self.output_size) * bias
        weights = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        
        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        if self.use_bias:
            return [(self.name, self.weights), (self.name + "_bias", self.bias)]
        else:
            return [(self.name, self.weights)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        if self.use_bias:
            return weights_size + bias_size
        else:
            return weights_size

    def forward(self, X):
        qw, sw = quantize_weights(self.weights)
        Z = tf.matmul(X, (qw / sw)) 
        if self.use_bias:
            qb, sb = quantize_weights(self.bias) 
            Z = Z + (qb / sb)
            
        return Z, (Z,)

    ###################################################################
        
    def bp(self, AI, AO, DO, cache):
        DI = tf.matmul(DO, tf.transpose(self.weights))
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)
        if self.use_bias:
            return DI, [(DW, self.weights), (DB, self.bias)]
        else:
            return DI, [(DW, self.weights)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ###################################################################
    
        
