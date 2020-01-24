
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.init_tensor import init_matrix

def quantize_weights(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (7. - (-8.))
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, -8, 7)
  return w, scale

def quantize_bias(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (1023. - (-1024.))
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, -1024, 1023)
  return w, scale

def quantize_activations(a):
  scale = (tf.reduce_max(a) - tf.reduce_min(a)) / (7. - (-8.))
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, -8, 7)
  return a, scale
  
def quantize_activations2(a, scale):
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, -8, 7)
  return a, scale

class FullyConnected(Layer):

    def __init__(self, input_shape, size, init, bias, use_bias, name, scale, load=None, train=True):
        self.input_size = input_shape
        self.output_size = size
        self.init = init
        self.name = name
        self.train_flag = train
        self.use_bias = use_bias
        assert(self.use_bias == True)
        self.scale = scale

        bias = np.ones(shape=self.output_size) * bias
        weights = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        
        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        weights, _ = quantize_weights(self.weights)
        bias, _ = quantize_bias(self.bias)
        return [(self.name, weights), (self.name + "_bias", bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        if self.use_bias:
            return weights_size + bias_size
        else:
            return weights_size

    ###################################################################

    def forward(self, X):
        qw, sw = quantize_weights(self.weights)
        qb, sb = quantize_bias(self.bias) 
        Z = tf.matmul(X, (qw * sw)) # + (qb * sb)
        Z, scale = quantize_activations(Z)
        Z = Z * scale
        return Z, (scale,)

    def forward1(self, X):
        qw, sw = quantize_weights(self.weights)
        qb, sb = quantize_bias(self.bias) 
        Z = tf.matmul(X, qw) # + qb
        Z, scale = quantize_activations(Z)
        # Z = Z * scale
        return Z, (scale,)
        
    def forward2(self, X):
        qw, sw = quantize_weights(self.weights)
        qb, sb = quantize_bias(self.bias) 
        Z = tf.matmul(X, qw) # + qb
        Z, scale = quantize_activations2(Z, self.scale)
        # Z = Z * scale
        return Z, (scale,)

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
    
        
