
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.init_tensor import init_matrix

from lib.quant import quantize_dense
from lib.quant import quantize_dense_bias
# from lib.quant import quantize_dense_activations
# from lib.quant import quantize_dense_activations2

class FullyConnected(Layer):

    def __init__(self, input_shape, size, init, bias, name, load=None, train=True):
        self.input_size = input_shape
        self.output_size = size
        self.init = init
        self.name = name
        self.train_flag = train

        bias = np.ones(shape=self.output_size) * bias
        weights = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        
        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        weights, _ = quantize_dense(self.weights)
        bias, _ = quantize_dense_bias(self.bias, self.weights)
        return [(self.name, weights), (self.name + "_bias", bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    ###################################################################

    def forward(self, X):
        qw, sw = quantize_dense(self.weights)
        qb, sb = quantize_dense_bias(self.bias, self.weights) 
        Z = tf.matmul(X, (qw * sw)) + (qb * sb)
        return Z, (sb,)

    def forward1(self, X):
        qw, sw = quantize_dense(self.weights)
        qb, sb = quantize_dense_bias(self.bias, self.weights) 
        Z = tf.matmul(X, qw) + qb
        return Z, (sb,)
        
    def forward2(self, X):
        qw, sw = quantize_dense(self.weights)
        qb, sb = quantize_dense_bias(self.bias, self.weights) 
        Z = tf.matmul(X, qw) + qb
        return Z, (sb,)

    ###################################################################
        
    def bp(self, AI, AO, DO, cache):
        DI = tf.matmul(DO, tf.transpose(self.weights))
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)
        return DI, [(DW, self.weights), (DB, self.bias)]

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ###################################################################
    
        
