
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Linear

from lib.init_tensor import init_matrix

class FullyConnected(Layer):

    def __init__(self, input_shape, size, init, activation=None, bias=0., use_bias=True, name=None, load=None, train=True):
        self.input_size = input_shape
        self.output_size = size
        self.init = init
        self.activation = Linear() if activation == None else activation
        self.name = name
        self.train_flag = train
        self.use_bias = use_bias

        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1', allow_pickle=True).item()
            weights = weight_dict[self.name]
            bias = weight_dict[self.name + '_bias']
        else:       
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

    def forward(self, X):
        Z = tf.matmul(X, self.weights) 
        if self.use_bias:
            Z = Z + self.bias
        A = self.activation.forward(Z)
        return {'aout':A, 'cache':{}}

    ###################################################################
        
    def bp(self, AI, AO, DO, cache):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights))
        
        DW = tf.matmul(tf.transpose(AI), DO) 
        DB = tf.reduce_sum(DO, axis=0)
        
        if self.train_flag:
            return {'dout':DI, 'cache':{}}, [(DW, self.weights), (DB, self.bias)]
        else:
            return {'dout':DI, 'cache':{}}, []

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ###################################################################
    
        
