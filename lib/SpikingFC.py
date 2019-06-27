
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class SpikingFC(Layer):

    def __init__(self, input_shape, size, init, activation, alpha=0., name=None, load=None, train=True):

        self.batch, self.times, self.input_size = input_shape
        self.output_size = size
        self.size = [self.input_size, self.output_size]
        self.alpha = alpha
        self.activation = activation
        self.name = name
        self._train = train
        
        sqrt_fan_in = math.sqrt(self.input_size)
        weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.size)
        self.weights = tf.Variable(weights, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        return weights_size

    def forward(self, X):
        # 50 8 784 64 | X
        # 1  1 784 64 | weights
        weights = tf.reshape(self.weights, (1, 1, self.input_size, self.output_size))
        Z = X * weights
        Z = tf.reduce_sum(Z, axis=2)
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        # bc we just testing 1 fc and fc is first.
        return AI
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        # 50 8  1   64
        # 1  1  784 64
        DO = tf.reshape(DO, (self.batch, self.times, 1, self.output_size))
        weights = tf.reshape(self.weights, (1, 1, self.input_size, self.output_size))
        DI = DO * weights
        return DI
        
    def gv(self, AI, AO, DO):
        if not self._train:
            return []
            
        DO = tf.multiply(DO, self.activation.gradient(AO))

        DO = tf.reshape(DO, (self.batch, self.times, 1, self.output_size))
        DW = AI * DO
        DW = tf.reduce_sum(DW, axis=[0, 1])

        return [(DW, self.weights)]

    def train(self, AI, AO, DO):
        assert(False)
        return []
        
    ###################################################################

        
        
