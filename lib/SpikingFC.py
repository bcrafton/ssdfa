
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
        Z = tf.keras.backend.dot(X, self.weights)
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.keras.backend.dot(DO, tf.transpose(self.weights))
        return DI
        
    def gv(self, AI, AO, DO):
        if not self._train:
            return []
            
        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        # put the gradient wrt 1D conv here.
        
        # do we want more or less spikes is the question... so we just sum along the time dimension I guess.
        # AI = [B T NI]
        # DO = [B T NO]
        AI = tf.reshape(AI, (self.batch * self.times, self.input_size))
        AI = tf.transpose(AI)
        DO = tf.reshape(DO, (self.batch * self.times, self.output_size))
        
        DW = tf.matmul(AI, DO) 

        return [(DW, self.weights)]

    def train(self, AI, AO, DO):
        assert(False)
        return []
        
    ###################################################################

        
        
