
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, input_shape):
        self.shape = input_shape
        
    ###################################################################

    def get_weights(self):
        return []
        
    def output_shape(self):
        return np.prod(self.shape)
        
    def num_params(self):
        return 0

    def forward(self, X):
        A = tf.reshape(X, [tf.shape(X)[0], -1])
        return {'aout':A, 'cache':{}}
    
    ###################################################################
        
    def backward(self, AI, AO, DO, cache):
        DI = tf.reshape(DO, [tf.shape(AI)[0]] + self.shape)
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache):    
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO, cache):
        return self.backward(AI, AO, DO, cache)
        
    def dfa_gv(self, AI, AO, E, DO, cache):
        return self.gv(AI, AO, DO, cache)
        
    ###################################################################    
    
    def lel_backward(self, AI, AO, DO, Y, cache):
        return self.backward(AI, AO, DO, cache)
        
    def lel_gv(self, AI, AO, DO, Y, cache):
        return self.gv(AI, AO, DO, cache)
