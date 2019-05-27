
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

    ###################################################################           

    def forward(self, X):
        A = tf.reshape(X, [tf.shape(X)[0], -1])
        return {'aout':A, 'cache':{}}
        
    def backward(self, AI, AO, DO, cache=None):    
        DI = tf.reshape(DO, [tf.shape(AI)[0]] + self.shape)
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache=None):    
        return []
        
    def train(self, AI, AO, DO): 
        assert(False)
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################    
    
    def lel_backward(self, AI, AO, E, DO, Y):
        return self.backward(AI, AO, DO)
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return self.gv(AI, AO, DO)
        
    def lel(self, AI, AO, E, DO, Y): 
        return self.train(AI, AO, DO)
