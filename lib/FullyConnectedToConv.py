
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class FullyConnectedToConv(Layer):

    def __init__(self, input_shape, output_shape):
        self.shape_in = shape_in
        self.batch, self.size = self.input_shape
        self.shape_out = shape_out
        tmp, self.h, self.w, self.c = self.output_shape
        assert(self.batch == tmp)
        
    ###################################################################

    def get_weights(self):
        return []
        
    def num_params(self):
        return 0

    def forward(self, X):
        A = tf.reshape(X, self.output_shape)
        return {'aout':A, 'cache':{}}
        
    ###################################################################           
        
    def backward(self, AI, AO, DO, cache):
        DI = tf.reshape(DO, self.input_shape)
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache):
        return []
        
    def train(self, AI, AO, DO): 
        assert(False)
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        assert(False)
        
    def dfa_gv(self, AI, AO, E, DO):
        assert(False)
        
    def dfa(self, AI, AO, E, DO): 
        assert(False)
        
    ###################################################################    
    
    def lel_backward(self, AI, AO, E, DO, Y, cache):
        assert(False)
        
    def lel_gv(self, AI, AO, E, DO, Y, cache):
        assert(False)
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)

