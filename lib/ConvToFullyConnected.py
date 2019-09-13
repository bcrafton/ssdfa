
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape        
      
    ###################################################################

    def get_weights(self):
        return []
        
    def output_shape(self):
        return self.h * self.w * self.fin
        
    def num_params(self):
        return 0

    def forward(self, X):
        A = tf.reshape(X, [self.batch, self.h * self.w * self.fin])
        return A, None
    
    ###################################################################
        
    def bp(self, AI, AO, DO, cache):
        DI = tf.reshape(DO, [self.batch, self.h, self.w, self.fin])
        return DI, []

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ###################################################################
