
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops

from lib.Layer import Layer 

class UpSample(Layer):
    def __init__(self, input_shape, ksize):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.ksize = ksize

    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    ###################################################################

    def forward(self, X):
        if self.ksize > 1:
            A = tf.stack([X] * self.ksize, axis=3)
            A = tf.reshape(A, (self.batch, self.h, self.w, self.fin * self.ksize))
        else:
            A = X
          
        return {'aout':A, 'cache':{}}
        
    def backward(self, AI, AO, DO, cache=None):
        if self.ksize > 1:
            DI = tf.reshape(DO, [self.batch, self.h, self.w, self.fin, self.ksize])
            DI = tf.reduce_mean(DI, axis=4)
        else:
            DI = DO
            
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache=None):    
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
    
    def lel_backward(self, AI, AO, E, DO, Y):
        assert(False)
        
    def lel_gv(self, AI, AO, E, DO, Y):
        assert(False)
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################
    
    
