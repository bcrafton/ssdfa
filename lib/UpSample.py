
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops

from lib.Layer import Layer 

class UpSample(Layer):
    def __init__(self, size, ksize):
        self.size = size
        self.batch_size, self.h, self.w, self.fin = self.size
        self.ksize = ksize

    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    ###################################################################

    def forward(self, X):
        if self.ksize > 1:
            ret = tf.stack([X] * self.ksize, axis=3)
            ret = tf.reshape(ret, (self.batch, self.h, self.w, self.fin * self.ksize))
        else:
            ret = X
            
        return ret
        
    def backward(self, AI, AO, DO, cache=None):
        if self.ksize > 1:
            dout = tf.reshape(DO, [self.batch, self.h, self.w, self.fin, self.ksize])
            dout = tf.reduce_mean(dout, axis=4)
        else:
            dout = DO
            
        return dout

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
    
    
