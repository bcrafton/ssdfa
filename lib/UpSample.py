
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
        A = X
        if self.ksize > 1:
            A = tf.reshape(A, (self.batch, self.h * self.w, 1, self.fin))
            A = tf.tile(A, [1, 1, self.ksize * self.ksize, 1])
            A = tf.reshape(A, (self.batch, self.h, self.w, self.ksize, self.ksize, self.fin))
            A = tf.reshape(A, (self.batch, self.h, self.w * self.ksize, self.ksize, self.fin))
            A = tf.transpose(A, (0, 1, 3, 2, 4))
            A = tf.reshape(A, (self.batch, self.h * self.ksize, self.w * self.ksize, self.fin))
           
        return {'aout':A, 'cache':{}}
        
    def backward(self, AI, AO, DO, cache=None):
        DI = DO
        if self.ksize > 1:
            DI = tf.reshape(DI, (self.batch, self.h, self.ksize, self.w, self.ksize, self.fin))
            DI = tf.transpose(DI, (0, 1, 3, 2, 4, 5))
            DI = tf.reshape(DI, (self.batch, self.h, self.w, self.ksize * self.ksize, self.fin))
            DI = tf.reduce_mean(DI, axis=3)
            
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
    
    
