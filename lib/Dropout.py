
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate

    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    def forward(self, X):
        self.dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(X)) > self.rate, tf.float32) # np.random.binomial(size=X.shape, n=1, p=1 - self.rate)
        A = X * self.dropout_mask
        return {'aout':A, 'cache':{}}

    ###################################################################

    def backward(self, AI, AO, DO, cache):
        DI = DO * self.dropout_mask
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache):
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO, cache):
        DI = DO * self.dropout_mask
        return {'dout':DI, 'cache':{}}

    def dfa_gv(self, AI, AO, E, DO, cache):
        return []

    def dfa(self, AI, AO, E, DO):
        return []

    ###################################################################

    def lel_backward(self, AI, AO, E, DO, Y, cache):
        DI = DO * self.dropout_mask
        return {'dout':DI, 'cache':{}}

    def lel_gv(self, AI, AO, E, DO, Y, cache):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
        

