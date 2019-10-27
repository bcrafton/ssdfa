
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock
from lib.AvgPool import AvgPool

class DenseTransition(Layer):

    def __init__(self, input_shape, init, name):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.init = init
        self.name = name

        self.pool    = AvgPool(size=[self.batch, self.h, self.w, self.fin], ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    ###################################################################

    def get_weights(self):
        weights = []
        return weights

    def output_shape(self):
        assert(False)

    def num_params(self):
        return 0

    def forward(self, X):
        pool, pool_cache = self.pool.forward(X)
        cache = (pool, pool_cache)
        return pool, cache
        
    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        pool, pool_cache = cache
        dpool, gpool = self.pool.bp(AI, pool, DO, pool_cache)
        grads = []
        grads.extend(gpool)
        return dpool, grads

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
