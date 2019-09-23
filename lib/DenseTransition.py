
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock
from lib.AvgPool import AvgPool

class DenseTransition(Layer):

    def __init__(self, input_shape, init, name, fb):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.init = init
        self.name = name
        self.fb = fb

        self.conv1x1 = ConvBlock(input_shape=self.input_shape, filter_shape=[1, 1, self.fin, self.fin], strides=[1,1,1,1], init=self.init, name=self.name + '_conv1x1_block', fb=self.fb)
        self.pool    = AvgPool(size=[self.batch, self.h, self.w, self.fin], ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.conv1x1.get_weights())
        return weights

    def output_shape(self):
        assert(False)

    def num_params(self):
        return self.conv1x1.num_params()

    def forward(self, X):
        conv1x1, conv1x1_cache = self.conv1x1.forward(X)
        pool, pool_cache = self.pool.forward(conv1x1)
        cache = (conv1x1, conv1x1_cache, pool, pool_cache)
        return pool, cache
        
    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        conv1x1, conv1x1_cache, pool, pool_cache = cache
        dpool, gpool = self.pool.bp(conv1x1, pool, DO, pool_cache)
        dconv1x1, gconv1x1 = self.conv1x1.bp(AI, conv1x1, dpool, conv1x1_cache)
        grads = []
        grads.extend(gconv1x1)
        grads.extend(gpool)
        return dconv1x1, grads

    def ss(self, AI, AO, DO, cache):    
        conv1x1, conv1x1_cache, pool, pool_cache = cache
        dpool, gpool = self.pool.ss(conv1x1, pool, DO, pool_cache)
        dconv1x1, gconv1x1 = self.conv1x1.ss(AI, conv1x1, dpool, conv1x1_cache)
        grads = []
        grads.extend(gconv1x1)
        grads.extend(gpool)
        return dconv1x1, grads

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
