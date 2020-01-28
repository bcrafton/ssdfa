
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.Dense import Dense
from lib.Activation import Relu

class DenseRelu(Layer):

    def __init__(self, input_shape, size, init, name, scale, load=None, train=True):
        self.dense = Dense(input_shape=input_shape, size=size, init=init, bias=0.1, name=name)
        self.relu = Relu(scale=scale)

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.dense.get_weights())
        return weights

    def output_shape(self):
        return self.size

    def num_params(self):
        return self.dense.num_params()

    def forward(self, X):
        dense, dense_cache = self.dense.forward(X)
        relu, relu_cache = self.relu.forward(dense)

        cache = (dense, dense_cache, relu, relu_cache)
        return relu, cache
        
    def forward1(self, X):
        dense, dense_cache = self.dense.forward1(X)
        relu, relu_cache = self.relu.forward1(dense)

        cache = (dense, dense_cache, relu, relu_cache)
        return relu, cache

    def forward2(self, X):
        dense, dense_cache = self.dense.forward2(X)
        relu, relu_cache = self.relu.forward2(dense)

        cache = (dense, dense_cache, relu, relu_cache)
        return relu, cache

    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        dense, dense_cache, relu, relu_cache = cache
        drelu, grelu = self.relu.bp(dense, relu, DO, relu_cache)
        ddense, gdense = self.dense.bp(AI, dense, drelu, dense_cache)
        grads = []
        grads.extend(gdense)
        return ddense, grads
        
    def dfa(self, AI, AO, E, DO, cache):    
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
    
    
    
