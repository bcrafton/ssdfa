
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.FullyConnected import FullyConnected
from lib.BatchNorm import BatchNorm
from lib.Activation import Relu

class FCBlock(Layer):

    def __init__(self, input_shape, size, init, name, load=None, train=True):
        self.batch, self.input_size = input_shape
        self.output_size = size
        
        self.init = init
        self.name = name
        self.load = load
        self.train_flag = train

        self.dense= FullyConnected(input_shape=self.input_size, 
                                   size=self.output_size, 
                                   init=init, 
                                   use_bias=False, 
                                   name=self.name + '_dense', 
                                   load=self.load, 
                                   train=self.train_flag)
                                
        self.bn = BatchNorm(input_size=[self.batch, self.output_size], name=self.name + '_bn')
        self.relu = Relu()

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.dense.get_weights())
        weights.extend(self.bn.get_weights())
        return weights

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return self.dense.num_params() + self.bn.num_params()

    def forward(self, X):
        dense, _ = self.dense.forward(X)
        bn, _    = self.bn.forward(dense)
        relu, _  = self.relu.forward(bn)

        cache = (dense, bn, relu)
        return relu, cache

    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        dense, bn, relu = cache
        drelu, grelu   = self.relu.bp(bn, relu, DO, None)
        dbn,   gbn     = self.bn.bp(dense, bn, drelu, None)
        ddense, gdense = self.dense.bp(AI, dense, dbn, None)
        grads = []
        grads.extend(gdense)
        grads.extend(gbn)
        return ddense, grads
        
    def dfa(self, AI, AO, E, DO, cache):    
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
    
    
    
