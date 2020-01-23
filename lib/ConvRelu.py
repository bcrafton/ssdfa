
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.Convolution import Convolution
from lib.Activation import Relu

class ConvRelu(Layer):

    def __init__(self, input_shape, filter_shape, strides, init, name, minval, maxval, load=None, train=True):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_shape
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        
        self.output_shape = [self.batch, self.h // self.sh, self.w // self.sw, self.fout]
        
        self.init = init
        self.name = name
        self.load = load
        self.train_flag = train
        
        self.conv = Convolution(input_shape=self.input_shape, 
                                filter_sizes=self.filter_shape, 
                                init=self.init, 
                                strides=self.strides, 
                                padding="SAME", 
                                use_bias=True,
                                name=self.name, 
                                load=self.load, 
                                train=self.train_flag)
                                
        self.relu = Relu(minval=minval, maxval=maxval)

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.conv.get_weights())
        return weights

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return self.conv.num_params()

    def forward(self, X):
        conv, conv_cache = self.conv.forward(X)
        relu, relu_cache = self.relu.forward(conv)

        cache = (conv, conv_cache, relu, relu_cache)
        return relu, cache
        
    def forward1(self, X):
        conv, conv_cache = self.conv.forward1(X)
        relu, relu_cache = self.relu.forward1(conv)

        cache = (conv, conv_cache, relu, relu_cache)
        return relu, cache

    def forward2(self, X):
        conv, conv_cache = self.conv.forward2(X)
        relu, relu_cache = self.relu.forward2(conv)

        cache = (conv, conv_cache, relu, relu_cache)
        return relu, cache

    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        conv, conv_cache, relu, relu_cache = cache
        drelu, grelu = self.relu.bp(conv, relu, DO, relu_cache)
        dconv, gconv = self.conv.bp(AI, conv, drelu, conv_cache)
        grads = []
        grads.extend(gconv)
        return dconv, grads
        
    def dfa(self, AI, AO, E, DO, cache):    
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
    
    
    
