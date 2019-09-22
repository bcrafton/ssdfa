
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock

class VGGBlock(Layer):

    def __init__(self, input_shape, filter_shape, init, name, load=None, train=True, fb='f_f', rate=0.5):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        
        self.strides = [1,1,1,1]
        _, self.sh, self.sw, _ = self.strides

        self.init = init
        self.name = name
        self.load = load
        self.train_flag = train
        self.fb = fb
        self.rate = rate
        
        self.conv = ConvBlock(input_shape=self.input_shape, 
                              filter_shape=[3, 3, self.fin, self.fout], 
                              strides=self.strides, 
                              init=self.init, 
                              name=self.name + '_conv_block', 
                              load=self.load, 
                              train=self.train_flag,
                              fb=self.fb,
                              rate=self.rate)

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

        cache = (conv, conv_cache)
        return conv, cache
        
    ###################################################################
        
    def bp(self, AI, AO, DO, cache):    
        conv, conv_cache = cache
        dconv, gconv = self.conv.bp(AI, conv, DO, conv_cache)
        grads = []
        grads.extend(gconv)
        return dconv, grads

    def ss(self, AI, AO, DO, cache):    
        conv, conv_cache = cache
        dconv, gconv = self.conv.ss(AI, conv, DO, conv_cache)
        grads = []
        grads.extend(gconv)
        return dconv, grads
        
    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
