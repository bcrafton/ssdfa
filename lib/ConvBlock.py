
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.Convolution import Convolution
from lib.BatchNorm import BatchNorm
from lib.Activation import Relu
from lib.Activation import SignedRelu

class ConvBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, init, name, load=None, train=True, fb='f_f', rate=0.5):
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
        self.fb = fb
        self.rate = rate
        
        self.conv = Convolution(input_shape=self.input_shape, 
                                filter_sizes=self.filter_shape, 
                                init=self.init, 
                                strides=self.strides, 
                                padding="SAME", 
                                use_bias=False,
                                name=self.name + '_conv', 
                                load=self.load, 
                                train=self.train_flag,
                                fb=self.fb, 
                                rate=self.rate)
                                
        self.bn = BatchNorm(input_size=self.output_shape, name=self.name + '_bn', load=self.load)
        self.relu = SignedRelu(size=self.fout, name=self.name + '_relu', load=self.load)

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.conv.get_weights())
        weights.extend(self.bn.get_weights())
        weights.extend(self.relu.get_weights())
        return weights

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return self.conv.num_params() + self.bn.num_params()

    def forward(self, X):
        conv, conv_cache = self.conv.forward(X)
        bn, bn_cache   = self.bn.forward(conv)
        relu, relu_cache = self.relu.forward(bn)

        cache = (conv, conv_cache, bn, bn_cache, relu, relu_cache)
        return relu, cache

    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        conv, conv_cache, bn, bn_cache, relu, relu_cache = cache
        drelu, drelus, grelu = self.relu.bp(bn, relu, DO, conv_cache)
        dbn,   dbns,   gbn   = self.bn.bp(conv, bn, drelu, bn_cache)
        dconv, dconvs, gconv = self.conv.bp(AI, conv, dbn, relu_cache)

        deriv = dconvs + dbns + drelus
        grads = gconv + gbn + grelu
        return dconv, deriv, grads

    def ss(self, AI, AO, DO, cache):    
        conv, conv_cache, bn, bn_cache, relu, relu_cache = cache
        drelu, drelus, grelu = self.relu.ss(bn, relu, DO, conv_cache)
        dbn,   dbns,   gbn   = self.bn.ss(conv, bn, drelu, bn_cache)
        dconv, dconvs, gconv = self.conv.ss(AI, conv, dbn, relu_cache)

        deriv = dconvs + dbns + drelus
        grads = gconv + gbn + grelu
        return dconv, deriv, grads
        
    def dfa(self, AI, AO, E, DO, cache):    
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
    
    
    
