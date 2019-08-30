
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvolutionDW import ConvolutionDW
from lib.Convolution import Convolution
from lib.BatchNorm import BatchNorm
from lib.Activation import Relu
from lib.Activation import SignedRelu

class ConvDWBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, init, name, load=None, train=True, fb='f'):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fh, self.fw, self.fin, self.mult = self.filter_shape
        self.fout = self.fin * self.mult

        self.strides = strides
        _, self.sh, self.sw, _ = self.strides

        self.output_shape = [self.batch, self.h // self.sh, self.w // self.sw, self.fout]

        self.init = init
        self.name = name
        self.load = load
        self.train_flag = train
        self.fb = fb
        
        self.conv = ConvolutionDW(input_shape=self.input_shape, 
                                  filter_sizes=self.filter_shape, 
                                  init=self.init, 
                                  strides=self.strides, 
                                  padding="SAME", 
                                  use_bias=False,
                                  name=self.name + '_conv_dw',
                                  load=self.load, 
                                  train=self.train_flag,
                                  fb=self.fb)
                                
        self.bn = BatchNorm(input_size=self.output_shape, name=self.name + '_bn_dw')
        signs = np.random.choice([1., -1.], size=self.fout) # np.array([1.] * (self.fout // 2) + [-1.] * (self.fout // 2))
        self.relu = SignedRelu(signs)

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.conv.get_weights())
        weights.extend(self.bn.get_weights())
        return weights

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return self.conv.num_params() + self.bn.num_params()

    def forward(self, X):
        conv = self.conv.forward(X)
        bn = self.bn.forward(conv['aout'])
        relu = self.relu.forward(bn['aout'])
        cache = {'conv':conv['aout'], 'bn':bn['aout'], 'relu':relu['aout']}
        return {'aout':relu['aout'], 'cache':cache}

    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        conv, bn, relu = cache['conv'], cache['bn'], cache['relu']
        drelu, grelu = self.relu.bp(bn, relu, DO, None)
        dbn,   gbn   = self.bn.bp(conv, bn, drelu, None)
        dconv, gconv = self.conv.bp(AI, conv, dbn, None)
        grads = []
        grads.extend(gconv)
        grads.extend(gbn)
        return dconv, grads
        
    def ss(self, AI, AO, DO, cache):    
        conv, bn, relu = cache['conv'], cache['bn'], cache['relu']
        drelu, grelu = self.relu.ss(bn, relu, DO, None)
        dbn,   gbn   = self.bn.ss(conv, bn, drelu, None)
        dconv, gconv = self.conv.ss(AI, conv, dbn, None)
        grads = []
        grads.extend(gconv)
        grads.extend(gbn)
        return dconv, grads

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
    
    
    
