
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvolutionDW import ConvolutionDW
from lib.Convolution import Convolution
from lib.BatchNorm import BatchNorm
from lib.Activation import Relu

class ConvDWBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, init, name):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fh, self.fw, self.fin, self.factor = self.filter_shape
        self.fout = self.fin * self.factor

        self.strides = strides
        _, self.sh, self.sw, _ = self.strides

        self.output_shape = [self.batch, self.h // self.sh, self.w // self.sw, self.fout]
        # print (self.output_shape, self.fout) 
        # print (self.filter_shape)

        self.init = init
        self.name = name
        
        # self.conv = Convolution(input_sizes=self.input_shape, filter_sizes=[self.fh, self.fw, self.fin, self.fout], init=self.init, strides=self.strides, padding="SAME", name=self.name + '_conv')
        self.conv = ConvolutionDW(input_sizes=self.input_shape, filter_sizes=self.filter_shape, init=self.init, strides=self.strides, padding="SAME", name=self.name + '_conv')
        self.bn = BatchNorm(input_size=self.output_shape, name=self.name + '_bn')
        self.relu = Relu()

    ###################################################################

    def get_weights(self):
        return []

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return 0

    ###################################################################

    def forward(self, X):

        conv = self.conv.forward(X)
        bn = self.bn.forward(conv['aout'])
        relu = self.relu.forward(bn['aout'])

        cache = {'conv':conv['aout'], 'bn':bn['aout'], 'relu':relu['aout']}
        return {'aout':relu['aout'], 'cache':cache}
        
    def backward(self, AI, AO, DO, cache):    
        conv, bn, relu = cache['conv'], cache['bn'], cache['relu']
        
        drelu = self.relu.backward(bn, relu, DO)
        dbn = self.bn.backward(conv, bn, drelu['dout'])
        dconv = self.conv.backward(AI, conv, dbn['dout'])
        
        cache.update({'dconv':dconv['dout'], 'dbn':dbn['dout'], 'drelu':drelu['dout']})
        return {'dout':dconv['dout'], 'cache':cache}
        
    def gv(self, AI, AO, DO, cache):
        conv, bn, relu = cache['conv'], cache['bn'], cache['relu']
        drelu, dbn, dconv = cache['drelu'], cache['dbn'], cache['dconv']
        
        dconv = self.conv.gv(AI, conv, dbn)
        dbn = self.bn.gv(conv, bn, drelu)
        
        grads = []
        grads.extend(dconv)
        grads.extend(dbn)
        
        return grads
        
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
        return self.backward(AI, AO, DO)

    def lel_gv(self, AI, AO, E, DO, Y):
        return self.gv(AI, AO, DO)
        
    def lel(self, AI, AO, E, DO, Y): 
        return self.train(AI, AO, DO)
        
    ###################################################################   
    
    
    
    
    
    
    
