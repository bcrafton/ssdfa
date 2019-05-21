
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.Convolution import Convolution
from lib.BatchNorm import BatchNorm
from lib.Activation import Relu

class ConvBlock(Layer):

    def __init__(self, input_shape, filter_shape, init, name):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_shape
        
        self.output_shape = [self.batch, self.h, self.w, self.fout]
        
        self.init = init
        self.name = name
        
        self.conv = Convolution(input_sizes=self.input_shape, filter_sizes=self.filter_shape, init=self.init, strides=[1,1,1,1], padding="SAME", name=self.name + '_conv')
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
        bn   = self.bn.forward(conv)
        relu = self.relu.forward(bn)
        return [conv, bn, relu]
        
    def backward(self, AI, AO, DO):    
        [conv, bn, relu] = AO
        
        drelu = self.relu.backward(bn, relu, DO)
        dbn = self.bn.backward(conv, bn, drelu)
        dconv = self.conv.backward(AI, conv, dbn)
        
        return [dconv, dbn, drelu]
        
    def gv(self, AI, AO, DO):    
        [conv, bn, relu] = AO
        
        drelu = self.relu.backward(bn, relu, DO)
        dbn = self.bn.backward(conv, bn, drelu)
        dconv = self.conv.backward(AI, conv, dbn)
        
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
        assert(False)

    def lel_gv(self, AI, AO, E, DO, Y):
        assert(False)
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################   
    
    
    
    
    
    
    
