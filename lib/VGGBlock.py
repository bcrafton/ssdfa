
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock
from lib.LELConv import LELConv

class VGGBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, pool_shape=[1,1,1,1], num_classes=1000, init='alexnet', name='VGGBlock'):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides

        self.pool_shape = pool_shape

        self.init = init
        self.name = name
        
        self.conv = ConvBlock(input_shape=self.input_shape, filter_shape=[3, 3, self.fin, self.fout], strides=self.strides, init=self.init, name='_conv_block')
        # self.lel = LELConv(input_shape=input_shape_3, pool_shape=self.pool_shape, num_classes=1000, name='_fb')

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

        cache = {'conv':conv}
        return {'aout':conv['aout'], 'cache':cache}
        
    def backward(self, AI, AO, DO, cache):    
        
        conv = cache['conv']
        
        ##########################################3
        
        dconv = self.conv_dw.backward(AI, conv['aout'], DO, conv['cache'])

        ##########################################

        cache.update({'dconv':dconv})
        return {'dout':dconv['dout'], 'cache':cache}
        
    def gv(self, AI, AO, DO, cache):

        conv = cache['conv']
        dconv = cache['dconv']
        
        ##########################################

        dconv = self.conv_dw.gv(AI, conv['aout'], DO, dconv['cache'])
        
        ##########################################
        
        grads = []
        grads.extend(dconv)
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
    
    def lel_backward(self, AI, AO, E, DO, Y, cache):
        assert(False)

    def lel_gv(self, AI, AO, E, DO, Y, cache):
        assert(False)
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################   
    
    
    
    
