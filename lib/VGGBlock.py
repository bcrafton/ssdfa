
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock
from lib.LELConv import LELConv

class VGGBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, init, pool_shape, num_classes, name, load=None, train=True):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides

        self.init = init

        self.pool_shape = pool_shape
        self.num_classes = num_classes
        
        self.name = name
        self.load = load
        self.train_flag = train

        self.lel_shape = [self.batch, self.h // self.sh, self.w // self.sw, self.fout]
        
        self.conv = ConvBlock(input_shape=self.input_shape, 
                              filter_shape=[3, 3, self.fin, self.fout], 
                              strides=self.strides, 
                              init=self.init, 
                              name=self.name + '_conv_block', 
                              load=self.load, 
                              train=self.train_flag)

        self.lel = LELConv(input_shape=self.lel_shape, pool_shape=self.pool_shape, num_classes=self.num_classes, name=self.name + '_lel')

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.conv.get_weights())
        # should add lel weights here, but we dont need them atm.
        return weights

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return self.conv.num_params()

    ###################################################################

    def forward(self, X):
        conv = self.conv.forward(X)
        lel = self.lel.forward(conv['aout'])

        cache = {'conv':conv, 'lel':lel}
        return {'aout':conv['aout'], 'cache':cache}
        
    def backward(self, AI, AO, DO, cache):    
        
        conv = cache['conv']
        
        ##########################################3
        
        dconv = self.conv.backward(AI, conv['aout'], DO, conv['cache'])

        ##########################################

        cache.update({'dconv':dconv})
        return {'dout':dconv['dout'], 'cache':cache}
        
    def gv(self, AI, AO, DO, cache):

        conv = cache['conv']
        dconv = cache['dconv']
        
        ##########################################

        dconv = self.conv.gv(AI, conv['aout'], DO, dconv['cache'])
        
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
    
        conv, lel = cache['conv'], cache['lel']
        
        ##########################################3
        
        dlel = self.lel.lel_backward(conv['aout'], lel['aout'], None, DO, Y, lel['cache'])
        dconv = self.conv.lel_backward(AI, conv['aout'], None, dlel['dout'], Y, conv['cache'])

        ##########################################

        cache.update({'dconv':dconv, 'dlel':dlel})
        return {'dout':dconv['dout'], 'cache':cache}

    def lel_gv(self, AI, AO, E, DO, Y, cache):

        conv, lel  = cache['conv'], cache['lel']
        dconv, dlel = cache['dconv'], cache['dlel']
        
        ##########################################

        dconv = self.conv.lel_gv(AI, conv['aout'], None, dlel['dout'], Y, dconv['cache'])
        dlel= self.lel.lel_gv(conv['aout'], lel['aout'], None, DO, Y, dlel['cache'])
        
        ##########################################
        
        grads = []
        grads.extend(dconv)
        grads.extend(dlel)
        return grads
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################   
    
    
    
    
