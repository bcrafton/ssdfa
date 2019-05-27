
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock
from lib.ConvDWBlock import ConvDWBlock
from lib.LELConv import LELConv

class MobileBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, pool_shape, num_classes=1000, init='alexnet', name='MobileBlock'):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        
        self.pool_shape = pool_shape
        
        self.init = init
        self.name = name
        
        input_shape_1 = [self.batch, self.h,            self.w,            self.fin]
        input_shape_2 = [self.batch, self.h // self.sh, self.w // self.sw, self.fin]
        input_shape_3 = [self.batch, self.h // self.sh, self.w // self.sw, self.fout]
        
        self.conv_dw = ConvDWBlock(input_shape=input_shape_1, filter_shape=[3, 3, self.fin, 1], strides=self.strides, init=self.init, name='_conv_block_dw')
        self.conv_pw = ConvBlock(input_shape=input_shape_2, filter_shape=[1, 1, self.fin, self.fout], strides=[1,1,1,1], init=self.init, name='_conv_block_pw')
        self.lel = LELConv(input_shape=input_shape_3, pool_shape=self.pool_shape, num_classes=1000, name='_fb')
        
    ###################################################################

    def get_weights(self):
        return []

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return 0

    ###################################################################

    def forward(self, X):

        conv_dw = self.conv_dw.forward(X)
        conv_pw = self.conv_pw.forward(conv_dw['aout'])
        lel = self.lel.forward(conv_pw['aout'])

        cache = {'conv_dw':conv_dw, 'conv_pw':conv_pw, 'lel':lel}
        return {'aout':conv_pw['aout'], 'cache':cache}
        
    def backward(self, AI, AO, DO, cache):    
        
        conv_dw, conv_pw, lel = cache['conv_dw'], cache['conv_pw'], cache['lel']
        
        ##########################################3
        
        dlel     = self.lel.backward(    conv_pw['aout'], lel['aout'],     DO,               lel['cache'])
        dconv_pw = self.conv_pw.backward(conv_dw['aout'], conv_pw['aout'], lel['dout'],      conv_pw['cache'])
        dconv_dw = self.conv_dw.backward(AI,              conv_dw['aout'], dconv_pw['dout'], conv_dw['cache'])

        ##########################################

        cache.update({'dconv_dw':dconv_dw, 'dconv_pw':dconv_pw, 'dlel':dlel})
        return {'dout':dconv_dw['dout'], 'cache':cache}
        
    def gv(self, AI, AO, DO, cache):

        conv_dw,  conv_pw, lel  = cache['conv_dw'],  cache['conv_pw'], cache['lel']
        dconv_dw, dconv_pw, dlel = cache['dconv_dw'], cache['dconv_pw'], cache['dlel']
        
        ##########################################

        dconv_dw = self.conv_dw.gv(AI,              conv_dw['aout'], dconv_pw['dout'], dconv_dw['cache'])
        dconv_pw = self.conv_pw.gv(conv_dw['aout'], conv_pw['aout'], lel['dout'],      dconv_dw['cache'])
        dlel     = self.lel.gv(    conv_pw['aout'], lel['aout'],     DO,               dlel['cache'])
        
        ##########################################
        
        grads = []

        grads.extend(dconv_dw)
        grads.extend(dconv_pw)
        grads.extend(dlel)

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
    
    
    
    
