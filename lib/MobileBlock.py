
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock

from lib.ConvolutionDW import ConvolutionDW
from lib.Convolution import Convolution
from lib.BatchNorm import BatchNorm
from lib.Activation import Relu

class MobileBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, init, name):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        
        self.init = init
        self.name = name
        
        input_shape_1 = [self.batch, self.h,            self.w,            self.fin]
        input_shape_2 = [self.batch, self.h // self.sh, self.w // self.sw, self.fin]
        
        conv_dw = ConvDWBlock(input_shape=input_shape_1, filter_shape=[3, 3, self.fin, 1], strides=self.strides, init=args.init, name='_conv_block_dw')
        conv_pw = ConvBlock(input_shape=input_shape_2, filter_shape=[1, 1, self.fin, self.fout], strides=[1,1,1,1], init=args.init, name='_conv_block_pw')
        
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
        conv_pw = self.conv_pw.forward(relu_dw['aout'])

        cache = {'conv_dw':conv_dw, 'conv_pw':conv_pw['aout']}
        return {'aout':conv_pw['aout'], 'cache':cache}
        
    def backward(self, AI, AO, DO, cache):    
        
        conv_dw, conv_pw = cache['conv_dw'], cache['conv_pw']
        
        ##########################################3
        
        dconv_pw = self.conv_pw.backward(conv_dw['aout'], conv_pw['aout'], DO)
        dconv_dw = self.conv_dw.backward(AI,              conv_dw['aout'], dconv_pw['dout'])

        ##########################################

        cache.update({'dconv_dw':dconv_dw['dout'], 'dconv_pw':dconv_pw['dout']})
        return {'dout':dconv_dw['dout'], 'cache':cache}
        
    def gv(self, AI, AO, DO, cache):

        conv_dw,  conv_pw  = cache['conv_dw'],  cache['conv_pw']
        dconv_dw, dconv_pw = cache['dconv_dw'], cache['dconv_pw']
        
        ##########################################

        dconv_dw = self.conv_dw.gv(AI,              conv_dw['aout'], dconv_pw['dout'])
        dconv_pw = self.conv_pw.gv(conv_dw['aout'], conv_pw['aout'], DO)

        ##########################################
        
        grads = []

        grads.extend(dconv_dw)
        grads.extend(dconv_pw)

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
    
    
    
    
