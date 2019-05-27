
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
        input_shape_3 = [self.batch, self.h // self.sh, self.w // self.sw, self.fout]
        
        self.conv_dw = ConvolutionDW(input_sizes=input_shape_1, filter_sizes=[3, 3, self.fin, 1], init=self.init, strides=self.strides, padding="SAME", name=self.name + '_conv_dw')
        self.bn_dw   = BatchNorm(input_size=input_shape_2, name=self.name + '_dw_bn')
        self.relu_dw = Relu()
        
        self.conv_pw = Convolution(input_sizes=input_shape_2, filter_sizes=[1, 1, self.fin, self.fout], init=self.init, strides=[1,1,1,1], padding="SAME", name=self.name + '_conv_pw')
        self.bn_pw   = BatchNorm(input_size=input_shape_3, name=self.name + '_pw_bn')
        self.relu_pw = Relu()
        
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
        bn_dw   = self.bn_dw.forward(conv_dw['aout'])
        relu_dw = self.relu_dw.forward(bn_dw['aout'])

        conv_pw = self.conv_pw.forward(relu_dw['aout'])
        bn_pw   = self.bn_pw.forward(conv_pw['aout'])
        relu_pw = self.relu_pw.forward(bn_pw['aout'])

        cache = {'conv_dw':conv_dw['aout'], 'bn_dw':bn_dw['aout'], 'relu_dw':relu_dw['aout'],
                 'conv_pw':conv_pw['aout'], 'bn_pw':bn_pw['aout'], 'relu_pw':relu_pw['aout']}
                 
        return {'aout':relu_pw['aout'], 'cache':cache}
        
    def backward(self, AI, AO, DO, cache):    
        
        conv_dw, bn_dw, relu_dw = cache['conv_dw'], cache['bn_dw'], cache['relu_dw']
        conv_pw, bn_pw, relu_pw = cache['conv_pw'], cache['bn_pw'], cache['relu_pw']
        
        ##########################################3
        
        drelu_pw = self.relu_pw.backward(bn_pw, relu_pw, DO)
        dbn_pw   = self.bn_pw.backward(conv_pw, bn_pw, drelu_pw['dout'])
        dconv_pw = self.conv_pw.backward(relu_dw, conv_pw, dbn_pw['dout'])

        drelu_dw = self.relu_dw.backward(bn_dw, relu_dw, dconv_pw['dout'])
        dbn_dw   = self.bn_dw.backward(conv_dw, bn_dw, drelu_dw['dout'])
        dconv_dw = self.conv_dw.backward(AI, conv_dw, dbn_dw['dout'])

        ##########################################

        cache.update({'dconv_dw':dconv_dw['dout'], 'dbn_dw':dbn_dw['dout'], 'drelu_dw':drelu_dw['dout']})
        cache.update({'dconv_pw':dconv_pw['dout'], 'dbn_pw':dbn_pw['dout'], 'drelu_pw':drelu_pw['dout']})
        
        return {'dout':dconv_dw['dout'], 'cache':cache}
        
    def gv(self, AI, AO, DO, cache):

        conv_dw, bn_dw, relu_dw = cache['conv_dw'], cache['bn_dw'], cache['relu_dw']
        conv_pw, bn_pw, relu_pw = cache['conv_pw'], cache['bn_pw'], cache['relu_pw']
        
        dconv_dw, dbn_dw, drelu_dw = cache['dconv_dw'], cache['dbn_dw'], cache['drelu_dw']
        dconv_pw, dbn_pw, drelu_pw = cache['dconv_pw'], cache['dbn_pw'], cache['drelu_pw']

        ##########################################

        dconv_dw = self.conv_dw.gv(AI, conv_dw, dbn_dw)
        dbn_dw   = self.bn_dw.gv(conv_dw, bn_dw, drelu_dw)

        dconv_pw = self.conv_pw.gv(relu_dw, conv_pw, dbn_pw)
        dbn_pw   = self.bn_pw.gv(conv_pw, bn_pw, drelu_pw)

        ##########################################
        
        grads = []

        grads.extend(dconv_dw)
        grads.extend(dbn_dw)

        grads.extend(dconv_pw)
        grads.extend(dbn_pw)

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
    
    
    
    
