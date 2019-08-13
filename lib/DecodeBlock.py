
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock
from lib.ConvDWBlock import ConvDWBlock
from lib.UpSample import UpSample

class DecodeBlock(Layer):

    def __init__(self, input_shape, filter_shape, ksize, init, name, load=None, train=True):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        
        self.ksize = ksize

        self.init = init
        self.name = name
        self.load = load
        self.train_flag = train

        input_shape_1 = [self.batch, self.h, self.w, self.fin]
        input_shape_2 = [self.batch, self.h, self.w, self.fout]
        input_shape_3 = input_shape_2

        self.conv_dw = ConvDWBlock(input_shape=input_shape_1, 
                                   filter_shape=[3, 3, self.fin, 1], 
                                   strides=[1,1,1,1], 
                                   init=self.init, 
                                   name=self.name + '_conv_block_dw',
                                   load=self.load, 
                                   train=self.train_flag)

        self.conv_pw = ConvBlock(input_shape=input_shape_2, 
                                 filter_shape=[1, 1, self.fin, self.fout], 
                                 strides=[1,1,1,1], 
                                 init=self.init, 
                                 name=self.name + '_conv_block_pw',
                                 load=self.load, 
                                 train=self.train_flag)

        self.upsample = UpSample(input_shape=input_shape_3, ksize=self.ksize)

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.conv_dw.get_weights())
        weights.extend(self.conv_pw.get_weights())
        return weights

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return self.conv_dw.num_params() + self.conv_pw.num_params()

    ###################################################################

    def forward(self, X):
        conv_dw = self.conv_dw.forward(X)
        conv_pw = self.conv_pw.forward(conv_dw['aout'])
        up      = self.upsample.forward(conv_pw['aout'])
        cache = {'conv_dw':conv_dw, 'conv_pw':conv_pw, 'up': up}
        return {'aout':up['aout'], 'cache':cache}
        
    def bp(self, AI, AO, DO, cache):    
        up, conv_dw, conv_pw = cache['up'], cache['conv_dw'], cache['conv_pw']
        dup,      gup      = self.up.bp(     conv_pw['aout'], up['aout'],      DO,               up['cache'])
        dconv_pw, gconv_pw = self.conv_pw.bp(conv_dw['aout'], conv_pw['aout'], dup['dout'],      conv_pw['cache'])
        dconv_dw, gconv_dw = self.conv_dw.bp(AI,              conv_dw['aout'], dconv_pw['dout'], conv_dw['cache'])
        cache.update({'dconv_dw':dconv_dw, 'dconv_pw':dconv_pw, 'dup':dup})
        grads = []
        grads.extend(gconv_dw)
        grads.extend(gconv_pw)
        return {'dout':dconv_dw['dout'], 'cache':cache}, grads

    def dfa(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
