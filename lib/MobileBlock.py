
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock
from lib.ConvDWBlock import ConvDWBlock

class MobileBlock(Layer):

    def __init__(self, input_shape, filter_shape, strides, init, name, load=None, train=True, fb_dw='f', fb_pw='f'):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides

        self.init = init
        self.name = name
        self.load = load
        self.train_flag = train
        self.fb_pw = fb_pw
        self.fb_dw = fb_dw

        input_shape_1 = [self.batch, self.h,            self.w,            self.fin]
        input_shape_2 = [self.batch, self.h // self.sh, self.w // self.sw, self.fin]
        
        self.conv_dw = ConvDWBlock(input_shape=input_shape_1, 
                                   filter_shape=[3, 3, self.fin, 1], 
                                   strides=self.strides, 
                                   init=self.init, 
                                   name=self.name + '_conv_block_dw',
                                   load=self.load, 
                                   train=self.train_flag,
                                   fb=self.fb_dw)

        self.conv_pw = ConvBlock(input_shape=input_shape_2, 
                                 filter_shape=[1, 1, self.fin, self.fout], 
                                 strides=[1,1,1,1], 
                                 init=self.init, 
                                 name=self.name + '_conv_block_pw',
                                 load=self.load, 
                                 train=self.train_flag,
                                 fb=self.fb_pw)

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
        conv_dw, conv_dw_cache = self.conv_dw.forward(X)
        conv_pw, conv_pw_cache = self.conv_pw.forward(conv_dw)
        cache = (conv_dw, conv_dw_cache, conv_pw, conv_pw_cache)
        return conv_pw, cache
        
    def bp(self, AI, AO, DO, cache):    
        conv_dw, conv_dw_cache, conv_pw, conv_pw_cache = cache
        dconv_pw, gconv_pw = self.conv_pw.bp(conv_dw, conv_pw, DO,       conv_pw_cache)
        dconv_dw, gconv_dw = self.conv_dw.bp(AI,      conv_dw, dconv_pw, conv_dw_cache)
        grads = []
        grads.extend(gconv_dw)
        grads.extend(gconv_pw)
        return dconv_dw, grads

    def ss(self, AI, AO, DO, cache):    
        conv_dw, conv_dw_cache, conv_pw, conv_pw_cache = cache
        dconv_pw, gconv_pw = self.conv_pw.ss(conv_dw, conv_pw, DO,       conv_pw_cache)
        dconv_dw, gconv_dw = self.conv_dw.ss(AI,      conv_dw, dconv_pw, conv_dw_cache)
        grads = []
        grads.extend(gconv_dw)
        grads.extend(gconv_pw)
        return dconv_dw, grads

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
