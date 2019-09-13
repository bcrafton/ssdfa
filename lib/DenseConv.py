
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock

class DenseConv(Layer):

    def __init__(self, input_shape, init, name, k, fb, fb_pw):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.init = init
        self.name = name
        self.k = k
        self.fb = fb
        self.fb_pw = fb_pw

        self.conv1x1 = ConvBlock(
                       input_shape=[self.batch, self.h, self.w, self.fin], 
                       filter_shape=[3, 3, self.fin, self.k * 4], 
                       strides=[1,1,1,1], 
                       init=self.init, 
                       name=self.name + '_conv1x1_block',
                       fb=self.fb_pw)

        self.conv3x3 = ConvBlock(
                       input_shape=[self.batch, self.h, self.w, self.k * 4], 
                       filter_shape=[3, 3, self.k * 4, self.k], 
                       strides=[1,1,1,1], 
                       init=self.init, 
                       name=self.name + '_conv3x3_block',
                       fb=self.fb)

    ###################################################################

    def get_weights(self):
        weights = []
        weights.extend(self.conv1x1.get_weights())
        weights.extend(self.conv3x3.get_weights())
        return weights

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return self.conv1x1.num_params() + self.conv3x3.num_params()

    def forward(self, X):
        conv1x1, conv1x1_cache = self.conv1x1.forward(X)
        conv3x3, conv3x3_cache = self.conv3x3.forward(conv1x1)
        cache = (conv1x1, conv1x1_cache, conv3x3, conv3x3_cache)
        return conv3x3, cache
        
    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        conv1x1, conv1x1_cache, conv3x3, conv3x3_cache = cache
        dconv3x3, gconv3x3 = self.conv3x3.bp(conv1x1, AO,      DO,       conv3x3_cache)
        dconv1x1, gconv1x1 = self.conv1x1.bp(AI,      conv1x1, dconv3x3, conv1x1_cache)
        grads = []
        grads.extend(gconv1x1)
        grads.extend(gconv3x3)
        return dconv1x1, grads

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
