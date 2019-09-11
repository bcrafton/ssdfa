
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock

class DenseBlock(Layer):

    def __init__(self, input_shape, init, name, k):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.init = init
        self.name = name
        self.k = k

        self.conv1x1 = ConvBlock(
                       input_shape=self.input_shape, 
                       filter_shape=[3, 3, self.fin, self.k * 4], 
                       strides=[1,1,1,1], 
                       init=self.init, 
                       name=self.name + ('_conv1x1_block_%d' % l))

        self.conv3x3 = ConvBlock(
                       input_shape=self.input_shape, 
                       filter_shape=[3, 3, self.k * 4, self.k], 
                       strides=[1,1,1,1], 
                       init=self.init, 
                       name=self.name + ('_conv3x3_block_%d' % l))

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
        conv1x1 = self.conv1x1.forward(X)
        conv3x3 = self.conv3x3.forward(conv1x1['aout'])
        cache = {'conv1x1':conv1x1, 'conv3x3':conv3x3}
        return {'aout':conv3x3['aout'], 'cache':cache}
        
    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        conv1x1, conv3x3 = cache['conv1x1'], cache['conv3x3']
        dconv3x3, gconv3x3 = self.conv3x3.bp(conv1x1['aout'], conv3x3['aout'], DO,       conv3x3['cache'])
        dconv1x1, gconv1x1 = self.conv1x1.bp(AI,              conv1x1['aout'], dconv3x3, conv1x1['cache'])
        grads = []
        grads.extend(gconv1x1)
        grads.extend(gconv3x3)
        return dconv1x1, grads

    def ss(self, AI, AO, DO, cache):    
        conv1x1, conv3x3 = cache['conv1x1'], cache['conv3x3']
        dconv3x3, gconv3x3 = self.conv3x3.ss(conv1x1['aout'], conv3x3['aout'], DO,       conv3x3['cache'])
        dconv1x1, gconv1x1 = self.conv1x1.ss(AI,              conv1x1['aout'], dconv3x3, conv1x1['cache'])
        grads = []
        grads.extend(gconv1x1)
        grads.extend(gconv3x3)
        return dconv1x1, grads

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
