
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.DenseBlock import DenseBlock

class DenseModel(Layer):

    def __init__(self, input_shape, init, name, k, Ls):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.filter_shape = filter_shape
        self.fin, self.fout = self.filter_shape
        self.init = init
        self.name = name
        self.k = k
        self.L = L

        self.blocks = []
        for ii in range(len(self.L)):
            dense = DenseBlock(input_shape=self.input_shape, init=self.init, name=self.name + ('_block_%d' % ii), k=self.k, l=self.L[ii])
            self.blocks.append(dense)

    ###################################################################

    def get_weights(self):
        assert(False)

    def output_shape(self):
        assert(False)

    def num_params(self):
        assert(False)

    def forward(self, X):
        A = [None] * self.num_layers

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                accum = X
                A[ii] = l.forward(accum)
            else:
                accum = tf.concat((accum, A[ii-1]['aout']), axis=3)
                A[ii] = l.forward(accum)

        return A[self.num_layers-1]['aout'], A
        
    ###################################################################
        
    def bp(self, AI, AO, DO, cache):    
        return DO, []

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
        
    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
