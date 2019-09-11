
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.DenseConv import DenseConv
from lib.AvgPool import AvgPool

class DenseBlock(Layer):

    def __init__(self, input_shape, init, name, k, L):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.init = init
        self.name = name
        self.k = k
        self.L = L

        self.layers = []
        for ii in range(self.L):
            c = self.fin if ii == 0 else self.fin + ii * k
            dense = DenseConv(input_shape=[self.batch, self.h, self.w, c], init=self.init, name=self.name + ('_dense_block_%d' % ii), k=self.k)
            self.layers.append(dense)
        
        self.pool = AvgPool(size=[self.batch, self.h, self.w, self.fin + L * k], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        self.num_layers = len(self.layers)

    ###################################################################

    def get_weights(self):
        assert(False)

    def output_shape(self):
        assert(False)

    def num_params(self):
        assert(False)

    def forward(self, X):
        A =     [None] * self.num_layers
        cache = [None] * self.num_layers

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                accum = X
                A[ii], cache[ii] = l.forward(accum)
            else:
                accum = tf.concat((accum, A[ii-1]), axis=3)
                A[ii], cache[ii] = l.forward(accum)

        concat = tf.concat(A, axis=3)
        return concat, (A, cache)
        
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
    
    
    
    
