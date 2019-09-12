
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.DenseConv import DenseConv

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
            dense = DenseConv(input_shape=[self.batch, self.h, self.w, c], init=self.init, name=self.name + ('_dense_conv_%d' % ii), k=self.k)
            self.layers.append(dense)
        
        self.num_layers = len(self.layers)

    ###################################################################

    def get_weights(self):
        assert(False)

    def output_shape(self):
        assert(False)

    def num_params(self):
        assert(False)

    def forward(self, X):
        AI    = [None] * self.num_layers
        AO    = [None] * self.num_layers
        cache = [None] * self.num_layers

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                accum = X
                AO[ii], cache[ii] = l.forward(accum)
            else:
                accum = tf.concat((accum, AO[ii-1]), axis=3)
                AO[ii], cache[ii] = l.forward(accum)

            AI[ii] = accum

        accum = tf.concat((accum, AO[-1]), axis=3)
        return accum, (AI, AO, cache)
        
    ###################################################################
        
    def bp(self, AI, AO, DO, cache):
        AI, AO, C = cache
        DI = [None] * self.num_layers
        GV = []

        for ii in range(self.num_layers-1, -1, -1):
            for jj in range(ii):
                if (jj == 0):
                    s = 0
                    e = self.fin
                else:
                    s = self.fin + (jj - 1) * self.k
                    e = self.fin + jj       * self.k

                if (ii == 0):
                    DI[jj] = DO[s:e]
                else:
                    DI[jj] = DI[jj] + DO[s:e]

            l = self.layers[ii]
            DO, gvs = l.bp(AI[ii], AO[ii], DI[ii], C[ii])
            grads_and_vars.extend(gvs)

        return D[0], GV

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
        
    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
