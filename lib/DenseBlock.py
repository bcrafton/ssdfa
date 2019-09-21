
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.DenseConv import DenseConv

class DenseBlock(Layer):

    def __init__(self, input_shape, init, name, k, L, fb, fb_pw):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.init = init
        self.name = name
        self.k = k
        self.L = L
        self.fb = fb
        self.fb_pw = fb_pw

        self.layers = []
        for ii in range(self.L):
            c = self.fin if ii == 0 else self.fin + ii * k
            dense = DenseConv(input_shape=[self.batch, self.h, self.w, c], init=self.init, name=self.name + ('_dense_conv_%d' % ii), k=self.k, fb=self.fb, fb_pw=self.fb_pw)
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
        D = [None] * self.num_layers
        DI = None
        GV = []

        for ii in range(self.num_layers-1, -1, -1):
            DI = DO[:, :, :, 0:self.fin] if DI == None else (DI + DO[:, :, :, 0:self.fin])

            for jj in range(ii + 1):
                s = self.fin +  jj    * self.k
                e = self.fin + (jj+1) * self.k
                D[jj] = DO[:, :, :, s:e] if D[jj] == None else (D[jj] + DO[:, :, :, s:e])

            l = self.layers[ii]
            DO, gv = l.bp(AI[ii], AO[ii], D[ii], C[ii])
            GV = gv + GV

        DI = DI + DO
        return DI, GV

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
        
    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
