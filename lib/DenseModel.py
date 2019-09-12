
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.DenseBlock import DenseBlock
from lib.DenseTransition import DenseTransition

class DenseModel(Layer):

    def __init__(self, input_shape, init, name, k, L):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.init = init
        self.name = name
        self.k = k
        self.L = L

        self.blocks = []
        for ii in range(len(self.L)):
            dense_fmaps = self.fin + sum(L[0:ii]) * k
            dense = DenseBlock(input_shape=[self.batch, self.h // 2 ** ii, self.w // 2 ** ii, dense_fmaps], init=self.init, name=self.name + ('_block_%d' % ii), k=self.k, L=self.L[ii])
            self.blocks.append(dense)

            trans_fmaps = self.fin + sum(L[0:ii+1]) * k
            trans = DenseTransition(input_shape=[self.batch, self.h // 2 ** ii, self.w // 2 ** ii, trans_fmaps], init=self.init, name=self.name + ('_block_%d' % ii))
            self.blocks.append(trans)

        self.num_blocks = len(self.blocks)

    ###################################################################

    def get_weights(self):
        assert(False)

    def output_shape(self):
        assert(False)

    def num_params(self):
        assert(False)

    def forward(self, X):
        A     = [None] * self.num_blocks
        cache = [None] * self.num_blocks

        for ii in range(self.num_blocks):
            block = self.blocks[ii]
            if ii == 0:
                A[ii], cache[ii] = block.forward(X)
            else:
                A[ii], cache[ii] = block.forward(A[ii-1])

        return A[self.num_blocks-1], (A, cache)
        
    ###################################################################
        
    def bp(self, AI, AO, DO, cache):
        # DI = tf.ones_like(AI)
        # DI = tf.Print(DI, [tf.shape(DO)], message="", summarize=1000)
        # return DI, []

        A, C = cache
        D = [None] * self.num_blocks
        GV = []

        for ii in range(self.num_blocks-1, -1, -1):
            block = self.blocks[ii]

            if (ii == self.num_blocks-1):
                D[ii], gv = block.bp(A[ii-1], A[ii], DO,      C[ii])
                GV.extend(gv)
            elif (ii == 0):
                D[ii], gv = block.bp(AI,      A[ii], D[ii+1], C[ii])
                GV.extend(gv)
            else:
                D[ii], gv = block.bp(A[ii-1], A[ii], D[ii+1], C[ii])
                GV.extend(gv)

        return D[0], GV

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
        
    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
