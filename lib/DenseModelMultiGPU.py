
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
            with tf.device('/device:GPU:%d' % (ii + 1)):
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

        with tf.device('/device:GPU:1'):
            A[0], cache[0] = self.blocks[0].forward(X)
            A[1], cache[1] = self.blocks[1].forward(A[0])

        with tf.device('/device:GPU:2'):
            A[2], cache[2] = self.blocks[2].forward(A[1])
            A[3], cache[3] = self.blocks[3].forward(A[2])

        with tf.device('/device:GPU:3'):
            A[4], cache[4] = self.blocks[4].forward(A[3])
            A[5], cache[5] = self.blocks[5].forward(A[4])

        with tf.device('/device:GPU:4'):
            A[6], cache[6] = self.blocks[6].forward(A[5])
            A[7], cache[7] = self.blocks[7].forward(A[6])

        return A[self.num_blocks-1], (A, cache)
        
    ###################################################################
        
    def bp(self, AI, AO, DO, cache):
        # DI = tf.ones_like(AI)
        # DI = tf.Print(DI, [tf.shape(DO)], message="", summarize=1000)
        # return DI, []

        A, C = cache
        D = [None] * self.num_blocks
        GV = []

        with tf.device('/device:GPU:1'):
            D[7], gv7 = self.blocks[7].bp(A[6], A[7], DO,      C[7])
            D[6], gv6 = self.blocks[6].bp(A[5], A[6], D[7],    C[6])

        with tf.device('/device:GPU:2'):
            D[5], gv5 = self.blocks[5].bp(A[4], A[5], D[6],    C[5])
            D[4], gv4 = self.blocks[4].bp(A[3], A[4], D[5],    C[4])

        with tf.device('/device:GPU:3'):
            D[3], gv3 = self.blocks[3].bp(A[2], A[3], D[4],    C[3])
            D[2], gv2 = self.blocks[2].bp(A[1], A[2], D[3],    C[2])

        with tf.device('/device:GPU:4'):
            D[1], gv1 = self.blocks[1].bp(A[0], A[1], D[2],    C[1])
            D[0], gv0 = self.blocks[0].bp(AI,   A[0], D[1],    C[0])

        GV.extend(gv7)
        GV.extend(gv6)
        GV.extend(gv5)
        GV.extend(gv4)
        GV.extend(gv3)
        GV.extend(gv2)
        GV.extend(gv1)
        GV.extend(gv0)

        return D[0], GV

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
        
    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
