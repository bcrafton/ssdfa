
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

        ###################################################################

        '''
        self.blocks = []
        for ii in range(len(self.L)):
            dense_fmaps = self.fin + sum(L[0:ii]) * k
            dense = DenseBlock(input_shape=[self.batch, self.h // 2 ** ii, self.w // 2 ** ii, dense_fmaps], init=self.init, name=self.name + ('_block_%d' % ii), k=self.k, L=self.L[ii])
            self.blocks.append(dense)

            trans_fmaps = self.fin + sum(L[0:ii+1]) * k
            trans = DenseTransition(input_shape=[self.batch, self.h // 2 ** ii, self.w // 2 ** ii, trans_fmaps], init=self.init, name=self.name + ('_block_%d' % ii))
            self.blocks.append(trans)

        self.num_blocks = len(self.blocks)
        '''

        #####################################

        if len(L) == 4:
            self.blocks = []

            dense_fmaps = self.fin 
            dense1 = DenseBlock(input_shape=[self.batch, self.h, self.w, dense_fmaps], init=self.init, name=self.name + '_block_1', k=self.k, L=self.L[0])
            trans_fmaps = self.fin + L[0] * k
            trans1 = DenseTransition(input_shape=[self.batch, self.h, self.w, trans_fmaps], init=self.init, name=self.name + '_trans_1')

            dense_fmaps = self.fin + L[0] * k
            dense2 = DenseBlock(input_shape=[self.batch, self.h // 2, self.w // 2, dense_fmaps], init=self.init, name=self.name + '_block_2', k=self.k, L=self.L[1])
            trans_fmaps = self.fin + (L[0]+L[1]) * k
            trans2 = DenseTransition(input_shape=[self.batch, self.h // 2, self.w // 2, trans_fmaps], init=self.init, name=self.name + '_trans_2')

            dense_fmaps = self.fin + (L[0]+L[1]) * k
            dense3 = DenseBlock(input_shape=[self.batch, self.h // 4, self.w // 4, dense_fmaps], init=self.init, name=self.name + '_block_3', k=self.k, L=self.L[2])
            trans_fmaps = self.fin + (L[0]+L[1]+L[2]) * k
            trans3 = DenseTransition(input_shape=[self.batch, self.h // 4, self.w // 4, trans_fmaps], init=self.init, name=self.name + '_trans_3')

            dense_fmaps = self.fin + (L[0]+L[1]+L[2]) * k
            dense4 = DenseBlock(input_shape=[self.batch, self.h // 8, self.w // 8, dense_fmaps], init=self.init, name=self.name + '_block_4', k=self.k, L=self.L[3])
            
            self.blocks.append(dense1)
            self.blocks.append(trans1)
            self.blocks.append(dense2)
            self.blocks.append(trans2)
            self.blocks.append(dense3)
            self.blocks.append(trans3)
            self.blocks.append(dense4)
            self.num_blocks = len(self.blocks)

        if len(L) == 5:
            self.blocks = []

            dense_fmaps = self.fin 
            dense1 = DenseBlock(input_shape=[self.batch, self.h, self.w, dense_fmaps], init=self.init, name=self.name + '_block_1', k=self.k, L=self.L[0])
            trans_fmaps = self.fin + L[0] * k
            trans1 = DenseTransition(input_shape=[self.batch, self.h, self.w, trans_fmaps], init=self.init, name=self.name + '_trans_1')

            dense_fmaps = self.fin + L[0] * k
            dense2 = DenseBlock(input_shape=[self.batch, self.h // 2, self.w // 2, dense_fmaps], init=self.init, name=self.name + '_block_2', k=self.k, L=self.L[1])
            trans_fmaps = self.fin + (L[0]+L[1]) * k
            trans2 = DenseTransition(input_shape=[self.batch, self.h // 2, self.w // 2, trans_fmaps], init=self.init, name=self.name + '_trans_2')

            dense_fmaps = self.fin + (L[0]+L[1]) * k
            dense3 = DenseBlock(input_shape=[self.batch, self.h // 4, self.w // 4, dense_fmaps], init=self.init, name=self.name + '_block_3', k=self.k, L=self.L[2])
            trans_fmaps = self.fin + (L[0]+L[1]+L[2]) * k
            trans3 = DenseTransition(input_shape=[self.batch, self.h // 4, self.w // 4, trans_fmaps], init=self.init, name=self.name + '_trans_3')

            dense_fmaps = self.fin + (L[0]+L[1]+L[2]) * k
            dense4 = DenseBlock(input_shape=[self.batch, self.h // 8, self.w // 8, dense_fmaps], init=self.init, name=self.name + '_block_4', k=self.k, L=self.L[3])
            trans_fmaps = self.fin + (L[0]+L[1]+L[2]+L[3]) * k
            trans4 = DenseTransition(input_shape=[self.batch, self.h // 8, self.w // 8, trans_fmaps], init=self.init, name=self.name + '_trans_4')

            dense_fmaps = self.fin + + (L[0]+L[1]+L[2]+L[3]) * k
            dense5 = DenseBlock(input_shape=[self.batch, self.h // 16, self.w // 16, dense_fmaps], init=self.init, name=self.name + '_block_5', k=self.k, L=self.L[4])
            
            self.blocks.append(dense1)
            self.blocks.append(trans1)
            self.blocks.append(dense2)
            self.blocks.append(trans2)
            self.blocks.append(dense3)
            self.blocks.append(trans3)
            self.blocks.append(dense4)
            self.blocks.append(trans4)
            self.blocks.append(dense5)
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
                GV = gv + GV
            elif (ii == 0):
                D[ii], gv = block.bp(AI,      A[ii], D[ii+1], C[ii])
                GV = gv + GV
            else:
                D[ii], gv = block.bp(A[ii-1], A[ii], D[ii+1], C[ii])
                GV = gv + GV

        return D[0], GV

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
        
    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
    
    
    
    
