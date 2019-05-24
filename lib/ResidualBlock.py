
import tensorflow as tf
import numpy as np

from lib.Layer import Layer 
from lib.ConvBlock import ConvBlock

class ResidualBlock(Layer):

    def __init__(self, input_shape, filter_shape, init, name):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        
        self.filter_shape = filter_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_shape
        
        self.output_shape = [self.batch, self.h, self.w, self.fout]
        
        self.init = init
        self.name = name
        
        self.block1 = ConvBlock(input_shape=self.input_shape, filter_shape=self.filter_shape, init=self.init, name=self.name + '_conv_block_1')
        self.block2 = ConvBlock(input_shape=self.input_shape, filter_shape=self.filter_shape, init=self.init, name=self.name + '_conv_block_2')
        self.block3 = ConvBlock(input_shape=self.input_shape, filter_shape=self.filter_shape, init=self.init, name=self.name + '_conv_block_3')

        self.nonsense = False

    ###################################################################

    def get_weights(self):
        return []

    def output_shape(self):
        return self.output_shape

    def num_params(self):
        return 0

    ###################################################################

    def forward(self, X):
        in1 = X
        block1 = self.block1.forward(in1)
        
        in2 = in1 + block1['aout']
        block2 = self.block2.forward(in2)
        
        in3 = in2 + block2['aout']
        block3 = self.block3.forward(in3)
        
        aout = in3 + block3['aout']
        cache = {'block1':block1, 'block2':block2, 'block3':block3}
        
        return {'aout':aout, 'cache':cache}
        
    def backward(self, AI, AO, DO, cache):    
        block1, block2, block3 = cache['block1'], cache['block2'], cache['block3']
        
        in1 = AI
        in2 = in1 + block1['aout']
        in3 = in2 + block2['aout']

        if self.nonsense:
            dblock3 = self.block3.backward(in3, block3['aout'], DO, block3['cache'])
            dblock2 = self.block2.backward(in2, block2['aout'], DO, block2['cache'])
            dblock1 = self.block1.backward(in1, block1['aout'], DO, block1['cache'])
        else:
            dblock3 = self.block3.backward(in3, block3['aout'], DO,              block3['cache'])
            dblock2 = self.block2.backward(in2, block2['aout'], dblock3['dout'], block2['cache'])
            dblock1 = self.block1.backward(in1, block1['aout'], dblock2['dout'], block1['cache'])

        cache.update({'dblock1':dblock1, 'dblock2':dblock2, 'dblock3':dblock3})
        return {'dout':dblock1['dout'], 'cache':cache}
        
    def gv(self, AI, AO, DO, cache):
        block1, block2, block3 = cache['block1'], cache['block2'], cache['block3']
        dblock1, dblock2, dblock3 = cache['dblock1'], cache['dblock2'], cache['dblock3']

        in1 = AI
        in2 = in1 + block1['aout']
        in3 = in2 + block2['aout']

        if self.nonsense:
            dblock1 = self.block1.gv(in1, block1['aout'], DO, dblock1['cache'])
            dblock2 = self.block2.gv(in2, block2['aout'], DO, dblock2['cache'])
            dblock3 = self.block3.gv(in3, block3['aout'], DO, dblock3['cache'])
        else:
            dblock1 = self.block1.gv(in1, block1['aout'], dblock2['dout'], dblock1['cache'])
            dblock2 = self.block2.gv(in2, block2['aout'], dblock3['dout'], dblock2['cache'])
            dblock3 = self.block3.gv(in3, block3['aout'], DO,              dblock3['cache'])

        grads = []
        grads.extend(dblock1)
        grads.extend(dblock2)
        grads.extend(dblock3)

        return grads
        
    def train(self, AI, AO, DO): 
        assert(False)
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        assert(False)
        
    def dfa_gv(self, AI, AO, E, DO):
        assert(False)
        
    def dfa(self, AI, AO, E, DO): 
        assert(False)
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y):
        assert(False)

    def lel_gv(self, AI, AO, E, DO, Y):
        assert(False)
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################   
    
    
    
    
    
    
    
