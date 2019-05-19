
import tensorflow as tf
import numpy as np

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.Dropout import Dropout
from lib.LELConv import LELConv
from lib.BatchNorm import BatchNorm
from lib.Activation import Activation
from lib.Activation import Relu

class Block(Layer):

    def __init__(self, input_shape, filter_shape, pool_shape, num_classes, init, name):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.filter_shape = filter_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_shape
        self.output_shape = [self.batch, self.h, self.w, self.fout]
        self.pool_shape = pool_shape
        self.num_classes = num_classes
        self.init = init
        self.name = name

        l0 = Convolution(input_sizes=self.input_shape, filter_sizes=self.filter_shape, init=self.init, strides=[1,1,1,1], padding="SAME", name=self.name + '_conv')
        l1 = BatchNorm(input_size=self.output_shape, name=self.name + '_bn')
        l2 = Relu()
        l3 = LELConv(input_shape=self.output_shape, pool_shape=self.pool_shape, num_classes=self.num_classes, name=self.name + '_fb')

        self.block = Model(layers=[l0, l1, l2, l3])

    ###################################################################

    def get_weights(self):
        # shud just always return dictionaries and do ".update()" no lists. 
        return self.block.get_weights()

    def output_shape(self):
        assert(False)

    def num_params(self):
        return self.block.num_params()

    def forward(self, X):
        return self.block.forward(X)
        
        '''
        conv = l0.forward(X)
        bn = l1.forward(conv)
        relu = l2.forward(bn)
        lel = l3.forward(relu)
        return lel
        '''
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return self.block.backward(AI, AO, DO)
        
    def gv(self, AI, AO, DO):    
        return self.block.gv(AI, AO, DO)
        
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
