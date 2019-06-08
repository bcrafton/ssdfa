
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.FeedbackMatrix import FeedbackMatrix

from lib.Model import Model
from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.Activation import Relu
from lib.Activation import Linear

class LELConv(Layer):

    def __init__(self, input_shape, pool_shape, num_classes, name=None):
        self.input_shape = input_shape
        self.batch_size, self.h, self.w, self.fin = self.input_shape
        self.pool_shape = pool_shape
        self.num_classes = num_classes
        self.name = name

        l1 = AvgPool(size=self.input_shape, ksize=self.pool_shape, strides=self.pool_shape, padding='SAME')

        l2_input_shape = l1.output_shape()
        l2 = ConvToFullyConnected(input_shape=l2_input_shape)
        
        l3_input_shape = l2.output_shape()
        l3 = FullyConnected(input_shape=l3_input_shape, size=self.num_classes, init='alexnet', activation=Linear(), bias=0., name=self.name)
        
        self.B = Model(layers=[l1, l2, l3])
        
    ###################################################################
    
    def get_weights(self):
        return []

    def get_feedback(self):
        assert(False)
        
    def output_shape(self):
        return self.input_shape

    def num_params(self):
        return 0
        
    def forward(self, X):
        return {'aout':X, 'cache':{}}
                
    ###################################################################           
        
    def backward(self, AI, AO, DO, cache=None):    
        return {'dout':DO, 'cache':{}}

    def gv(self, AI, AO, DO, cache=None):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return DO
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
        
    def lel_backward(self, AI, AO, E, DO, Y, cache):
        DI = self.B.backwards(AI, Y)
        return {'dout':DI, 'cache':{}}
        
    def lel_gv(self, AI, AO, E, DO, Y, cache):
        gvs = self.B.gvs(AI, Y)
        return gvs

    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
        
        


