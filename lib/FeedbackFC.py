
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.FeedbackMatrix import FeedbackMatrix

np.set_printoptions(threshold=np.inf)

class FeedbackFC(Layer):
    num = 0
    def __init__(self, size, num_classes, sparse, rank, name=None):
        self.size = size
        self.input_size, self.output_size = self.size
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.name = name

        b = FeedbackMatrix(size=(self.num_classes, self.output_size), sparse=self.sparse, rank=self.rank)
        self.B = tf.cast(tf.Variable(b), tf.float32) 

    def get_weights(self):
        return [(self.name, self.B)]

    def num_params(self):
        return 0
        
    def forward(self, X):
        A = X
        return {'aout':A, 'cache':{}}
        
    ###################################################################           
        
    def backward(self, AI, AO, DO, cache):
        DI = DO
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache):    
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO, cache):
        DI = tf.matmul(E, self.B)
        return {'dout':DI, 'cache':{}}
        
    def dfa_gv(self, AI, AO, E, DO, cache):
        return []
        
    ###################################################################  
        
    def lel_backward(self, AI, AO, DO, Y, cache):
        S = tf.matmul(AO, tf.transpose(self.B))
        ES = tf.subtract(tf.nn.softmax(S), Y)
        DI = tf.matmul(ES, self.B)
        return {'dout':DI, 'cache':{}}
        
    def lel_gv(self, AI, AO, DO, Y, cache):
        return []
        
    ###################################################################  
        
        
        
