
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.FeedbackMatrix import FeedbackMatrix

class FeedbackConv(Layer):

    def __init__(self, size, num_classes, sparse=0, rank=0, name=None):
        self.size = size
        self.batch_size, self.h, self.w, self.f = self.size
        self.num_output = self.h * self.w * self.f
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.name = name
        
        b = FeedbackMatrix(size=(self.num_classes, self.num_output), sparse=self.sparse, rank=self.rank)
        self.B = tf.cast(tf.Variable(b), tf.float32) 

    ###################################################################
    
    def get_weights(self):
        return [(self.name, self.B)]
    
    def num_params(self):
        return 0
        
    def forward(self, X):
        A = X
        return {'aout':A, 'cache':{}}
                
    ###################################################################           
        
    def bp(self, AI, AO, DO, cache):    
        DI = DO
        return {'dout':DI, 'cache':{}}, []

    def dfa(self, AI, AO, E, DO, cache):
        DI = tf.matmul(E, self.B)
        DI = tf.reshape(DI, self.size)
        return {'dout':DI, 'cache':{}}, []

    def lel(self, AI, AO, DO, Y, cache):
        shape = tf.shape(AO)
        N = shape[0]
        AO = tf.reshape(AO, (N, self.num_output))
        S = tf.matmul(AO, tf.transpose(self.B))
        ES = tf.subtract(tf.nn.softmax(S), Y)
        DI = tf.matmul(ES, self.B)
        DI = tf.reshape(DI, self.size)
        return {'dout':DI, 'cache':{}}, []
        
    ###################################################################
        
        


