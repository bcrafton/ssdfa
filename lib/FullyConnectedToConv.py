
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class FullyConnectedToConv(Layer):

    def __init__(self, shape_in, shape_out):
        self.shape_in = shape_in
        self.shape_out = shape_out
        
    ###################################################################

    def get_weights(self):
        return []
        
    def num_params(self):
        return 0

    def forward(self, X):
        N = tf.shape(X)[0]
        return tf.reshape(X, [N] + self.shape_out)
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):
        N = tf.shape(AI)[0]    
        return tf.reshape(DO, [N] + self.shape_in)

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################    
    
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.ones(shape=(tf.shape(AI)))
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
