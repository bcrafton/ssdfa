
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, shape):
        self.shape = shape
        
    ###################################################################

    def get_weights(self):
        return []
        
    def num_params(self):
        return 0

    def forward(self, X):
        return tf.reshape(X, [tf.shape(X)[0], -1])
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return tf.reshape(DO, [tf.shape(AI)[0]] + self.shape)

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
