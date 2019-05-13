
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class ConvToFullyConnected(Layer):

    def __init__(self, input_shape):
        self.shape = input_shape
        
    ###################################################################

    def get_weights(self):
        return []
        
    def output_shape(self):
        return np.prod(self.shape)
        
    def num_params(self):
        return 0

    def forward(self, X):
        return tf.reshape(X, [tf.shape(X)[0], tf.shape(X)[1], -1])
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return tf.reshape(DO, [tf.shape(AI)[0], tf.shape(AI)[1]] + self.shape)

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []

    ###################################################################
