
import numpy as np
import tensorflow as tf
from lib.Layer import Layer

###################################################################

class Relu(Layer):

    def __init__(self):
        pass

    #########

    def get_weights(self):
        return []
        
    def num_params(self):
        return 0

    def forward(self, x):
        A = tf.nn.relu(x)
        return {'aout': A, 'cache': {}}

    #########

    def bp(self, AI, AO, DO, cache):    
        DI = tf.cast(AO > 0.0, dtype=tf.float32) * DO
        return {'dout': DI, 'cache': {}}, []

    def dfa(self, AI, AO, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, cache): 
        return self.bp(AI, AO, DO, cache)

###################################################################

class Tanh(Layer):

    def __init__(self):
        pass

    #########

    def get_weights(self):
        return []
        
    def num_params(self):
        return 0

    def forward(self, x):
        A = tf.tanh(x)
        return {'aout': A, 'cache': {}}

    #########

    def bp(self, AI, AO, DO, cache):    
        DI = (1. - tf.pow(AO, 2)) * DO
        return {'dout': DI, 'cache': {}}, []

    def dfa(self, AI, AO, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, cache): 
        return self.bp(AI, AO, DO, cache)

###################################################################

'''
class Tanh(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return tf.tanh(x)

    def gradient(self, x):
        # this is gradient wtf A, not Z
        return 1 - tf.pow(x, 2)
'''
        
        
        
