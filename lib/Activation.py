
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
        return A, None

    #########

    def bp(self, AI, AO, DO, cache):    
        DI = tf.cast(AO > 0.0, dtype=tf.float32) * DO
        return DI, []

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)

###################################################################

class SignedRelu(Layer):

    def __init__(self, size, name, load):
        # self.signs = tf.constant(signs, dtype=tf.float32)

        self.size = size
        self.name = name
        self.load = load

        if self.load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1', allow_pickle=True).item()
            signs = weight_dict[self.name]
        else:
            signs = np.random.choice([1., -1.], size=self.size) 

        self.signs = tf.constant(signs, dtype=tf.float32)

    #########

    def get_weights(self):
        return [(self.name, self.signs)]
        
    def num_params(self):
        return 0

    def forward(self, x):
        A = tf.nn.relu(x) * self.signs
        return A, None

    #########

    def bp(self, AI, AO, DO, cache):    
        DI = tf.cast(AI > 0.0, dtype=tf.float32) * self.signs * DO
        return DI, []

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache): 
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
        return A, None

    #########

    def bp(self, AI, AO, DO, cache):    
        DI = (1. - tf.pow(AO, 2)) * DO
        return DI, []

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache): 
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
        
        
        
