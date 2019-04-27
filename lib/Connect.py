
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class Connect(Layer):

    def __init__(self, input_sizes, name=None, load=None, train=True):
        self.input_sizes = input_sizes
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        
        connect = np.eye(self.fin)
        connect = np.reshape(connect, (1, 1, 1, self.fin, self.fin))
        self.connect = tf.Variable(connect, dtype=tf.float32)
        
    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        connect_weights_size = self.fin * self.fin
        return connect_weights_size

    ###################################################################

    def forward(self, X):
        X = tf.reshape(X, (self.batch_size, self.h, self.w, self.fin, 1))
        Z = X * self.connect
        Z = tf.reduce_sum(Z, axis=4) / self.fin
        return Z
        
    # okay so writing these two is gonna suck.
    
    def backward(self, AI, AO, DO):
        DO = tf.reshape(DO, (self.batch_size, self.h, self.w, self.fin, 1))
        DI = DO * self.connect
        DI = tf.reduce_sum(DI, axis=4) / self.fin
        return DI

    def gv(self, AI, AO, DO):
        A = tf.reduce_sum(AI, axis=[1, 2]) / self.h / self.w
        D = tf.reduce_sum(DO, axis=[1, 2]) / self.h / self.w
        DC = tf.matmul(tf.transpose(A), D)
        DC = tf.reshape(DC, (1, 1, 1, self.fin, self.fin))
        return [(DC, self.connect)]
    
    def train(self, AI, AO, DO): 
        assert(False)
        return []
        
    ###################################################################

