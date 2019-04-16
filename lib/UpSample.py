
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 

class UpSample(Layer):
    def __init__(self, size, ksize):
        self.size = size
        self.batch, self.h, self.w, self.fin = self.size
        self.ksize = ksize

    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    ###################################################################

    def forward(self, X):
        N = tf.shape(X)[0]
        
        Z = X
        Z = tf.reshape(Z, (N, self.h * self.w, 1, self.fin))
        Z = tf.tile(Z, [1, 1, self.ksize * self.ksize, 1])
        Z = tf.reshape(Z, (N, self.h, self.w, self.ksize, self.ksize, self.fin))
        Z = tf.reshape(Z, (N, self.h, self.w * self.ksize, self.ksize, self.fin))
        # think there is a bug here
        # u shud have to transpose the first ksize to h before combining with w or something
        # but bc they all the same values, it dosnt matter.
        Z = tf.transpose(Z, (0, 1, 3, 2, 4))
        Z = tf.reshape(Z, (N, self.h * self.ksize, self.w * self.ksize, self.fin))
        return Z
        
    def backward(self, AI, AO, DO):
        N = tf.shape(DO)[0]
        
        DI = DO
        DI = tf.reshape(DI, (N, self.h, self.ksize, self.w, self.ksize, self.fin))
        DI = tf.transpose(DI, (0, 1, 3, 2, 4, 5))
        DI = tf.reshape(DI, (N, self.h, self.w, self.ksize * self.ksize, self.fin))
        DI = tf.reduce_mean(DI, axis=3)
        return DI

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

