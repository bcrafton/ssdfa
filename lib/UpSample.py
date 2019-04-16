
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
        Z = X
        Z = np.reshape(Z, (self.batch, self.h * self.w, 1, self.fin))
        Z = tf.tile(Z, [1, 1, self.ksize * self.ksize, 1])
        Z = tf.reshape(Z, (self.batch, self.h, self.w, self.ksize, self.ksize, self.fin))
        Z = tf.reshape(Z, (self.batch, self.h, self.w * self.ksize, self.ksize, self.fin))
        # think there is a bug here
        # u shud have to transpose the first ksize to h before combining with w or something
        # but bc they all the same values, it dosnt matter.
        Z = np.transpose(Z, (0, 1, 3, 2, 4))
        Z = np.reshape(Z, (self.batch, self.h * self.ksize, self.w * self.ksize, self.fin))
        return Z
        
    def backward(self, AI, AO, DO):
        DI = DO
        DI = tf.reshape(DI, (self.h, self.ksize, self.w, self.ksize, self.fin))
        DI = tf.transpose(DI, (0, 2, 1, 3, 4))
        DI = tf.reshape(DI, (self.h, self.w, self.ksize * self.ksize, self.fin))
        DI = tf.reduce_mean(DI, axis=2)
        return DI

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

