
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer

# http://cthorey.github.io./backpropagation/ 
# https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://kevinzakka.github.io/2016/09/14/batch_normalization/

class BatchNorm(Layer):
    def __init__(self, size, eps=1e-3):
        self.size = size
        self.eps = eps
        self.num_parameters = np.prod(self.size) * 2
        
        gamma = np.ones(shape=size)
        beta = np.zeros(shape=size)
        
        self.gamma = tf.Variable(gamma, dtype=tf.float32)
        self.beta = tf.Variable(beta, dtype=tf.float32)
        
    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return self.num_parameters

    def forward(self, X):
        mean = tf.reduce_mean(X, axis=0, keepdims=True)
        _, var = tf.nn.moments(X - mean, axes=0, keep_dims=True)
        xhat = (X - mean) / tf.sqrt(var + self.eps)
        # means we are gonna need to make gamma and beta (1, shape) I think.
        # think we get away with it:
        # https://www.tensorflow.org/api_docs/python/tf/math/multiply
        # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        Z = self.gamma * xhat + self.beta
        # Z = tf.Print(Z, [tf.shape(self.gamma), tf.shape(X)], message='', summarize=1000)
        return Z
            
    ###################################################################           
        
    def backward(self, AI, AO, DO): 
        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)
        X = AI
        
        ####### 
        mean = (1./N) * tf.reduce_sum(X, axis=0)
        var = (1./N) * tf.reduce_sum((X - mean) ** 2, axis=0)
        DI = (1./N) * self.gamma * ((var + self.eps) ** (-1. / 2.)) * (N * DO - tf.reduce_sum(DO, axis=0) - (X - mean) * ((var + self.eps) ** (-1.0)) * tf.reduce_sum(DO * (X - mean), axis=0))
        #######
        
        return DI

    def gv(self, AI, AO, DO):    
        X = AI
        
        mean = tf.reduce_mean(X, axis=0, keepdims=True)
        _, var = tf.nn.moments(X - mean, axes=0, keep_dims=True)
        xhat = (X - mean) / tf.sqrt(var + self.eps)
        
        #######

        dgamma = tf.reduce_sum(DO * xhat, axis=0)
        dbeta = tf.reduce_sum(DO, axis=0)
        
        #######

        return [(dgamma, self.gamma), (dbeta, self.beta)]
        
    def train(self, AI, AO, DO): 
        assert(False)
        return []
        
    ###################################################################

    
