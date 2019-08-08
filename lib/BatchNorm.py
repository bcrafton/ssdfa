
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops

from lib.Layer import Layer

class BatchNorm(Layer):
    def __init__(self, input_size, name=None, load=None, train=True, eps=1e-3):
        self.input_size = list(input_size)
        if len(self.input_size) == 2:
            self.dims = [0]
        elif len(self.input_size) == 4:
            self.dims = [0, 1, 2]
        else:
            assert(False)
        self.size = self.input_size[-1]

        self.name = name
        self._train = train
        self.eps = eps
        self.num_parameters = np.prod(self.size) * 2
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load).item()
            gamma = weight_dict[self.name + '_gamma']
            beta = weight_dict[self.name + '_beta']
            
            if np.shape(gamma) != (self.size,):
                print (np.shape(gamma), self.size)
                assert(np.shape(gamma) == (self.size,))

            if np.shape(beta) != (self.size,):
                print (np.shape(beta), self.size)
                assert(np.shape(beta) == (self.size,))
            
        else:
            gamma = np.ones(shape=self.size)
            beta = np.zeros(shape=self.size)
        
        self.gamma = tf.Variable(gamma, dtype=tf.float32)
        self.beta = tf.Variable(beta, dtype=tf.float32)
        
    ###################################################################

    def get_weights(self):
        return [(self.name + '_gamma', self.gamma), (self.name + '_beta', self.beta)]

    def num_params(self):
        return self.num_parameters

    def output_shape(self):
        if len(self.input_size) == 2:
            return self.input_size[1]
        elif len(self.input_size) == 4:
            return self.input_size[1:4]
        else:
            assert(False)

    def forward(self, X):
        mean = tf.reduce_mean(X, axis=self.dims)
        _, var = tf.nn.moments(X - mean, axes=self.dims)
        A = tf.nn.batch_normalization(x=X, mean=mean, variance=var, offset=self.beta, scale=self.gamma, variance_epsilon=self.eps)
        return {'aout':A, 'cache':{}}

    ###################################################################

    def bp(self, AI, AO, DO, cache):
        mean = tf.reduce_mean(AI, axis=self.dims)
        _, var = tf.nn.moments(AI - mean, axes=self.dims)
        ivar = 1. / tf.sqrt(self.eps + var)

        if len(self.input_size) == 2:
            AI = tf.reshape(AI, (self.input_size[0], 1, 1, self.input_size[1]))
            DO = tf.reshape(AI, (self.input_size[0], 1, 1, self.size))
            
        [DI, dgamma, dbeta, _, _] = gen_nn_ops.fused_batch_norm_grad_v2(y_backprop=DO, x=AI, scale=self.gamma, reserve_space_1=mean, reserve_space_2=ivar, epsilon=self.eps, is_training=True)
        
        if len(self.input_size) == 2:
            DI = tf.reshape(DI, (self.input_size[0], self.size))
            
        return {'dout':DI, 'cache':{}}, [(dgamma, self.gamma), (dbeta, self.beta)]

    def dfa(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, cache):
        return self.bp(AI, AO, DO, cache)

    ###################################################################  

    
    
    
    
    
