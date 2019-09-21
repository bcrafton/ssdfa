
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
        self.variance_epsilon = eps
        self.num_parameters = np.prod(self.size) * 2
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1', allow_pickle=True).item()
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
        A, mean, var = tf.nn.fused_batch_norm(x=X, offset=self.beta, scale=self.gamma, mean=None, variance=None, epsilon=self.variance_epsilon, is_training=True)
        return A, (mean, var)

    ###################################################################

    def bp(self, AI, AO, DO, cache):
        mean, var = cache
        ivar = 1. / tf.sqrt(var + self.variance_epsilon)
        [DI, dgamma, dbeta, _, _] = gen_nn_ops.fused_batch_norm_grad_v2(y_backprop=DO, 
                                                                        x=AI, 
                                                                        scale=self.gamma, 
                                                                        reserve_space_1=mean, 
                                                                        reserve_space_2=ivar, 
                                                                        epsilon=self.variance_epsilon, 
                                                                        is_training=True)

        return DI, [(dgamma, self.gamma), (dbeta, self.beta)]

    def dfa(self, AI, AO, E, DO, cache):    
        return self.bp(AI, AO, DO, cache)
    
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ###################################################################  

    
    
    
    
    
