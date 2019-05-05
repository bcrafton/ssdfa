
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer

# http://cthorey.github.io./backpropagation/ 
# https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://kevinzakka.github.io/2016/09/14/batch_normalization/
# https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network

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
            '''
            if np.shape(gamma) != self.size:
                print (np.shape(gamma), self.size)
                assert(np.shape(gamma) == self.size)

            if np.shape(beta) != self.size:
                print (np.shape(beta), self.size)
                assert(np.shape(beta) == self.size)
            '''
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

    # means we are gonna need to make gamma and beta (1, shape) I think.
    # think we get away with it:
    # https://www.tensorflow.org/api_docs/python/tf/math/multiply
    # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    def forward(self, X):
        mean = tf.reduce_mean(X, axis=self.dims)
        _, var = tf.nn.moments(X - mean, axes=self.dims)
        xhat = (X - mean) / tf.sqrt(var + self.eps)
        Z = self.gamma * xhat + self.beta
        # Z = tf.Print(Z, [tf.shape(self.gamma), tf.shape(X)], message='', summarize=1000)
        return Z
            
    ###################################################################           
        
    def backward(self, AI, AO, DO): 
        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        ####### 
        mean = (1./N) * tf.reduce_sum(AI, axis=self.dims)
        var = (1./N) * tf.reduce_sum((AI - mean) ** 2, axis=self.dims)
        DI = (1./N) * self.gamma * ((var + self.eps) ** (-1. / 2.)) * (N * DO - tf.reduce_sum(DO, axis=self.dims) - (AI - mean) * ((var + self.eps) ** (-1.0)) * tf.reduce_sum(DO * (AI - mean), axis=self.dims))
        #######
        
        # DI = tf.Print(DI, [tf.shape(DI)], message='', summarize=1000)
        return DI

    def gv(self, AI, AO, DO):
        if not self._train:
            return []

        mean = tf.reduce_mean(AI, axis=self.dims)
        _, var = tf.nn.moments(AI - mean, axes=self.dims)
        xhat = (AI - mean) / tf.sqrt(var + self.eps)
        
        #######

        dgamma = tf.reduce_sum(DO * xhat, axis=self.dims)
        dbeta = tf.reduce_sum(DO, axis=self.dims)
        
        #######

        return [(dgamma, self.gamma), (dbeta, self.beta)]
        
    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        assert(False)
        return []
        
    ###################################################################

    
