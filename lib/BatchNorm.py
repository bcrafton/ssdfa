
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops

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

    # means we are gonna need to make gamma and beta (1, shape) I think.
    # think we get away with it:
    # https://www.tensorflow.org/api_docs/python/tf/math/multiply
    # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    def forward1(self, X):
        mean = tf.reduce_mean(X, axis=self.dims)
        _, var = tf.nn.moments(X - mean, axes=self.dims)
        xhat = (X - mean) / tf.sqrt(var + self.eps)
        Z = self.gamma * xhat + self.beta

        # Z = tf.Print(Z, [mean], message='', summarize=1000)
        
        return Z

    def forward2(self, X):
        mean = tf.reduce_mean(X, axis=self.dims)
        _, var = tf.nn.moments(X - mean, axes=self.dims)
        
        Z = tf.nn.batch_normalization(x=X, mean=mean, variance=var, offset=self.beta, scale=self.gamma, variance_epsilon=self.eps)

        return Z

    def forward(self, X):
        return self.forward2(X)
            
    ###################################################################           
        
    def backward1(self, AI, AO, DO): 
        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        ####### 
        mean = (1./N) * tf.reduce_sum(AI, axis=self.dims)
        var = (1./N) * tf.reduce_sum((AI - mean) ** 2, axis=self.dims)
        DI = (1./N) * self.gamma * ((var + self.eps) ** (-1. / 2.)) * (N * DO - tf.reduce_sum(DO, axis=self.dims) - (AI - mean) * ((var + self.eps) ** (-1.0)) * tf.reduce_sum(DO * (AI - mean), axis=self.dims))
        #######
        
        # DI = tf.Print(DI, [tf.shape(DI)], message='', summarize=1000)
        return DI

    def backward2(self, AI, AO, DO): 
        mean = tf.reduce_mean(AI, axis=self.dims)
        _, var = tf.nn.moments(AI - mean, axes=self.dims)
        # inverted variance in the cuDNN case
        # so how do we do this ? 
        # grep -r "var_to_inv_var" * -B 4
        # lib/python3.5/site-packages/tensorflow/include/tensorflow/stream_executor/dnn.h-  //  reserve_space_2: saved inv_var (1/sqrt(epsilon + variance), to be reused
        ivar = 1. / tf.sqrt(self.eps + var)

        '''
        # reserve_space_1: When is_training is True, a 1D Tensor for the computed batch mean to be reused in gradient computation. 
                           When is_training is False, a 1D Tensor for the population mean to be reused in both 1st and 2nd order gradient computation.
        # reserve_space_2: When is_training is True, a 1D Tensor for the computed batch variance (inverted variance in the cuDNN case) to be reused in gradient computation. 
                           When is_training is False, a 1D Tensor for the population variance to be reused in both 1st and 2nd order gradient computation.
        '''

        [DI, dgamma, dbeta, reserve_space_3, reserve_space_4] = gen_nn_ops.fused_batch_norm_grad_v2(y_backprop=DO, x=AI, scale=self.gamma, reserve_space_1=mean, reserve_space_2=ivar, epsilon=self.eps, is_training=True)
        # DI = tf.Print(DI, [tf.shape(dgamma), tf.shape(dbeta)], message='', summarize=1000)
        return DI

    def backward(self, AI, AO, DO):
        return self.backward2(AI, AO, DO)

    ###################################################################

    def gv1(self, AI, AO, DO):
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

    def gv2(self, AI, AO, DO):
        if not self._train:
            return []

        mean = tf.reduce_mean(AI, axis=self.dims)
        _, var = tf.nn.moments(AI - mean, axes=self.dims)
        # inverted variance in the cuDNN case
        # so how do we do this ? 
        # grep -r "var_to_inv_var" * -B 4
        # lib/python3.5/site-packages/tensorflow/include/tensorflow/stream_executor/dnn.h-  //  reserve_space_2: saved inv_var (1/sqrt(epsilon + variance), to be reused
        ivar = 1. / tf.sqrt(self.eps + var)
        
        '''
        # reserve_space_1: When is_training is True, a 1D Tensor for the computed batch mean to be reused in gradient computation. 
                           When is_training is False, a 1D Tensor for the population mean to be reused in both 1st and 2nd order gradient computation.
        # reserve_space_2: When is_training is True, a 1D Tensor for the computed batch variance (inverted variance in the cuDNN case) to be reused in gradient computation. 
                           When is_training is False, a 1D Tensor for the population variance to be reused in both 1st and 2nd order gradient computation.
        '''

        [DI, dgamma, dbeta, reserve_space_3, reserve_space_4] = gen_nn_ops.fused_batch_norm_grad_v2(y_backprop=DO, x=AI, scale=self.gamma, reserve_space_1=mean, reserve_space_2=ivar, epsilon=self.eps, is_training=True)
        return [(dgamma, self.gamma), (dbeta, self.beta)]

    def gv(self, AI, AO, DO):
        return self.gv2(AI, AO, DO)

    ###################################################################

    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        assert(False)
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return self.backward(AI, AO, DO)
        
    def dfa_gv(self, AI, AO, E, DO):
        return self.gv(AI, AO, DO)
        
    def dfa(self, AI, AO, E, DO): 
        return self.train(AI, AO, DO)
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y):
        return self.backward(AI, AO, DO)
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return self.gv(AI, AO, DO)
        
    def lel(self, AI, AO, E, DO, Y): 
        return self.train(AI, AO, DO)
        
    ###################################################################  

    
    
    
    
    
