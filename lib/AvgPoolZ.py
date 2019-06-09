
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops

from lib.Layer import Layer 

from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

# /home/brian/environments/py3/lib/python3.5/site-packages/tensorflow/python/ops/gen_nn_ops.py
# def avg_pool_grad(orig_input_shape, grad, ksize, strides, padding, data_format="NHWC", name=None):

class AvgPool(Layer):
    def __init__(self, size, ksize, strides, padding):
        self.size = size
        self.batch_size, self.h, self.w, self.fin = self.size
        
        self.ksize = ksize
        _, self.kh, self.kw, self.kz = self.ksize
        
        self.strides = strides
        _, self.sh, self.sw, self.sz = self.strides
        
        self.padding = padding
        
        assert(self.kh == self.sh)
        assert(self.kw == self.sw)
        assert(self.kz == self.sz)
        
        self.oh = self.h   // self.kh
        self.ow = self.w   // self.kw
        self.oz = self.fin // self.kz

    ###################################################################

    def get_weights(self):
        return []

    def output_shape(self):
        return [self.oh, self.ow, self.oz]

    def num_params(self):
        return 0

    def forward(self, X):
        # A = tf.nn.avg_pool(X, ksize=self.ksize, strides=self.strides, padding=self.padding)
        
        A = tf.reshape(X, [self.batch_size, self.kh, self.oh, self.kw, self.ow, self.kz, self.oz])
        A = tf.reduce_mean(A, [1, 3, 5])
        return {'aout':A, 'cache':{}}
            
    ###################################################################           
        
    def backward(self, AI, AO, DO, cache=None):    
        # DI = gen_nn_ops.avg_pool_grad(orig_input_shape=self.size, grad=DO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        
        DI = tf.reshape(DO, [self.batch_size, 1, self.oh, 1, self.ow, 1, self.oz])

        DI = tf.zeros(shape=[self.batch_size, self.kh, self.oh, self.kw, self.ow, self.kz, self.oz]) + (1. / (self.kw * self.kw * self.kz)) * DI

        DI = tf.reshape(DI, [self.batch_size, self.h, self.w, self.fin])
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache=None):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        grad = gen_nn_ops.avg_pool_grad(orig_input_shape=self.size, grad=DO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return grad
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y, cache):
        return self.backward(AI, AO, DO, cache)

    def lel_gv(self, AI, AO, E, DO, Y, cache):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
    
    
