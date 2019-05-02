
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops

from lib.Layer import Layer 

# /home/brian/environments/py3/lib/python3.5/site-packages/tensorflow/python/ops/gen_nn_ops.py
# def avg_pool_grad(orig_input_shape, grad, ksize, strides, padding, data_format="NHWC", name=None):

class AvgPool(Layer):
    def __init__(self, size, ksize, strides, padding):
        self.size = size
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    def forward(self, X):
        Z = tf.nn.avg_pool(X, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return Z
            
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        grad = gen_nn_ops.avg_pool_grad(orig_input_shape=self.size, grad=DO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return grad

    def gv(self, AI, AO, DO):    
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
    
    def lel_backward(self, AI, AO, E, DO, Y):
        grad = gen_nn_ops.avg_pool_grad(orig_input_shape=self.size, grad=DO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return grad
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
    
    
