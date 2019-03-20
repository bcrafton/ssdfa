
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops
# return gen_nn_ops.max_pool_v2(value=X, ksize=self.size, strides=self.strides, padding="SAME")

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid

from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

class MaxPool(Layer):
    def __init__(self, batch_size, input_shape, ksize, strides, padding):
        self.input_shape = input_shape
        self.h, self.w, self.fin = self.input_shape
        self.batch_size = batch_size 
        
        self.ksize = ksize
        self.kh, self.kw = self.ksize
        self.full_ksize = [1, self.kh, self.kw, 1]
        
        self.strides = strides
        self.sh, self.sw = self.strides
        self.full_strides = [1, self.sh, self.sw, 1]
        
        self.padding = padding

    ###################################################################

    def get_weights(self):
        return []

    def output_shape(self):
        oh = conv_output_length(self.h, self.kh, self.padding.lower(), self.sh)
        ow = conv_output_length(self.w, self.kw, self.padding.lower(), self.sw)
        od = self.fin
        return [oh, oh, od]

    def num_params(self):
        return 0

    def forward(self, X):
        Z = tf.nn.max_pool(X, ksize=self.full_ksize, strides=self.full_strides, padding=self.padding)
        # Z = tf.Print(Z, [Z], message="", summarize=1000)
        return Z
            
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.full_ksize, strides=self.full_strides, padding=self.padding)
        return grad

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.full_ksize, strides=self.full_strides, padding=self.padding)
        # grad = tf.Print(grad, [tf.shape(grad), tf.count_nonzero(tf.equal(grad, 1)), tf.count_nonzero(tf.equal(grad, 2)), tf.count_nonzero(tf.equal(grad, 3)), tf.count_nonzero(tf.equal(grad, 4)), tf.count_nonzero(tf.equal(grad, 5))], message="", summarize=1000)
        return grad
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.full_ksize, strides=self.full_strides, padding=self.padding)
        # grad = tf.Print(grad, [tf.shape(grad), tf.count_nonzero(tf.equal(grad, 1)), tf.count_nonzero(tf.equal(grad, 2)), tf.count_nonzero(tf.equal(grad, 3)), tf.count_nonzero(tf.equal(grad, 4)), tf.count_nonzero(tf.equal(grad, 5))], message="", summarize=1000)
        return grad
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
    
    
