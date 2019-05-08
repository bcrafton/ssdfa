
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops
# return gen_nn_ops.max_pool_v2(value=X, ksize=self.size, strides=self.strides, padding="SAME")

from lib.Layer import Layer 

from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

class MaxPool(Layer):
    def __init__(self, size, ksize, strides, padding):
        self.size = size
        self.batch_size, self.h, self.w, self.fin = self.size
        self.ksize = ksize
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        self.padding = padding

    ###################################################################

    def get_weights(self):
        return []

    def output_shape(self):
        oh = conv_output_length(self.h, self.fh, self.padding.lower(), self.sh)
        ow = conv_output_length(self.w, self.fw, self.padding.lower(), self.sw)
        od = self.fout
        return [oh, oh, od]

    def num_params(self):
        return 0

    def forward(self, X):
        Z = tf.nn.max_pool(X, ksize=self.ksize, strides=self.strides, padding=self.padding)
        # Z = tf.Print(Z, [Z], message="", summarize=1000)
        return Z
            
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return grad

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        # grad = tf.Print(grad, [tf.shape(grad), tf.count_nonzero(tf.equal(grad, 1)), tf.count_nonzero(tf.equal(grad, 2)), tf.count_nonzero(tf.equal(grad, 3)), tf.count_nonzero(tf.equal(grad, 4)), tf.count_nonzero(tf.equal(grad, 5))], message="", summarize=1000)
        return grad
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        # grad = tf.Print(grad, [tf.shape(grad), tf.count_nonzero(tf.equal(grad, 1)), tf.count_nonzero(tf.equal(grad, 2)), tf.count_nonzero(tf.equal(grad, 3)), tf.count_nonzero(tf.equal(grad, 4)), tf.count_nonzero(tf.equal(grad, 5))], message="", summarize=1000)
        return grad
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
    
    
