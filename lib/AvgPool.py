
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
        _, self.kh, self.kw, _ = self.ksize
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        
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
        shape_AO = tf.shape(AO)[3]
        shape_DO = tf.shape(DO)[3]
        assert_op = tf.assert_equal(shape_AO, shape_DO)
        with tf.control_dependencies([assert_op]):
            grad = gen_nn_ops.avg_pool_grad(orig_input_shape=self.size, grad=DO, ksize=self.ksize, strides=self.strides, padding=self.padding)
            # grad = tf.Print(grad, [tf.shape(AI), tf.shape(AO), tf.shape(DO)], message='', summarize=1000)
            # grad = tf.Print(grad, ['pool', tf.reduce_sum(DO), tf.reduce_sum(grad)], message='', summarize=1000)
            return grad

    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
    
    
