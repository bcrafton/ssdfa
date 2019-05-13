
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
        self.batch, self.times, self.h, self.w, self.fin = self.size
        
        self.ksize = ksize
        _, self.kh, self.kw, _ = self.ksize
        
        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        
        self.padding = padding

        self.oh = conv_output_length(self.h, self.kh, self.padding.lower(), self.sh)
        self.ow = conv_output_length(self.w, self.kw, self.padding.lower(), self.sw)
        self.od = self.fin

    ###################################################################

    def get_weights(self):
        return []

    def output_shape(self):
        return [self.oh, self.oh, self.od]

    def num_params(self):
        return 0

    def forward(self, X):
        X = tf.reshape(X, (self.batch * self.times, self.h, self.w, self.fin))
        Z = tf.nn.max_pool(X, ksize=self.ksize, strides=self.strides, padding=self.padding)
        Z = tf.reshape(Z, (self.batch, self.times, self.oh, self.ow, self.od))
        return Z
            
    ###################################################################           
        
    def backward(self, AI, AO, DO):
        AI = tf.reshape(AI , (self.batch * self.times, self.h, self.w, self.fin))
        DO = tf.reshape(DO, (self.batch * self.times, self.oh, self.ow, self.od))
        AO =tf.reshape(AO, (self.batch * self.times, self.oh, self.ow, self.od))
        
        DI = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        DI = tf.reshape(DI, (self.batch, self.times, self.h, self.w, self.fin))
        return DI

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################
