
import tensorflow as tf
import numpy as np
import math
import itertools

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid

class FullyConnected(Layer):

    def __init__(self, size, num_classes, init_weights, alpha, activation, bias, last_layer, l2=0., name=None, load=None, train=True):
    
        self.size = size
        self.last_layer = last_layer
        self.input_size, self.output_size = size
        self.num_classes = num_classes
        self.bias = tf.Variable(tf.ones(shape=[self.output_size]) * bias)
        self.alpha = alpha
        self.l2 = l2
        self.activation = activation
        self.name = name
        self._train = train
        
        self.num = int(0.5 * np.prod(self.size))
        
        combs = np.array(list(itertools.product(range(self.input_size), range(self.output_size))))
        choices = range(len(combs))
        idx = np.random.choice(a=choices, size=self.num, replace=False).tolist()
        idx = combs[idx]
        
        sqrt_fan_in = math.sqrt(self.input_size)
        val = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.num)
        
        print (np.shape(idx))
        print (np.shape(val))
        
        self.idx = tf.Variable(idx, dtype=tf.int64)
        self.val = tf.Variable(val, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.val), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    def forward(self, X):
        weights = tf.SparseTensor(indices=self.idx, values=self.val, dense_shape=self.size)
        # Z = tf.matmul(X, weights) + self.bias
        Z = tf.sparse_tensor_dense_matmul(tf.sparse_transpose(weights), tf.transpose(X))
        Z = tf.transpose(Z) + self.bias
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
    
        weights = tf.SparseTensor(indices=self.idx, values=self.val, dense_shape=self.size)
        # DI = tf.matmul(DO, tf.transpose(self.weights))
        
        DI = tf.sparse_tensor_dense_matmul(weights, tf.transpose(DO))
        DI = tf.transpose(DI)
        
        return DI
        
    def gv(self, AI, AO, DO):
        if not self._train:
            return []
            
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DB = tf.reduce_sum(DO, axis=0)
        
        # DW = tf.matmul(tf.transpose(AI), DO) 
        slice1 = tf.slice(self.idx, [0, 0], [self.num, 1])
        slice2 = tf.slice(self.idx, [0, 1], [self.num, 1])
        slice_AI = tf.gather_nd(tf.transpose(AI), slice1)
        slice_DO = tf.gather_nd(tf.transpose(DO), slice2)
        DW = tf.multiply(slice_AI, slice_DO)
        DW = tf.reduce_sum(DW, axis=1)

        return [(DW, self.val), (DB, self.bias)]

    def train(self, AI, AO, DO):
        assert(False)
    
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################

