
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class SparseFC(Layer):

    def __init__(self, size, num_classes, init_weights, alpha, activation, bias, last_layer, name=None, load=None, train=True, rate=1., swap=0., sign=1.):
        self.size = size
        self.input_size, self.output_size = size
        self.num_classes = num_classes
        self.last_layer = last_layer
        self.sign = sign
        self.rate = rate
        self.swap = swap
        self.nswap = int(self.rate * self.swap * self.input_size * self.output_size)
        self.bias = tf.Variable(tf.ones(shape=[self.output_size]) * bias)
        self.alpha = alpha
        self.activation = activation
        self.name = name
        self._train = train
        
        #########################################################

        '''
        mask = np.zeros(self.size)
        for ii in range(self.input_size):
            idx = np.random.choice(range(self.output_size), size=int(self.rate * self.output_size), replace=False)
            mask[ii][idx] = 1.
        '''

        mask = np.random.choice([0., -1., 1.], size=self.size, replace=True, p=[1.-rate, rate*(1.-sign), rate*sign])

        # total_connects = int(np.count_nonzero(mask))
        # assert(total_connects == int(self.rate * self.output_size) * self.input_size)
        
        assert(not load)
        if init_weights == "zero":
            # weights = np.zeros(shape=self.size)
            weights = np.ones(shape=self.size) * 1e-6

        elif init_weights == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.input_size)
            weights = np.random.uniform(low=1e-6, high=1.0/sqrt_fan_in, size=self.size)

        elif init_weights == "alexnet":
            assert(False)
            # weights = np.random.normal(loc=0.0, scale=0.01, size=self.size)

        else:
            # Glorot
            assert(False)

        weights = np.absolute(mask) * weights
            
        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.mask = tf.Variable(mask, dtype=tf.float32)
        self.total_connects = tf.Variable(tf.count_nonzero(self.mask))
        self.slice_size = np.count_nonzero(mask) - self.nswap
        
    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights * self.mask), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    def forward(self, X):
        Z = tf.matmul(X, tf.clip_by_value(self.weights, 1e-6, 1e6) * self.mask) + self.bias
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights * self.mask))
        return DI
        
    def gv(self, AI, AO, DO):
        # _assert = tf.assert_greater_equal(self.total_connects, tf.count_nonzero(self.weights))
        # with tf.control_dependencies([_assert]):

        if not self._train:
                return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        weights = tf.clip_by_value(self.weights - self.alpha * DW, 1e-6, 1e6) * tf.abs(self.mask)
        bias = self.bias - self.alpha * DB

        self.weights = self.weights.assign(weights)
        self.bias = self.bias.assign(bias)

        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        # _assert = tf.assert_greater_equal(self.total_connects, tf.count_nonzero(self.weights))
        # with tf.control_dependencies([_assert]):

        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        weights = tf.clip_by_value(self.weights - self.alpha * DW, 1e-6, 1e6) * tf.abs(self.mask)
        bias = self.bias - self.alpha * DB

        self.weights = self.weights.assign(weights)
        self.bias = self.bias.assign(bias)

        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
        
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.ones(shape=(tf.shape(AI)))
        
    def lel_gv(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]
        
    def lel(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def SET(self):
        shape = tf.shape(self.weights)
        abs_m = tf.abs(tf.identity(self.mask))
        vld_i = tf.where(abs_m > 0)
        vld_w = tf.gather_nd(self.weights, vld_i)
        sorted_i = tf.contrib.framework.argsort(vld_w, axis=0, direction="DESCENDING")

        new_i = tf.where(self.weights <= 0)
        new_i = tf.random_shuffle(new_i)
        new_i = tf.slice(new_i, [0, 0], [self.nswap, 2])
        new_i = tf.cast(new_i, tf.int32)
        sqrt_fan_in = math.sqrt(self.input_size)
        new_w = tf.random_uniform(minval=1e-6, maxval=1.0/sqrt_fan_in, shape=(self.nswap,))

        large_i = tf.gather(vld_i, sorted_i, axis=0)
        large_i = tf.cast(large_i, tf.int32)
        large_i = tf.slice(large_i, [0, 0], [self.slice_size, 2])
        large_w = tf.gather_nd(self.weights, large_i)

        # update weights
        indices = tf.concat((large_i, new_i), axis=0)
        updates = tf.concat((large_w, new_w), axis=0)
        weights = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

        # update mask
        num_pos = np.ceil(self.nswap * self.sign)
        num_neg = np.floor(self.nswap * (1 - self.sign))
        large_w = tf.gather_nd(self.mask, large_i)
        pos = tf.ones(shape=(num_pos, 1))
        neg = tf.ones(shape=(num_neg, 1)) * -1.
        new_w = tf.concat((pos, neg), axis=0)
        new_w = tf.reshape(new_w, (-1,))
        updates = tf.concat((large_w, new_w), axis=0)
        mask = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

        # assign 
        weights = self.weights.assign(weights)
        mask = self.mask.assign(mask)

        return [(mask, weights)]
        
    def NSET(self):    
        return [(self.mask, self.weights)]
        
    ###################################################################
        
    def set_fb(self, fb):
        masked = tf.multiply(self.weights, self.mask)
        if self.last_layer:
            fb = masked
        else:
            fb = tf.matmul(masked, fb)
            
        return (fb, [])

    def nset_fb(self, fb):
        masked = tf.multiply(self.weights, self.mask)
        if self.last_layer:
            fb = masked
        else:
            fb = tf.matmul(masked, fb)

        return (fb, [])

    ###################################################################
    
    
    
    

