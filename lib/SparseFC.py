
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid

from lib.Memory import Memory
from lib.Memory import DRAM
from lib.Memory import RRAM

from lib.Compute import Compute
from lib.Compute import CMOS

from lib.Movement import Movement
from lib.Movement import vonNeumann
from lib.Movement import Neuromorphic

from lib.add_dict import add_dict

class SparseFC(Layer):

    def __init__(self, size, num_classes, init_weights, alpha, activation, bias, last_layer, l2=0., name=None, load=None, train=True, rate=1.):
        self.size = size
        self.last_layer = last_layer
        self.rate = rate
        self.input_size, self.output_size = size
        self.num_classes = num_classes
        self.bias = tf.Variable(tf.ones(shape=[self.output_size]) * bias)
        self.alpha = alpha
        self.l2 = l2
        self.activation = activation
        self.name = name
        self._train = train
        
        #########################################################

        mask = np.random.choice([0., 1.], size=self.size, replace=True, p=[1.-rate, rate])
            
        # total_connects = int(np.count_nonzero(mask))
        # assert(total_connects == int(self.rate * self.output_size) * self.input_size)
        
        assert(not load)
        if init_weights == "zero":
            weights = np.zeros(shape=self.size)
            
        elif init_weights == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.input_size)
            weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.size)
            
        elif init_weights == "alexnet":
            weights = np.random.normal(loc=0.0, scale=0.01, size=self.size)
            
        else:
            # Glorot
            assert(False)

        weights = mask * weights
            
        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.mask = tf.Variable(mask, dtype=tf.float32)
        self.total_connects = tf.Variable(tf.count_nonzero(self.mask))
        
    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    def forward(self, X):
        Z = tf.matmul(X, self.weights) + self.bias
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights))
        return DI
        
    def gv(self, AI, AO, DO):
        if not self._train:
            return []
            
        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
        # assert(tf.count_nonzero(self.weights) == self.total_connects)
        # _assert = tf.assert_greater_equal(self.total_connects, tf.count_nonzero(self.weights))

        # with tf.control_dependencies([_assert]):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DW = tf.multiply(DW, self.mask) 
        DB = tf.reduce_sum(DO, axis=0)
        
        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
        # assert(tf.count_nonzero(self.weights) == self.total_connects)
        # _assert = tf.assert_greater_equal(self.total_connects, tf.count_nonzero(self.weights))

        # with tf.control_dependencies([_assert]):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
        
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.ones(shape=(tf.shape(AI)))
        
    def lel_gv(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DB = tf.reduce_sum(DO, axis=0)
        
        return [(DW, self.weights), (DB, self.bias)]
        
    def lel(self, AI, AO, E, DO, Y):
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
    
    def metrics(self, dfa=False, memory=None, compute=None, movement=None, examples=1, epochs=1):

        memory = DRAM() if memory is None else memory
        compute = CMOS() if compute is None else compute
        movement = vonNeumann() if movement is None else movement

        total_examples = examples * epochs
    
        size = (self.input_size, self.output_size)
        size_T = (self.output_size, self.input_size)
        
        input_size = (total_examples, self.input_size)
        input_size_T = (self.input_size, total_examples)
        
        output_size = (total_examples, self.output_size)
        output_size_T = (self.output_size, total_examples)

        #############################
    
        # forward
        
        if type(memory) in [DRAM]:
            memory.read(size, rate_X=self.rate)
            compute.matmult(input_size, size, rate_Y=self.rate)
        elif type(memory) in [RRAM]:
            memory.matmult(input_size, size, rate_Y=self.rate)
        else:
            assert(False)
        
        movement.receive(input_size)
        movement.send(output_size)
        
        # backward
        if not dfa:
            if type(memory) in [DRAM]:
                memory.read(size, rate_X=self.rate)
                compute.matmult(output_size, size_T, rate_Y=self.rate)
            elif type(memory) in [RRAM]:
                memory.matmult(output_size, size_T, rate_Y=self.rate)
            else:
                assert(False)
            
            movement.receive(output_size)
            movement.send(input_size)

        # update
        # memory.read(size) # done in backward
        memory.write(size, rate_X=self.rate)
        
        compute.matmult(input_size_T, output_size, rate_Y=self.rate)
        
        # movement.receive(output_size) # done in backward

        #############################
    
        total = {}
        total = add_dict(total, memory.total())
        total = add_dict(total, compute.total())
        total = add_dict(total, movement.total())
        
        #############################
        
        return total
        
        

