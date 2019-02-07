
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
from lib.Compute import RRAM

from lib.Movement import Movement
from lib.Movement import vonNeumann
from lib.Movement import Neuromorphic

class FullyConnected(Layer):

    def __init__(self, size, num_classes, init_weights, alpha, activation, bias, last_layer, l2=0., name=None, load=None, train=True):
        
        # TODO
        # check to make sure what we put in here is correct
        
        # input size
        self.size = size
        self.last_layer = last_layer
        self.input_size, self.output_size = size
        self.num_classes = num_classes

        # bias
        self.bias = tf.Variable(tf.ones(shape=[self.output_size]) * bias)

        # lr
        self.alpha = alpha

        # l2 loss lambda
        self.l2 = l2

        # activation function
        self.activation = activation

        self.name = name
        self._train = train
        
        self.memory = DRAM()
        self.compute = CMOS()
        self.movement = vonNeumann()
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load).item()
            self.weights = tf.Variable(weight_dict[self.name])
            self.bias = tf.Variable(weight_dict[self.name + '_bias'])
        else:
            if init_weights == "zero":
                weights = np.zeros(shape=self.size)
            elif init_weights == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.input_size)
                weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.size)
            elif init_weights == "alexnet":
                weights = np.random.normal(loc=0.0, scale=0.01, size=self.size)
            else:
                # glorot
                assert(False)

        self.weights = tf.Variable(weights, dtype=tf.float32)

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
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
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
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DB = tf.reduce_sum(DO, axis=0)
        
        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
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
        
    def metrics(self, dfa=False, sparsity=0., batch_size=1):
    
        #############################
    
        # forward
        self.memory.read(self.size)
        self.memory.read(self.output_size)
        
        self.compute.matmult(self.size, self.input_size)
        self.compute.add(self.output_size)
        
        self.movement.receive(self.input_size)
        self.movement.send(self.output_size)
        
        # backward
        self.memory.read(self.size)
        
        self.compute.matmult((self.input_size, self.output_size), self.output_size)
        
        self.movement.receive(self.output_size)
        self.movement.send(self.input_size)

        # update
        # self.memory.read(self.size) # done in backward
        self.memory.write(self.size)
        
        self.compute.mac((self.input_size, 1), (1, self.output_size))
        
        # self.movement.receive(self.output_size) # done in backward

        #############################
    
        read = self.memory.read_count
        write = self.memory.write_count
        
        mac = self.compute.mac_count
        add = self.compute.add_count
        
        send = self.movement.send_count
        receive = self.movement.receive_count
        
        #############################
        
        return [read, write, mac, add, send, receive]
        
        
        
        
        
        
