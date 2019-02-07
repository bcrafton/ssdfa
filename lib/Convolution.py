
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

from lib.Memory import Memory
from lib.Memory import DRAM
from lib.Memory import RRAM

from lib.Compute import Compute
from lib.Compute import CMOS
from lib.Compute import RRAM

from lib.Movement import Movement
from lib.Movement import vonNeumann
from lib.Movement import Neuromorphic

class Convolution(Layer):

    def __init__(self, input_sizes, filter_sizes, num_classes, init_filters, strides, padding, alpha, activation: Activation, bias, last_layer, name=None, load=None, train=True):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # self.h and self.w only equal this for input sizes when padding = "SAME"...
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes

        self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)

        self.strides = strides
        self.padding = padding

        self.alpha = alpha

        self.activation = activation
        self.last_layer = last_layer

        self.name = name
        self._train = train

        self.memory = DRAM()
        self.compute = CMOS()
        self.movement = vonNeumann()

        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1').item()
            self.filters = tf.Variable(weight_dict[self.name])
            self.bias = tf.Variable(weight_dict[self.name + '_bias'])
        else:
            if init_filters == "zero":
                filters = np.zeros(shape=self.filter_sizes)
            elif init_filters == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
                filters = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.filter_sizes)
            elif init_filters == "alexnet":
                filters = np.random.normal(loc=0.0, scale=0.01, size=self.filter_sizes)
            else:
                # glorot
                assert(False)
                
        self.filters = tf.Variable(filters, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        Z = tf.add(tf.nn.conv2d(X, self.filters, self.strides, self.padding), tf.reshape(self.bias, [1, 1, self.fout]))
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        return DI

    def gv(self, AI, AO, DO):    
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################    
        
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.ones(shape=(tf.shape(AI)))
        
    def lel_gv(self, AI, AO, E, DO, Y):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def lel(self, AI, AO, E, DO, Y): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ################################################################### 

    # for convolution we can just use conv_utils code.
    def metrics(self, dfa=False, sparsity=0., examples=1, epochs=1):
        
        total_examples = examples * epochs
        
        # filter_size = self.filter_sizes # filters
        # output_size = self.fout # bias
        input_size = (total_examples, self.h, self.w, self.fin)

        output_row = conv_output_length(self.h, self.fh, self.padding.lower(), self.strides[1])
        output_col = conv_output_length(self.w, self.fw, self.padding.lower(), self.strides[2])
        output_size = (total_examples, output_row, output_col, self.fout)

        #############################
    
        # forward
        self.memory.read(self.filter_sizes)
        self.memory.read(self.fout)
        
        self.compute.conv(self.filter_sizes, input_size, self.padding, self.strides)
        self.compute.add(self.fout)
        
        self.movement.receive(input_size)
        self.movement.send(output_size)
        
        # backward
        if not dfa:
            self.memory.read(self.filter_sizes)
            
            self.compute.conv(self.filter_sizes, output_size, "full", [1, 1, 1, 1])
            
            self.movement.receive(output_size)
            self.movement.send(input_size)

        # update
        # self.memory.read(size) # done in backward
        self.memory.write(self.filter_sizes)
        
        self.compute.conv(output_size, input_size, self.padding, self.strides)
        
        # self.movement.receive(output_size) # done in backward

        #############################
    
        read = self.memory.read_count
        write = self.memory.write_count
        
        mac = self.compute.mac_count
        add = self.compute.add_count
        
        send = self.movement.send_count
        receive = self.movement.receive_count
        
        #############################
        
        return [read, write, mac, add, send, receive]
        
        
        
