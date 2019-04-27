
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid

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
        
        connect = np.ones(shape=(1, 1, self.fin, self.fout))
        # connect = np.zeros(shape=(1, 1, self.fin, self.fout))
        self.connect = tf.Variable(connect, dtype=tf.float32)
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1').item()
            
            filters = weight_dict[self.name]

            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            _filters = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.filter_sizes)

            filters = filters * (np.std(_filters) / np.std(filters))

            self.filters = tf.Variable(filters, dtype=tf.float32)
            # self.bias = tf.Variable(weight_dict[self.name + '_bias'])
            
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
        # Z = tf.add(tf.nn.conv2d(X, self.filters, self.strides, self.padding), tf.reshape(self.bias, [1, 1, self.fout]))
        Z = tf.nn.conv2d(X, self.filters * self.connect, self.strides, self.padding)
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters * self.connect, out_backprop=DO, strides=self.strides, padding=self.padding)
        return DI

    def gv(self, AI, AO, DO):    
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))

        # option  1 
        A = tf.reduce_sum(AI, axis=[1, 2])
        D = tf.reduce_sum(DO, axis=[1, 2])
        DC = tf.matmul(tf.transpose(A), D)
        DC = tf.reshape(DC, (1, 1, self.fin, self.fout))
        DC = DC / (self.h * self.w)

        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # option 2 
        DC = tf.reduce_sum(DF / self.filters, axis=[0, 1])
        DC = tf.reshape(DC, (1, 1, self.fin, self.fout))
        DC = DC / (self.fh * self.fw)

        # option 3 
        # Z = tf.nn.conv2d(X, tf.ones(shape=(self.fh, self.fw, self.fin, self.fout)) * self.connect, self.strides, self.padding)
        # DC = tf.nn.conv2d_backprop_filter(input=C, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)

        # DC = tf.Print(DC, [self.name, tf.keras.backend.std(DC), tf.keras.backend.std(DF)], message='', summarize=1000)
        # DC = tf.Print(DC, [self.name, tf.keras.backend.std(self.connect), tf.reduce_max(self.connect), tf.reduce_min(tf.abs(self.connect))], message='', summarize=1000)

        # return [(DC, self.connect)]
        return [(DF, self.filters), (DB, self.bias)]
        # return [(DC, self.connect), (DF, self.filters), (DB, self.bias)]
    
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
        
        
