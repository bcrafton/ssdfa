
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid

class Convolution3D(Layer):

    def __init__(self, input_sizes, filter_sizes, init, strides, padding, alpha, activation, bias, name=None, load=None, train=True):

        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fc, self.fg, self.fout = self.filter_sizes
        # self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)
        self.strides = strides
        self.padding = padding
        self.alpha = alpha
        self.activation = activation
        self.name = name
        self._train = train

        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1').item()

            filters = weight_dict[self.name]
            # bias = weight_dict[self.name + '_bias']
        else:
            if init == "zero":
                filters = np.zeros(shape=self.filter_sizes)
            elif init == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
                filters = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.filter_sizes)
            elif init == "alexnet":
                filters = np.random.normal(loc=0.0, scale=0.01, size=self.filter_sizes)
            else:
                # glorot
                assert(False)
                
        self.filters = tf.Variable(filters, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        # return [(self.name, self.filters), (self.name + "_bias", self.bias)]
        return [(self.name, self.filters)]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        # bias_weights_size = self.fout
        return filter_weights_size # + bias_weights_size
                
    def forward(self, X):
        As = []
        for ii in range(self.fg): 
            start = ii * self.fc
            end = (ii + 1) * self.fc
            input_slice = slice(start, end)
 
            Z = tf.nn.conv2d(X[:, :, :, input_slice], self.filters[:, :, :, ii, :], self.strides, self.padding)
            A = self.activation.forward(Z)
            As.append(A)

        A = tf.concat(As, axis=3)
        return A
        
    def backward(self, AI, AO, DO): 
        DO = tf.multiply(DO, self.activation.gradient(AO))

        DIs = []
        for ii in range(self.fg): 
            start = ii * self.fout
            end = (ii + 1) * self.fout
            output_slice = slice(start, end)

            input_sizes = (self.batch_size, self.h, self.w, self.fc)

            DI = tf.nn.conv2d_backprop_input(input_sizes=input_sizes, filter=self.filters[:, :, :, ii, :], out_backprop=DO[:, :, :, output_slice], strides=self.strides, padding=self.padding)
            DIs.append(DI)

        DI = tf.concat(DIs, axis=3)
        return DI

    def gv(self, AI, AO, DO):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))

        DFs = []
        for ii in range(self.fg): 
            start = ii * self.fc
            end = (ii + 1) * self.fc
            input_slice = slice(start, end)

            start = ii * self.fout
            end = (ii + 1) * self.fout
            output_slice = slice(start, end)

            filter_sizes = (self.fh, self.fw, self.fc, self.fout)

            DF = tf.nn.conv2d_backprop_filter(input=AI[:, :, :, input_slice], filter_sizes=filter_sizes, out_backprop=DO[:, :, :, output_slice], strides=self.strides, padding=self.padding)
            DF = tf.reshape(DF, (self.fh, self.fw, self.fc, 1, self.fout))
            DFs.append(DF)

        # if 4 right ?
        # [N fh fw fc fg fout]
        # where N is the length of the list
        DF = tf.concat(DFs, axis=4)
        DF = tf.reshape(DF, (self.fh, self.fw, self.fc, self.fg, self.fout))
        return [(DF, self.filters)]        

    ################################################################### 
        
        
