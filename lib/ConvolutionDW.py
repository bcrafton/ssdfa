
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer
from lib.Activation import Linear

class ConvolutionDW(Layer):

    def __init__(self, input_sizes, filter_sizes, init, strides, padding, alpha=0., activation=None, bias=0., name=None, load=None, train=True):

        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        # ah we copy and pasted the "Convolution" header here and erased this part causing the error with bias
        self.fh, self.fw, self.fin, self.factor = self.filter_sizes
        self.fout = self.fin * self.factor

        bias = np.ones(shape=self.fout) * bias
        
        self.strides = strides
        self.padding = padding
        self.alpha = alpha
        self.activation = Linear() if activation == None else activation
        self.name = name
        self._train = train
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1').item()
            filters = weight_dict[self.name]
            bias = weight_dict[self.name + '_bias']
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
        self.bias = tf.Variable(bias, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.mult
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        Z = tf.nn.depthwise_conv2d(X, self.filters, self.strides, self.padding)
        A = self.activation.act_forward(Z)
        return {'aout':A, 'cache':{}}
        
    def backward(self, AI, AO, DO, cache=None): 
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.depthwise_conv2d_native_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        return {'dout':DI, 'cache':{}}

    def gv(self, AI, AO, DO, cache=None):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.depthwise_conv2d_native_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)

        return [(DF, self.filters)]     

    ###################################################################    

    def lel_backward(self, AI, AO, E, DO, Y, cache):
        return self.backward(AI, AO, DO, cache)

    def lel_gv(self, AI, AO, E, DO, Y, cache):
        return self.gv(AI, AO, DO, cache)

    def lel(self, AI, AO, E, DO, Y):
        return self.train(AI, AO, DO)
    
    ###################################################################
        
