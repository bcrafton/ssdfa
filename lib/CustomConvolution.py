
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length
from lib.conv_utils import get_pad

class Convolution(Layer):

    def __init__(self, input_sizes, filter_sizes, init, strides, padding, alpha, activation, bias, name=None, load=None, train=True, custom=True):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.strides = strides
        self.padding = padding
        
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        _, self.stride_row, self.stride_col, _ = self.strides
        self.pad_h, self.pad_w = get_pad(self.padding.lower(), np.array([self.fh, self.fw]))

        self.alpha = alpha
        self.activation = activation
        self.name = name
        self._train = train
        self.custom = custom
        
        self.output_row = conv_output_length(self.h, self.fh, self.padding.lower(), self.strides[0])
        self.output_col = conv_output_length(self.w, self.fw, self.padding.lower(), self.strides[1])
        self.output_shape = (self.output_row, self.output_col, self.fout)
        # self.filter_shape = (self.output_row * self.output_col, self.fh * self.fw * self.fin, self.fout)
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load, encoding='latin1').item()
            filters = weight_dict[self.name]
            bias = weight_dict[self.name + '_bias']
        else:
            assert(False)

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

            bias = np.ones(shape=self.fout) * bias
            
        self.filters = tf.Variable(filters, dtype=tf.float32)
        self.bias = tf.Variable(bias, dtype=tf.float32)
            
    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size

    ###################################################################

    def forward1(self, X):
        N = tf.shape(X)[0]

        X = tf.pad(X, [[0, 0], [self.pad_h, self.pad_h], [self.pad_w, self.pad_w], [0, 0]])
        xs = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i * self.stride_row, i * self.stride_row + self.fh)
                slice_col = slice(j * self.stride_col, j * self.stride_col + self.fw)
                xs.append(tf.reshape(X[:, slice_row, slice_col, :], (N, 1, self.fh * self.fw * self.fin)))

        x_aggregate = tf.concat(xs, axis=1)
        x_aggregate = tf.reshape(x_aggregate, (N * self.output_row * self.output_col, self.fh * self.fw * self.fin))

        filters = self.filters
        filters = tf.reshape(filters, (self.fh * self.fw * self.fin, self.fout))        

        Z = tf.matmul(x_aggregate, filters)
        Z = tf.reshape(Z, (N, self.output_row, self.output_col, self.fout))
        
        A = self.activation.forward(Z)
        return A

    def forward2(self, X):
        Z = tf.add(tf.nn.conv2d(X, self.filters, self.strides, self.padding), tf.reshape(self.bias, [1, 1, self.fout]))
        A = self.activation.forward(Z)
        return A

    def forward(self, X):
        if self.custom:
            return self.forward1(X)
        else:
            return self.forward2(X)

    ###################################################################           

    # in either (backward, gv) we have to account for that 90 degree rotation.
    def backward1(self, AI, AO, DO):
        N = tf.shape(AI)[0]

        DO = tf.multiply(DO, self.activation.gradient(AO))
        [pad_w, pad_h] = get_pad('full', np.array([self.fh, self.fw]))
        DO = tf.pad(DO, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        es = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i, i + self.fh)
                slice_col = slice(j, j + self.fw)
                es.append(tf.reshape(DO[:, slice_row, slice_col, :], (1, N, self.fh * self.fw * self.fout)))
        
        DO = tf.concat(es, axis=0)
        DO = tf.reshape(DO, (self.output_row * self.output_col * N, self.fh * self.fw * self.fout))
        
        filters = self.filters
        '''
        filters = tf.reshape(filters, (self.fh, self.fw, self.fin * self.fout))
        filters = tf.image.rot90(filters, k=2)
        filters = tf.reshape(filters, (self.fh, self.fw, self.fin, self.fout))
        '''
        filters = tf.reverse(filters, [0, 1])

        filters = tf.transpose(filters, (0, 1, 3, 2))
        filters = tf.reshape(filters, (self.fh * self.fw * self.fout, self.fin))
        
        DI = tf.matmul(DO, filters)
        DI = tf.reshape(DI, (self.output_row, self.output_col, N, self.fin))
        DI = tf.transpose(DI, (2, 0, 1, 3))
        
        return DI

    def backward2(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
        return DI

    def backward(self, AI, AO, DO):
        
        # return self.backward2(AI, AO, DO)

        if self.custom:
            return self.backward1(AI, AO, DO)
        else:
            return self.backward2(AI, AO, DO)

    ###################################################################

    def gv1(self, AI, AO, DO): 
        if not self._train:
            return []

        N = tf.shape(AI)[0]

        AI = tf.pad(AI, [[0, 0], [self.pad_h, self.pad_h], [self.pad_w, self.pad_w], [0, 0]])
        xs = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i * self.stride_row, i * self.stride_row + self.fh)
                slice_col = slice(j * self.stride_col, j * self.stride_col + self.fw)
                xs.append(tf.reshape(AI[:, slice_row, slice_col, :], (N, 1, self.fh * self.fw * self.fin)))

        x_aggregate = tf.concat(xs, axis=1)
        x_aggregate = tf.reshape(x_aggregate, (N * self.output_row * self.output_col, self.fh * self.fw * self.fin))
        x_aggregate = tf.transpose(x_aggregate)

        DO = tf.multiply(DO, self.activation.gradient(AO))

        # need to do this before we mess with DO
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        DO = tf.reshape(DO, (N * self.output_row * self.output_col, self.fout))
        DF = tf.matmul(x_aggregate, DO)
        DF = tf.reshape(DF, (self.fh, self.fw, self.fin, self.fout))

        return [(DF, self.filters), (DB, self.bias)]

    def gv2(self, AI, AO, DO):    
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
    
    def gv(self, AI, AO, DO):
        
        # return self.gv2(AI, AO, DO)

        if self.custom:
            return self.gv1(AI, AO, DO)
        else:
            return self.gv2(AI, AO, DO)

    ###################################################################

    def train1(self, AI, AO, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]

        AI = tf.pad(AI, [[0, 0], [self.pad_h, self.pad_h], [self.pad_w, self.pad_w], [0, 0]])
        xs = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i * self.stride_row, i * self.stride_row + self.fh)
                slice_col = slice(j * self.stride_col, j * self.stride_col + self.fw)
                xs.append(tf.reshape(AI[:, slice_row, slice_col, :], (N, 1, self.fh * self.fw * self.fin)))

        x_aggregate = tf.concat(xs, axis=1)
        x_aggregate = tf.reshape(x_aggregate, (N * self.output_row * self.output_col, self.fh * self.fw * self.fin))
        x_aggregate = tf.transpose(x_aggregate)

        DO = tf.multiply(DO, self.activation.gradient(AO))

        # need to do this before we mess with DO
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        DO = tf.reshape(DO, (N * self.output_row * self.output_col, self.fout))
        DF = tf.matmul(x_aggregate, DO)
        DF = tf.reshape(DF, (self.fh, self.fw, self.fin, self.fout))

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]

    def train2(self, AI, AO, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]

    def train(self, AI, AO, DO):

        # return self.train2(AI, AO, DO)

        if self.custom:
            return self.train1(AI, AO, DO)
        else:
            return self.train2(AI, AO, DO)


