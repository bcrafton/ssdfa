
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer

################################################

def conv1d_extract_patches(input_shape, filter_shape, x):
    batch, time, input_size = input_shape
    fw, tmp = filter_shape
    assert(input_size == tmp)
        
    pad = fw // 2
    x = tf.pad(x, [[0, 0], [pad, pad], [0, 0]])
    time = time + 2 * pad
    
    xs = []
    for ii in range(fw):
        start = ii 
        end = ii + time - fw + 1
        next = x[:, start:end, :]
        next = tf.reshape(next, (batch, -1, 1, input_size))
        xs.append(next)
        
    xs = tf.concat(xs, axis=2)
    return xs

###################################################################

def conv1d(input_shape, filter_shape, x, f):
    batch, time, input_size = input_shape
    fw, tmp = filter_shape
    assert(input_size == tmp)
    
    # 50, 64, 5, 64
    patches = conv1d_extract_patches(input_shape=input_shape, filter_shape=filter_shape, x=x)
    _f = tf.reshape(f, (1, 1, fw, input_size))
    out = patches * _f
    out = tf.reduce_sum(out, axis=2)
    return out

################################################

def conv1d_grad(input_shape, filter_shape, x, d):
    batch, time, input_size = input_shape
    fw, tmp = filter_shape
    assert(input_size == tmp)
    
    # so i think u can do extract patches here
    # or we cud conv a with d with padding 2 on each side ...
    patches_x = conv1d_extract_patches(input_shape=input_shape, filter_shape=filter_shape, x=x)
    patches_d = conv1d_extract_patches(input_shape=input_shape, filter_shape=filter_shape, x=d)
    df = patches_x * patches_d
    df = tf.reduce_sum(df, axis=[0, 1]) 
    return df

################################################

class SpikingTimeConv(Layer):

    def __init__(self, input_shape, filter_size, init, activation, alpha=0., name=None, load=None, train=True):

        self.batch, self.times, self.input_size = input_shape
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.filter_shape = [self.filter_size, self.input_size]

        self.alpha = alpha
        self.activation = activation
        self.name = name
        self._train = train
        
        
        filters = np.array([0., 0., 1., 0., 0.])
        # filters = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
        filters = np.reshape(filters, (-1, 1))
        filters = np.repeat(filters, self.input_size, 1)
        '''
        low = -1./self.filter_size
        high = 1./self.filter_size
        filters = np.random.uniform(low=low, high=high, size=self.filter_shape)
        '''
        
        self.filters = tf.Variable(filters, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.filters)]

    def num_params(self):
        filters_size = self.filter_size * self.input_size
        return filters_size

    def forward(self, X):
        Z = conv1d(self.input_shape, self.filter_shape, X, self.filters)
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        rotated_filters = tf.reverse(self.filters, axis=[1])
        DI = conv1d(self.input_shape, self.filter_shape, DO, rotated_filters)
        return DI
        
    def gv(self, AI, AO, DO):
        if not self._train:
            return []
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = conv1d_grad(input_shape=self.input_shape, filter_shape=self.filter_shape, x=AI, d=DO)
        
        # DF = tf.Print(DF, [DF], message=self.name, summarize=1000)

        return [(DF, self.filters)]

    def train(self, AI, AO, DO):
        assert(False)
        return []
        
    ###################################################################

        
        
