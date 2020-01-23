
import tensorflow as tf
import keras
import numpy as np

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.Activation import Relu

from lib.ConvRelu import ConvRelu

'''
def cifar_conv(batch_size, dropout_rate, init='glorot_uniform', sparse=0, bias=0, num_classes=10):
    l1 = ConvRelu(input_shape=[batch_size,32,32,3], filter_shape=[4,4,3,32], strides=[1,1,1,1], init=init, name='conv1')
    
    l2 = ConvRelu(input_shape=[batch_size,32,32,32],  filter_shape=[2,2,32,128], strides=[1,1,1,1], init=init, name='conv1')
    l3 = ConvRelu(input_shape=[batch_size,32,32,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv1')
    l4 = ConvRelu(input_shape=[batch_size,32,32,32],  filter_shape=[4,4,32,32],  strides=[1,2,2,1], init=init, name='conv1')
    
    l5 = ConvRelu(input_shape=[batch_size,16,16,32],  filter_shape=[2,2,32,128], strides=[1,1,1,1], init=init, name='conv1')
    l6 = ConvRelu(input_shape=[batch_size,16,16,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv1')
    l7 = ConvRelu(input_shape=[batch_size,16,16,32],  filter_shape=[4,4,32,32],  strides=[1,2,2,1], init=init, name='conv1')
    
    l8 = ConvRelu(input_shape=[batch_size,8,8,32],  filter_shape=[2,2,32,128], strides=[1,1,1,1], init=init, name='conv1')
    l9 = ConvRelu(input_shape=[batch_size,8,8,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv1')
    l10 = ConvRelu(input_shape=[batch_size,8,8,32],  filter_shape=[4,4,32,32],  strides=[1,2,2,1], init=init, name='conv1')
    
    l11 = ConvToFullyConnected(input_shape=[batch_size,4,4,32])
    l12 = FullyConnected(input_shape=512, size=10, init=init, bias=bias, name='fc4')

    layers=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12]
    model = Model(layers=layers)

    return model
'''

'''
def cifar_conv(batch_size, dropout_rate, init='glorot_uniform', sparse=0, bias=0, num_classes=10):
    l1 = ConvRelu(input_shape=[batch_size,32,32,3],  filter_shape=[4,4,3,32],  strides=[1,1,1,1], init=init, name='conv1')
    l2 = ConvRelu(input_shape=[batch_size,32,32,32], filter_shape=[4,4,32,64], strides=[1,2,2,1], init=init, name='conv1')
    l3 = ConvRelu(input_shape=[batch_size,16,16,64], filter_shape=[4,4,64,64], strides=[1,2,2,1], init=init, name='conv1')
    l4 = ConvRelu(input_shape=[batch_size,8,8,64], filter_shape=[4,4,64,32], strides=[1,2,2,1], init=init, name='conv1')
    
    l5 = ConvToFullyConnected(input_shape=[batch_size,4,4,32])
    l6 = FullyConnected(input_shape=512, size=10, init=init, bias=bias, name='fc4')

    layers=[l1,l2,l3,l4,l5,l6]
    model = Model(layers=layers)

    return model
'''

'''
def cifar_conv(batch_size, dropout_rate, init='glorot_uniform', sparse=0, bias=0, num_classes=10):
    l1 = ConvRelu(input_shape=[batch_size,32,32,3], filter_shape=[4,4,3,32], strides=[1,2,2,1], init=init, name='conv1')
    
    l2 = ConvRelu(input_shape=[batch_size,16,16,32],  filter_shape=[2,2,32,128], strides=[1,1,1,1], init=init, name='conv1')
    l3 = ConvRelu(input_shape=[batch_size,16,16,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv1')
    l4 = ConvRelu(input_shape=[batch_size,16,16,32],  filter_shape=[4,4,32,32],  strides=[1,2,2,1], init=init, name='conv1')
    
    l5 = ConvRelu(input_shape=[batch_size,8,8,32],  filter_shape=[2,2,32,128], strides=[1,1,1,1], init=init, name='conv1')
    l6 = ConvRelu(input_shape=[batch_size,8,8,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv1')
    l7 = ConvRelu(input_shape=[batch_size,8,8,32],  filter_shape=[4,4,32,32],  strides=[1,2,2,1], init=init, name='conv1')
    
    l8 = ConvToFullyConnected(input_shape=[batch_size,4,4,32])
    l9 = FullyConnected(input_shape=512, size=10, init=init, bias=bias, name='fc4')

    layers=[l1,l2,l3,l4,l5,l6,l7,l8,l9]
    model = Model(layers=layers)

    return model
'''

def cifar_conv(batch_size, scale, init='glorot_uniform'):
    l1 = ConvRelu(input_shape=[batch_size,32,32,3], filter_shape=[4,4,3,32], strides=[1,1,1,1], init=init, name='conv1', scale=scale[0])
    
    l2 = ConvRelu(input_shape=[batch_size,32,32,32],  filter_shape=[4,4,32,128], strides=[1,2,2,1], init=init, name='conv2', scale=scale[1])
    l3 = ConvRelu(input_shape=[batch_size,16,16,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv3', scale=scale[2])
    
    l4 = ConvRelu(input_shape=[batch_size,16,16,32],  filter_shape=[4,4,32,128], strides=[1,2,2,1], init=init, name='conv4', scale=scale[3])
    l5 = ConvRelu(input_shape=[batch_size,8,8,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv5', scale=scale[4])

    l6 = ConvRelu(input_shape=[batch_size,8,8,32],  filter_shape=[4,4,32,128], strides=[1,2,2,1], init=init, name='conv6', scale=scale[5])
    l7 = ConvRelu(input_shape=[batch_size,4,4,128], filter_shape=[1,1,128,32], strides=[1,1,1,1], init=init, name='conv7', scale=scale[6])

    l8 = ConvToFullyConnected(input_shape=[batch_size,4,4,32])
    l9 = FullyConnected(input_shape=512, size=10, init=init, bias=0, use_bias=True, name='dense8', scale=scale[7])

    layers=[l1,l2,l3,l4,l5,l6,l7,l8,l9]
    model = Model(layers=layers)

    return model

######################################################################

def cifar_conv_bn(batch_size, dropout_rate, init='glorot_uniform', sparse=0, bias=0, num_classes=10):
    l1 = Convolution(input_shape=[batch_size, 32, 32, 3],  filter_sizes=[4, 4, 3,  64], strides=[1,2,2,1], init=init, name='conv1')
    l2 = BatchNorm(input_size=[batch_size, 16, 16, 64], name='bn1')
    l3 = Relu()
    
    l4 = Convolution(input_shape=[batch_size, 16, 16, 64], filter_sizes=[4, 4, 64, 64], strides=[1,2,2,1], init=init, name='conv2')
    l5 = BatchNorm(input_size=[batch_size, 8, 8, 64], name='bn2')
    l6 = Relu()
    
    l7 = Convolution(input_shape=[batch_size,  8,  8, 64], filter_sizes=[4, 4, 64, 32], strides=[1,2,2,1], init=init, name='conv3')
    l8 = BatchNorm(input_size=[batch_size, 4, 4, 32], name='bn3')
    l9 = Relu()

    l10 = ConvToFullyConnected(input_shape=[batch_size, 4, 4, 32])
    l11 = FullyConnected(input_shape=512, size=10, init=init, bias=bias, name='fc4')

    layers=[l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]
    model = Model(layers=layers)

    return model


