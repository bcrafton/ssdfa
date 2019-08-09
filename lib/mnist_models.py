
import tensorflow as tf
import keras
import numpy as np

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.Activation import Relu
from lib.Activation import Tanh

def mnist_conv(batch_size, dropout_rate, init='alexnet', sparse=0, bias=0.1):
    l0 = Convolution(input_shape=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], init=init, bias=bias, name='conv1')
    l1 = Relu()
    l2 = FeedbackConv(size=[batch_size, 28, 28, 32], num_classes=10, sparse=sparse, name='conv1_fb')

    l3 = Convolution(input_shape=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], init=init, bias=bias, name='conv2')
    l4 = Relu()
    l5 = MaxPool(size=[batch_size, 28, 28, 64], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    l6 = FeedbackConv(size=[batch_size, 14, 14, 64], num_classes=10, sparse=sparse, name='conv2_fb')

    l7 = ConvToFullyConnected(input_shape=[14, 14, 64])

    l8 = FullyConnected(input_shape=14*14*64, size=128, init=init, bias=bias, name='fc1')
    l9 = Relu()
    l10 = Dropout(rate=dropout_rate)
    l11 = FeedbackFC(size=[14*14*64, 128], num_classes=10, sparse=sparse, name='fc1_fb')

    l12 = FullyConnected(input_shape=128, size=10, init=init, bias=bias, name='fc2')

    ##############################################

    layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]
    model = Model(layers=layers)
    return model

def mnist_fc(batch_size, dropout_rate, init='alexnet', sparse=0, bias=0.1):

    l0 = ConvToFullyConnected(input_shape=[28, 28, 1])

    l1 = FullyConnected(input_shape=784, size=400, init=init, bias=bias, name='fc1')
    l2 = Relu()
    l3 = Dropout(rate=dropout_rate)
    l4 = FeedbackFC(size=[784, 400], num_classes=10, sparse=sparse, name='fc1_fb')

    l5 = FullyConnected(input_shape=400, size=10, init=init, bias=bias, name='fc2')

    layers=[l0, l1, l2, l3, l4, l5]
    model = Model(layers=layers)
    return model

