
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

def cifar_conv(batch_size, dropout_rate, init='alexnet', sparse=0, bias=0.1, num_classes=10):
    l0  = Convolution(input_shape=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], init=init, bias=bias, name='conv1')
    l1  = Relu()
    l2  = MaxPool(size=[batch_size, 32, 32, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    l3  = FeedbackConv(size=[batch_size, 16, 16, 96], num_classes=num_classes, sparse=sparse, name='conv1_fb')

    l4  = Convolution(input_shape=[batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], init=init, bias=bias, name='conv2')
    l5  = Relu()
    l6  = MaxPool(size=[batch_size, 16, 16, 128], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    l7  = FeedbackConv(size=[batch_size, 8, 8, 128], num_classes=num_classes, sparse=sparse, name='conv2_fb')

    l8  = Convolution(input_shape=[batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 256], init=init, bias=bias, name='conv3')
    l9  = Relu()
    l10 = MaxPool(size=[batch_size, 8, 8, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    l11 = FeedbackConv(size=[batch_size, 4, 4, 256], num_classes=num_classes, sparse=sparse, name='conv3_fb')

    l12 = ConvToFullyConnected(input_shape=[4, 4, 256])

    l13 = FullyConnected(input_shape=4*4*256, size=2048, init=init, bias=bias, name='fc1')
    l14 = Relu()
    l15 = Dropout(rate=dropout_rate)
    l16 = FeedbackFC(size=[4*4*256, 2048], num_classes=num_classes, sparse=sparse, name='fc1_fb')

    l17 = FullyConnected(input_shape=2048, size=2048, init=init, bias=bias, name='fc2')
    l18 = Relu()
    l19 = Dropout(rate=dropout_rate)
    l20 = FeedbackFC(size=[2048, 2048], num_classes=num_classes, sparse=sparse, name='fc2_fb')

    l21 = FullyConnected(input_shape=2048, size=num_classes, init=init, bias=bias, name='fc3')

    layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21]
    model = Model(layers=layers)
    return model

def cifar_fc(batch_size, dropout_rate, init='alexnet', sparse=0, bias=0.1, num_classes=10):
    l0 = ConvToFullyConnected(input_shape=[32, 32, 3])
    l1 = Dropout(rate=0.1)

    l2 = FullyConnected(input_shape=3072, size=1000, init=init, bias=bias, name='fc1')
    l3 = Dropout(rate=dropout_rate)
    l4 = FeedbackFC(size=[3072, 1000], num_classes=num_classes, sparse=sparse, name='fc1_fb')

    l5 = FullyConnected(input_shape=1000, size=1000, init=init, bias=bias, name='fc2')
    l6 = Dropout(rate=dropout_rate)
    l7 = FeedbackFC(size=[1000, 1000], num_classes=num_classes, sparse=sparse, name='fc2_fb')

    l8 = FullyConnected(input_shape=1000, size=1000, init=init, bias=bias, name='fc3')
    l9 = Dropout(rate=dropout_rate)
    l10 = FeedbackFC(size=[1000, 1000], num_classes=num_classes, sparse=sparse, name='fc3_fb')

    l11 = FullyConnected(input_shape=1000, size=num_classes, init=init, bias=bias, name='fc4')

    model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11])
    return model

