
import tensorflow as tf
import keras
import numpy as np

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.Activation import Relu
from lib.Activation import Tanh

from lib.ConvBlock import ConvBlock
from lib.VGGBlock import VGGBlock
from lib.MobileBlock import MobileBlock
from lib.BatchNorm import BatchNorm

def cifar_conv(batch_size, dropout_rate, init='alexnet', sparse=0, bias=0.1, num_classes=10):
    l1 = ConvBlock(input_shape=[batch_size, 32, 32, 6],  filter_shape=[5, 5, 6,   96],  strides=[1,1,1,1], init=init, name='block1')
    l2 = AvgPool(size=[batch_size, 32, 32, 96], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l3 = ConvBlock(input_shape=[batch_size, 16, 16, 96], filter_shape=[5, 5, 96,  128], strides=[1,1,1,1], init=init, name='block1')
    l4 = AvgPool(size=[batch_size, 16, 16, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l5 = ConvBlock(input_shape=[batch_size, 8, 8, 128],  filter_shape=[5, 5, 128, 256], strides=[1,1,1,1], init=init, name='block1')
    l6 = AvgPool(size=[batch_size, 8, 8, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l7 = ConvToFullyConnected(input_shape=[4, 4, 256])
    l8 = FullyConnected(input_shape=4*4*256, size=num_classes, init=init, bias=bias, name='fc1')

    layers=[l1, l2, l3, l4, l5, l6, l7, l8]
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

