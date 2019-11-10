
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
    l1_1 = ConvBlock(input_shape=[batch_size, 32, 32, 3],  filter_shape=[3, 3, 3, 64],  strides=[1,1,1,1], init=init, name='block1')
    l1_2 = ConvBlock(input_shape=[batch_size, 32, 32, 64], filter_shape=[3, 3, 64, 64], strides=[1,1,1,1], init=init, name='block2')
    l1_3 = AvgPool(size=[batch_size, 32, 32, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = ConvBlock(input_shape=[batch_size, 16, 16, 64],  filter_shape=[3, 3, 64, 128],  strides=[1,1,1,1], init=init, name='block3')
    l2_2 = ConvBlock(input_shape=[batch_size, 16, 16, 128], filter_shape=[3, 3, 128, 128], strides=[1,1,1,1], init=init, name='block4')
    l2_3 = AvgPool(size=[batch_size, 16, 16, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l3_1 = ConvBlock(input_shape=[batch_size, 8, 8, 128],  filter_shape=[3, 3, 128, 256], strides=[1,1,1,1], init=init, name='block5')
    l3_2 = ConvBlock(input_shape=[batch_size, 8, 8, 256],  filter_shape=[3, 3, 256, 256], strides=[1,1,1,1], init=init, name='block6')
    l3_3 = AvgPool(size=[batch_size, 8, 8, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l4_1 = ConvBlock(input_shape=[batch_size, 4, 4, 256],  filter_shape=[3, 3, 256, 512], strides=[1,1,1,1], init=init, name='block7')
    l4_2 = ConvBlock(input_shape=[batch_size, 4, 4, 512],  filter_shape=[3, 3, 512, 512], strides=[1,1,1,1], init=init, name='block8')
    l4_3 = AvgPool(size=[batch_size, 4, 4, 512], ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    l5_1 = ConvToFullyConnected(input_shape=[batch_size, 1, 1, 512])
    l5_2 = FullyConnected(input_shape=512, size=num_classes, init=init, bias=bias, name='fc1')

    layers=[
    l1_1, l1_2, l1_3,
    l2_1, l2_2, l2_3,
    l3_1, l3_2, l3_3,
    l4_1, l4_2, l4_3,
    l5_1, l5_2
    ]
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

