
import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

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

def AlexNet224(batch_size, dropout_rate, init='alexnet', sparse=0):
    l1_1 = Convolution(input_shape=[batch_size, 227, 227, 3], filter_sizes=[11, 11, 3, 96], init=init, strides=[1,4,4,1], padding="VALID", bias=0., name='conv1')
    l1_2 = Relu()
    l1_3 = MaxPool(size=[batch_size, 55, 55, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    l1_4 = FeedbackConv(size=[batch_size, 27, 27, 96], num_classes=1000, sparse=sparse, name='conv1_fb')

    l2_1 = Convolution(input_shape=[batch_size, 27, 27, 96], filter_sizes=[5, 5, 96, 256], init=init, bias=1., name='conv2')
    l2_2 = Relu()
    l2_3 = MaxPool(size=[batch_size, 27, 27, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    l2_4 = FeedbackConv(size=[batch_size, 13, 13, 256], num_classes=1000, sparse=sparse, name='conv2_fb')

    l3_1 = Convolution(input_shape=[batch_size, 13, 13, 256], filter_sizes=[3, 3, 256, 384], init=init, bias=0., name='conv3')
    l3_2 = Relu()
    l3_3 = FeedbackConv(size=[batch_size, 13, 13, 384], num_classes=1000, sparse=sparse, name='conv3_fb')

    l4_1 = Convolution(input_shape=[batch_size, 13, 13, 384], filter_sizes=[3, 3, 384, 384], init=init, bias=1., name='conv4')
    l4_2 = Relu()
    l4_3 = FeedbackConv(size=[batch_size, 13, 13, 384], num_classes=1000, sparse=sparse, name='conv4_fb')

    l5_1 = Convolution(input_shape=[batch_size, 13, 13, 384], filter_sizes=[3, 3, 384, 256], init=init, bias=1., name='conv5')
    l5_2 = Relu()
    l5_3 = MaxPool(size=[batch_size, 13, 13, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    l5_4 = FeedbackConv(size=[batch_size, 6, 6, 256], num_classes=1000, sparse=sparse, name='conv5_fb')

    l6 = ConvToFullyConnected(input_shape=[6, 6, 256])

    l7_1 = FullyConnected(input_shape=6*6*256, size=4096, init=init, bias=1., name='fc1')
    l7_2 = Relu()
    l7_3 = Dropout(rate=dropout_rate)
    l7_4 = FeedbackFC(size=[6*6*256, 4096], num_classes=1000, sparse=sparse, name='fc1_fb')

    l8_1 = FullyConnected(input_shape=4096, size=4096, init=init, bias=1., name='fc2')
    l8_2 = Relu()
    l8_3 = Dropout(rate=dropout_rate)
    l8_4 = FeedbackFC(size=[4096, 4096], num_classes=1000, sparse=sparse, name='fc2_fb')

    l9 = FullyConnected(input_shape=4096, size=1000, init=init, bias=1., name='fc3')

    layers = [
    l1_1, l1_2, l1_3, l1_4,
    l2_1, l2_2, l2_3, l2_4,
    l3_1, l3_2, l3_3,
    l4_1, l4_2, l4_3,
    l5_1, l5_2, l5_3, l5_4,
    l6,
    l7_1, l7_2, l7_3, l7_4,
    l8_1, l8_2, l8_3, l8_4,
    l9
    ]

    model = Model(layers=layers)
    return model

