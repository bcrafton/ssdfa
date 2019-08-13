
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
from lib.AvgPool import AvgPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv
from lib.Activation import Relu

from lib.ConvBlock import ConvBlock
from lib.VGGBlock import VGGBlock
from lib.MobileBlock import MobileBlock
from lib.BatchNorm import BatchNorm

def VGGNet224(batch_size, dropout_rate, init='alexnet', sparse=0):
    l0 = BatchNorm(input_size=[batch_size, 224, 224, 3], name='bn0')

    l1_1 = VGGBlock(input_shape=[batch_size, 224, 224, 3],  filter_shape=[3, 64], init=init, name='block1')
    l1_2 = VGGBlock(input_shape=[batch_size, 224, 224, 64], filter_shape=[64, 64], init=init, name='block2')
    l1_3 = AvgPool(size=[batch_size, 224, 224, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = VGGBlock(input_shape=[batch_size, 112, 112, 64],  filter_shape=[64, 128], init=init, name='block3')
    l2_2 = VGGBlock(input_shape=[batch_size, 112, 112, 128], filter_shape=[128, 128], init=init, name='block4')
    l2_3 = AvgPool(size=[batch_size, 112, 112, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l3_1 = VGGBlock(input_shape=[batch_size, 56, 56, 128], filter_shape=[128, 256], init=init, name='block5')
    l3_2 = VGGBlock(input_shape=[batch_size, 56, 56, 256], filter_shape=[256, 256], init=init, name='block6')
    l3_3 = AvgPool(size=[batch_size, 56, 56, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l4_1 = VGGBlock(input_shape=[batch_size, 28, 28, 256], filter_shape=[256, 512], init=init, name='block7')
    l4_2 = VGGBlock(input_shape=[batch_size, 28, 28, 512], filter_shape=[512, 512], init=init, name='block8')
    l4_3 = AvgPool(size=[batch_size, 28, 28, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l5_1 = VGGBlock(input_shape=[batch_size, 14, 14, 512],  filter_shape=[512, 1024],  init=init, name='block9')
    l5_2 = VGGBlock(input_shape=[batch_size, 14, 14, 1024], filter_shape=[1024, 1024], init=init, name='block10')
    l5_3 = AvgPool(size=[batch_size, 14, 14, 1024], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l6_1 = VGGBlock(input_shape=[batch_size, 7, 7, 1024], filter_shape=[1024, 1024],  init=init, name='block11')
    l6_2 = VGGBlock(input_shape=[batch_size, 7, 7, 1024], filter_shape=[1024, 1024], init=init, name='block12')
    l6_3 = AvgPool(size=[batch_size, 1, 1, 1024], ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="SAME")

    l7 = ConvToFullyConnected(input_shape=[1, 1, 1024])
    l8 = FullyConnected(input_shape=1024, size=1000, init=init, name="fc1")

    ###############################################################

    layers = [
    l0,
    l1_1, l1_2, l1_3,
    l2_1, l2_2, l2_3,
    l3_1, l3_2, l3_3,
    l4_1, l4_2, l4_3,
    l5_1, l5_2, l5_3,
    l6_1, l6_2, l6_3,
    l7, 
    l8
    ]
    model = Model(layers=layers)

    return model

'''
def VGGNet224(batch_size, dropout_rate, init='alexnet', sparse=0):
    l1_1 = Convolution(input_shape=[batch_size, 224, 224, 3],  filter_sizes=[3, 3, 3, 64],  init=init, padding="SAME", name='conv1')
    l1_2 = Relu()
    l1_3 = Convolution(input_shape=[batch_size, 224, 224, 64], filter_sizes=[3, 3, 64, 64], init=init, padding="SAME", name='conv2')
    l1_4 = Relu()
    l1_5 = MaxPool(size=[batch_size, 224, 224, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = Convolution(input_shape=[batch_size, 112, 112, 64], filter_sizes=[3, 3, 64, 128], init=init, padding="SAME", name='conv3')
    l2_2 = Relu()
    l2_3 = Convolution(input_shape=[batch_size, 112, 112, 128], filter_sizes=[3, 3, 128, 128], init=init, padding="SAME", name='conv4')
    l2_4 = Relu()
    l2_5 = MaxPool(size=[batch_size, 112, 112, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l3_1 = Convolution(input_shape=[batch_size, 56, 56, 128], filter_sizes=[3, 3, 128, 256], init=init, padding="SAME", name='conv5')
    l3_2 = Relu()
    l3_3 = Convolution(input_shape=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], init=init, padding="SAME", name='conv6')
    l3_4 = Relu()
    l3_5 = Convolution(input_shape=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], init=init, padding="SAME", name='conv7')
    l3_6 = Relu()
    l3_7 = MaxPool(size=[batch_size, 56, 56, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l4_1 = Convolution(input_shape=[batch_size, 28, 28, 256], filter_sizes=[3, 3, 256, 512], init=init, padding="SAME", name='conv8')
    l4_2 = Relu()
    l4_3 = Convolution(input_shape=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], init=init, padding="SAME", name='conv9')
    l4_4 = Relu()
    l4_5 = Convolution(input_shape=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], init=init, padding="SAME", name='conv10')
    l4_6 = Relu()
    l4_7 = MaxPool(size=[batch_size, 28, 28, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l5_1 = Convolution(input_shape=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], init=init, padding="SAME", name='conv11')
    l5_2 = Relu()
    l5_3 = Convolution(input_shape=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], init=init, padding="SAME", name='conv12')
    l5_4 = Relu()
    l5_5 = Convolution(input_shape=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], init=init, padding="SAME", name='conv13')
    l5_6 = Relu()
    l5_7 = MaxPool(size=[batch_size, 14, 14, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l6 = ConvToFullyConnected(input_shape=[7, 7, 512])

    l7_1 = FullyConnected(input_shape=7*7*512, size=4096, init=init, name='fc1')
    l7_2 = Relu()
    l7_3 = Dropout(rate=dropout_rate)
    l7_4 = FeedbackFC(size=[7*7*512, 4096], num_classes=1000, sparse=sparse, name='fc1_fb')

    l8_1 = FullyConnected(input_shape=4096, size=4096, init=init, name='fc2')
    l8_2 = Relu()
    l8_3 = Dropout(rate=dropout_rate)
    l8_4 = FeedbackFC(size=[4096, 4096], num_classes=1000, sparse=sparse, name='fc2_fb')

    l9 = FullyConnected(input_shape=4096, size=1000, init=init, name='fc3')

    ###############################################################

    layers = [
    l1_1, l1_2, l1_3, l1_4, l1_5, 
    l2_1, l2_2, l2_3, l2_4, l2_5, 
    l3_1, l3_2, l3_3, l3_4, l3_5, l3_6, l3_7, 
    l4_1, l4_2, l4_3, l4_4, l4_5, l4_6, l4_7, 
    l5_1, l5_2, l5_3, l5_4, l5_5, l5_6, l5_7, 
    l6,
    l7_1, l7_2, l7_3, l7_4, 
    l8_1, l8_2, l8_3, l8_4, 
    l9
    ]
    model = Model(layers=layers)

    return model
'''

def VGGNet64(batch_size, dropout_rate, init='alexnet', sparse=0):
    l0 = BatchNorm(input_size=[batch_size, 64, 64, 3], name='bn0')

    l1_1 = VGGBlock(input_shape=[batch_size, 64, 64, 3], filter_shape=[3, 64], init=init, name='block1')
    l1_2 = VGGBlock(input_shape=[batch_size, 64, 64, 64], filter_shape=[64, 64], init=init, name='block2')
    l1_3 = AvgPool(size=[batch_size, 64, 64, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l2_1 = VGGBlock(input_shape=[batch_size, 32, 32, 64],  filter_shape=[64, 128], init=init, name='block3')
    l2_2 = VGGBlock(input_shape=[batch_size, 32, 32, 128], filter_shape=[128, 128], init=init, name='block4')
    l2_3 = AvgPool(size=[batch_size, 32, 32, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l3_1 = VGGBlock(input_shape=[batch_size, 16, 16, 128], filter_shape=[128, 256], init=init, name='block5')
    l3_2 = VGGBlock(input_shape=[batch_size, 16, 16, 256], filter_shape=[256, 256], init=init, name='block6')
    l3_3 = AvgPool(size=[batch_size, 16, 16, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l4_1 = VGGBlock(input_shape=[batch_size, 8, 8, 256],   filter_shape=[256, 512], init=init, name='block7')
    l4_2 = VGGBlock(input_shape=[batch_size, 8, 8, 512],   filter_shape=[512, 512], init=init, name='block8')
    l4_3 = AvgPool(size=[batch_size, 8, 8, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    l5_1 = VGGBlock(input_shape=[batch_size, 4, 4, 512],   filter_shape=[512, 1024],  init=init, name='block9')
    l5_2 = VGGBlock(input_shape=[batch_size, 4, 4, 1024],  filter_shape=[1024, 1024], init=init, name='block10')
    l5_3 = AvgPool(size=[batch_size, 4, 4, 1024], ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    l6 = ConvToFullyConnected(input_shape=[1, 1, 1024])
    l7 = FullyConnected(input_shape=1024, size=1000, init=init, name="fc1")

    ###############################################################

    layers = [
    l0,
    l1_1, l1_2, l1_3,
    l2_1, l2_2, l2_3,
    l3_1, l3_2, l3_3,
    l4_1, l4_2, l4_3,
    l5_1, l5_2, l5_3,
    l6, 
    l7
    ]
    model = Model(layers=layers)

    return model

