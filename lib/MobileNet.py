
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

def MobileNet224(batch_size, dropout_rate, init='alexnet', sparse=0):

    l0 = BatchNorm(input_size=[batch_size, 224, 224, 3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size, 224, 224, 3], filter_shape=[3, 3, 3, 32], strides=[1,2,2,1], init=init, name='block1')
    # need to add a LELPool() right here.

    l2  = MobileBlock(input_shape=[batch_size, 112, 112, 32], filter_shape=[32, 64],  strides=[1,1,1,1], init=init, name='block2')
    l3  = MobileBlock(input_shape=[batch_size, 112, 112, 64], filter_shape=[64, 128], strides=[1,2,2,1], init=init, name='block3')

    l4  = MobileBlock(input_shape=[batch_size, 56, 56, 128], filter_shape=[128, 128], strides=[1,1,1,1], init=init, name='block4')
    l5  = MobileBlock(input_shape=[batch_size, 56, 56, 128], filter_shape=[128, 256], strides=[1,2,2,1], init=init, name='block5')

    l6  = MobileBlock(input_shape=[batch_size, 28, 28, 256], filter_shape=[256, 256], strides=[1,1,1,1], init=init, name='block6')
    l7  = MobileBlock(input_shape=[batch_size, 28, 28, 256], filter_shape=[256, 512], strides=[1,2,2,1], init=init, name='block7')

    l8  = MobileBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block8')
    l9  = MobileBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block9')
    l10 = MobileBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block10')
    l11 = MobileBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block11')
    l12 = MobileBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block12')

    l13 = MobileBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 1024], strides=[1,2,2,1], init=init, name='block13')

    l14 = MobileBlock(input_shape=[batch_size, 7, 7, 1024], filter_shape=[1024, 1024], strides=[1,1,1,1], init=init, name='block14')

    l15 = AvgPool(size=[batch_size, 7, 7, 1024], ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="SAME")
    l16 = ConvToFullyConnected(input_shape=[1, 1, 1024])
    l17 = FullyConnected(input_shape=1024, size=1000, init=init, name="fc1")

    layers = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17]
    model = Model(layers=layers)

    return model

def MobileNet64(batch_size, dropout_rate, init='alexnet', sparse=0):
    l0 = BatchNorm(input_size=[batch_size, 64, 64, 3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size, 64, 64, 3], filter_shape=[3, 3, 3, 32], strides=[1,1,1,1], init=init, name='block1')

    l2 = MobileBlock(input_shape=[batch_size, 64, 64, 32],  filter_shape=[32, 64],   strides=[1,2,2,1], init=init, name='block2')
    l3 = MobileBlock(input_shape=[batch_size, 32, 32, 64],  filter_shape=[64, 128],  strides=[1,1,1,1], init=init, name='block3')
    l4 = MobileBlock(input_shape=[batch_size, 32, 32, 128], filter_shape=[128, 256], strides=[1,2,2,1], init=init, name='block4')
    l5 = MobileBlock(input_shape=[batch_size, 16, 16, 256], filter_shape=[256, 512], strides=[1,1,1,1], init=init, name='block5')
    l6 = MobileBlock(input_shape=[batch_size, 16, 16, 512], filter_shape=[512, 512], strides=[1,2,2,1], init=init, name='block6')

    l7 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block7')
    l8 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block8')
    l9 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block9')

    l10 = MobileBlock(input_shape=[batch_size, 8, 8, 512],  filter_shape=[512, 1024],  strides=[1,2,2,1], init=init, name='block10')
    l11 = MobileBlock(input_shape=[batch_size, 4, 4, 1024], filter_shape=[1024, 1024], strides=[1,1,1,1], init=init, name='block11')

    l12 = AvgPool(size=[batch_size, 4, 4, 1024], ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
    l13 = ConvToFullyConnected(input_shape=[1, 1, 1024])
    l14 = FullyConnected(input_shape=1024, size=1000, init=init, name="fc1")

    ###############################################################

    layers = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14]
    model = Model(layers=layers)

    return model

