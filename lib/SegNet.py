
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
from lib.DecodeBlock import DecodeBlock

'''
def SegNet(batch_size, init='alexnet'):

    ###########################################################################################

    l0 = BatchNorm(input_size=[batch_size, 224, 224, 3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size, 224, 224, 3], filter_shape=[3, 3, 3, 32], strides=[1,2,2,1], init=init, name='block1')

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

    ###########################################################################################

    l15 = DecodeBlock(input_shape=[batch_size, 7, 7, 1024], filter_shape=[1024, 1024], ksize=1, init=init, name='block15')
    l16 = DecodeBlock(input_shape=[batch_size, 7, 7, 1024], filter_shape=[1024, 512],  ksize=2, init=init, name='block16')

    l17 = DecodeBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 512], ksize=1, init=init, name='block17')
    l18 = DecodeBlock(input_shape=[batch_size, 14, 14, 512], filter_shape=[512, 256], ksize=2, init=init, name='block18')

    l19 = DecodeBlock(input_shape=[batch_size, 28, 28, 256], filter_shape=[256, 256], ksize=1, init=init, name='block19')
    l20 = DecodeBlock(input_shape=[batch_size, 28, 28, 256], filter_shape=[256, 128], ksize=2, init=init, name='block20')

    l21 = DecodeBlock(input_shape=[batch_size, 56, 56, 128], filter_shape=[128, 128], ksize=1, init=init, name='block21')
    l22 = DecodeBlock(input_shape=[batch_size, 56, 56, 128], filter_shape=[128, 64],  ksize=2, init=init, name='block22')

    l23 = DecodeBlock(input_shape=[batch_size, 112, 112, 64], filter_shape=[64, 64], ksize=1, init=init, name='block23')
    l24 = DecodeBlock(input_shape=[batch_size, 112, 112, 64], filter_shape=[64, 64], ksize=2, init=init, name='block24')

    l25 = ConvBlock(input_shape=[batch_size, 224, 224, 64], filter_shape=[3, 3, 64, 30], strides=[1,1,1,1], init=init, name='block25')

    ###########################################################################################

    layers = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25]
    model = Model(layers=layers)
    return model

    ###########################################################################################
'''

def SegNet(batch_size, init='alexnet', load=None):

    ###########################################################################################

    l0 = BatchNorm(input_size=[batch_size, 480, 480, 3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size, 480, 480, 3], filter_shape=[3, 3, 3, 32], strides=[1,2,2,1], init=init, name='block1', load=load, train=False)

    l2  = MobileBlock(input_shape=[batch_size, 240, 240, 32], filter_shape=[32, 64],  strides=[1,1,1,1], init=init, name='block2', load=load, train=False)
    l3  = MobileBlock(input_shape=[batch_size, 240, 240, 64], filter_shape=[64, 128], strides=[1,2,2,1], init=init, name='block3', load=load, train=False)

    l4  = MobileBlock(input_shape=[batch_size, 120, 120, 128], filter_shape=[128, 128], strides=[1,1,1,1], init=init, name='block4', load=load, train=False)
    l5  = MobileBlock(input_shape=[batch_size, 120, 120, 128], filter_shape=[128, 256], strides=[1,2,2,1], init=init, name='block5', load=load, train=False)

    l6  = MobileBlock(input_shape=[batch_size, 60, 60, 256], filter_shape=[256, 256], strides=[1,1,1,1], init=init, name='block6', load=load, train=False)
    l7  = MobileBlock(input_shape=[batch_size, 60, 60, 256], filter_shape=[256, 512], strides=[1,2,2,1], init=init, name='block7', load=load, train=False)

    l8  = MobileBlock(input_shape=[batch_size, 30, 30, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block8', load=load, train=False)
    l9  = MobileBlock(input_shape=[batch_size, 30, 30, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block9', load=load, train=False)
    l10 = MobileBlock(input_shape=[batch_size, 30, 30, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block10', load=load, train=False)
    l11 = MobileBlock(input_shape=[batch_size, 30, 30, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block11', load=load, train=False)
    l12 = MobileBlock(input_shape=[batch_size, 30, 30, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=init, name='block12', load=load, train=False)

    l13 = MobileBlock(input_shape=[batch_size, 30, 30, 512],  filter_shape=[512, 1024], strides=[1,2,2,1], init=init, name='block13', load=load, train=False)
    l14 = MobileBlock(input_shape=[batch_size, 15, 15, 1024], filter_shape=[1024, 1024], strides=[1,1,1,1], init=init, name='block14', load=load, train=False)

    ###########################################################################################

    l15 = DecodeBlock(input_shape=[batch_size, 15, 15, 1024], filter_shape=[1024, 1024], ksize=1, init=init, name='block15')
    l16 = DecodeBlock(input_shape=[batch_size, 15, 15, 1024], filter_shape=[1024, 512],  ksize=2, init=init, name='block16')

    l17 = DecodeBlock(input_shape=[batch_size, 30, 30, 512], filter_shape=[512, 512], ksize=1, init=init, name='block17')
    l18 = DecodeBlock(input_shape=[batch_size, 30, 30, 512], filter_shape=[512, 256], ksize=2, init=init, name='block18')

    l19 = DecodeBlock(input_shape=[batch_size, 60, 60, 256], filter_shape=[256, 256], ksize=1, init=init, name='block19')
    l20 = DecodeBlock(input_shape=[batch_size, 60, 60, 256], filter_shape=[256, 128], ksize=2, init=init, name='block20')

    l21 = DecodeBlock(input_shape=[batch_size, 120, 120, 128], filter_shape=[128, 128], ksize=1, init=init, name='block21')
    l22 = DecodeBlock(input_shape=[batch_size, 120, 120, 128], filter_shape=[128, 64],  ksize=2, init=init, name='block22')

    l23 = DecodeBlock(input_shape=[batch_size, 240, 240, 64], filter_shape=[64, 64], ksize=1, init=init, name='block23')
    l24 = DecodeBlock(input_shape=[batch_size, 240, 240, 64], filter_shape=[64, 64], ksize=2, init=init, name='block24')

    l25 = ConvBlock(input_shape=[batch_size, 480, 480, 64], filter_shape=[3, 3, 64, 30], strides=[1,1,1,1], init=init, name='block25')

    ###########################################################################################

    layers = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25]
    model = Model(layers=layers)
    return model

    ###########################################################################################


