
import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from lib.Model import Model
from lib.FullyConnected import FullyConnected
from lib.AvgPool import AvgPool
from lib.ConvBlock import ConvBlock
from lib.VGGBlock import VGGBlock
from lib.MobileBlock import MobileBlock
from lib.BatchNorm import BatchNorm

def DenseNet64(batch_size, dropout_rate, init='alexnet', sparse=0):

    l0 = BatchNorm(input_size=[batch_size, 64, 64, 3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size, 64, 64, 3], filter_shape=[3, 3, 3, 64], strides=[1,1,1,1], init=init, name='conv1')
    l2 = DenseModel(input_shape=[batch_size, 64, 64, 64], init=init, name='dense_model', k=12, Ls=[6, 12, 24, 24])
    l3 = ConvToFullyConnected(input_shape=[1, 1, 1024])
    l4 = FullyConnected(input_shape=1024, size=1000, init=init, name="fc1")

    layers = [l0, l1, l2, l3, l4]
    model = Model(layers=layers)
    return model

