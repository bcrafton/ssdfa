
import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from lib.Model import Model
from lib.FullyConnected import FullyConnected
from lib.AvgPool import AvgPool
from lib.ConvBlock import ConvBlock
from lib.BatchNorm import BatchNorm
from lib.DenseModel import DenseModel
from lib.ConvToFullyConnected import ConvToFullyConnected

def DenseNet64(batch_size, dropout_rate, init='alexnet'):

    # 856 = 6*12 + 12*12 + 24*12 + 24*12 + 64
    # 1920 = 6*32 + 12*32 + 24*32 + 16*32 + 64 --- too big apparently.

    l0 = BatchNorm(input_size=[batch_size, 64, 64, 3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size, 64, 64, 3], filter_shape=[3, 3, 3, 64], strides=[1,1,1,1], init=init, name='conv1')
    l2 = DenseModel(input_shape=[batch_size, 64, 64, 64], init=init, name='dense_model', k=32, L=[6, 12, 24, 16])
    l3 = AvgPool(size=[batch_size, 4, 4, 1920], ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    l4 = ConvToFullyConnected(input_shape=[1, 1, 1920]) 
    l5 = FullyConnected(input_shape=1920, size=1000, init=init, name="fc1")
    layers = [l0, l1, l2, l3, l4, l5]
    model = Model(layers=layers)
    return model

