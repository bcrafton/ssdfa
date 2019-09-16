
import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from lib.Model import Model
from lib.FullyConnected import FullyConnected
from lib.AvgPool import AvgPool
from lib.MaxPool import MaxPool
from lib.ConvBlock import ConvBlock
from lib.BatchNorm import BatchNorm
from lib.DenseModel import DenseModel
from lib.ConvToFullyConnected import ConvToFullyConnected

def DenseNet64(batch_size, dropout_rate, init='alexnet'):

    # 856 = 6*12 + 12*12 + 24*12 + 24*12 + 64
    # 1920 = 6*32 + 12*32 + 24*32 + 16*32 + 64 --- too big apparently.
    # 992 = 6*16 + 12*16 + 24*16 + 16*16 + 64

    k = 32
    L = [4, 6, 12, 24, 12]
    F = 32
    size = k * sum(L) + F

    l0 = BatchNorm(input_size=[batch_size,64,64,3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size,64,64,3], filter_shape=[3,3,3,F], strides=[1,1,1,1], init=init, name='conv1')

    l2 = DenseModel(input_shape=[batch_size,64,64,F], init=init, name='dense_model', k=k, L=L)
    l3 = AvgPool(size=[batch_size,4,4,size], ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

    l4 = ConvToFullyConnected(input_shape=[batch_size,1,1,size]) 
    l5 = FullyConnected(input_shape=size, size=1000, init=init, name="fc1")

    model = Model(layers=[l0, l1, l2, l3, l4, l5])
    return model

def DenseNet224(batch_size, dropout_rate, init='alexnet'):

    k = 32
    L = [6, 12, 24, 16]
    F = 64
    size = k * sum(L) + F

    # l0 = BatchNorm(input_size=[batch_size,224,224,3], name='bn0')
    l1 = ConvBlock(input_shape=[batch_size,224,224,3], filter_shape=[7,7,3,F], strides=[1,2,2,1], init=init, name='conv1')
    l2 = MaxPool(size=[batch_size,112,112,size], ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    l3 = DenseModel(input_shape=[batch_size,56,56,F], init=init, name='dense_model', k=k, L=L)
    l4 = AvgPool(size=[batch_size,7,7,size], ksize=[1,7,7,1], strides=[1,7,7,1], padding='SAME')

    l5 = ConvToFullyConnected(input_shape=[batch_size,1,1,size]) 
    l6 = FullyConnected(input_shape=size, size=1000, init=init, name="fc1")

    model = Model(layers=[l1, l2, l3, l4, l5, l6])
    return model
