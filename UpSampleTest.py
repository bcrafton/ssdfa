
import argparse
import os
import sys

##############################################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

##############################################

import time
import tensorflow as tf
import keras
import math
import numpy as np
import matplotlib.pyplot as plt

from lib.Model import Model

from lib.Activation import Relu
from lib.UpSample import UpSample

##############################################

X = tf.placeholder(tf.float32, [1, 32, 32, 3])
X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)

l0 = UpSample(input_shape=[1, 32, 32, 3], ksize=2)
forward = l0.forward(X=X)
backward = l0.backward(AI=X, AO=None, DO=forward['aout'])

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

##############################################

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train / np.max(x_train)

##############################################

XS = x_train[0]
XS = np.reshape(XS, (1, 32, 32, 3))

[_f, _b] = sess.run([forward, backward], feed_dict={X:XS})

_f = _f['aout']
_b = _b['dout']

_f = np.reshape(_f, (64, 64, 3))
_b = np.reshape(_b, (32, 32, 3))

plt.imsave('forward.jpg', _f)
plt.imsave('backward.jpg', _b)

##############################################













