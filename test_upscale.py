
##############################################

import time
import tensorflow as tf
import keras
import math
import numpy as np
import matplotlib.pyplot as plt

from lib.UpSample import UpSample

##############################################

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 32

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

##############################################

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
batch_size = tf.placeholder(tf.int32, shape=())

l0 = UpSample(size=[batch_size, 32, 32, 3], ksize=2)

forward = l0.forward(X)
backward = l0.backward(X, forward, forward)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

##############################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
mean = np.mean(x_train, axis=(1, 2, 3), keepdims=True)
std = np.std(x_train, axis=(1, 2, 3), ddof=1, keepdims=True)
scale = std + 1.
# x_train = x_train - mean
x_train = x_train / scale

xs = x_train[0:BATCH_SIZE]
# print (np.shape(xs))
[_forward, _backward] = sess.run([forward, backward], feed_dict={batch_size: BATCH_SIZE, X: xs})
# print (np.shape(_forward), np.shape(_backward))

_forward = _forward / np.max(_forward)
_backward = _backward / np.max(_backward)

plt.imsave('up0.png', _forward[0, :, :, :])
plt.imsave('down0.png', _backward[0, :, :, :])

plt.imsave('up1.png', _forward[1, :, :, :])
plt.imsave('down1.png', _backward[1, :, :, :])



