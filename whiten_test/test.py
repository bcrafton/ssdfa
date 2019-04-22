
import numpy as np
import tensorflow as tf
from whiten import whiten1
from whiten import whiten2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784)

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#x_train = x_train.reshape(50000, 1024*3)

white = whiten1(x_train)
white = whiten2(x_train)
