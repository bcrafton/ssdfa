
import os
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist

##############################################

mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
rate = 0.25
mask = np.random.choice([0, 1], size=(784, 10), p=[1.-rate, rate])
num = np.count_nonzero(mask)

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_train /= 255.
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')
x_test /= 255.
y_test = keras.utils.to_categorical(y_test, 10)

EPOCHS = 10
BATCH_SIZE = 64

##############################################

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
# mask = tf.placeholder(tf.float32, [None, 10])

sqrt_fan_in = np.sqrt(784)
idx = tf.where(mask > 0)
val = tf.Variable(tf.random_uniform(minval=-1./sqrt_fan_in, maxval=1./sqrt_fan_in, shape=(num,)))
bias = tf.Variable(tf.zeros(shape=[10]))

##############################################
# FEED FORWARD
##############################################

weights = tf.SparseTensor(indices=idx, values=val, dense_shape=(784, 10))
Z = tf.sparse_tensor_dense_matmul(tf.sparse_transpose(weights), tf.transpose(X))
Z = tf.transpose(Z)
Z = Z + bias
A = tf.nn.softmax(Z)

##############################################
# BACK PROP
##############################################

E = tf.subtract(A, Y)

slice1 = tf.slice(idx, [0, 0], [num, 1])
slice2 = tf.slice(idx, [0, 1], [num, 1])
slice_X = tf.gather_nd(tf.transpose(X), slice1)
slice_E = tf.gather_nd(tf.transpose(E), slice2)
DW = tf.multiply(slice_X, slice_E)
DW = tf.reduce_sum(DW, axis=1)
# DW = tf.matmul(tf.transpose(X), E) * mask
# DW = tf.gather_nd(DW, idx)

DB = tf.reduce_sum(E, axis=0)

val = val.assign(tf.subtract(val, tf.scalar_mul(0.01, DW)))
bias = bias.assign(tf.subtract(bias, tf.scalar_mul(0.01, DB)))

##############################################

correct = tf.equal(tf.argmax(A,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(EPOCHS):
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        w, b = sess.run([val, bias], feed_dict={X: xs, Y: ys})
        # x, e = sess.run([slice_X, slice_E], feed_dict={X: xs, Y: ys})
        # print (np.shape(x))
        # print (np.shape(e))

    acc = sess.run(total_correct, feed_dict={X: x_test, Y: y_test})
    print (acc)

assert (np.count_nonzero(w) == np.count_nonzero(mask))

##############################################












