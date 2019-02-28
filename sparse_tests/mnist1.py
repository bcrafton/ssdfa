
import os
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist

##############################################

mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

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

sqrt_fan_in = np.sqrt(784)
weights = tf.Variable(tf.random_uniform(minval=-1./sqrt_fan_in, maxval=1./sqrt_fan_in, shape=[784, 10]))
bias = tf.Variable(tf.zeros(shape=[10]))

##############################################
# FEED FORWARD
##############################################

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

Z = tf.matmul(X, weights) + bias
A = tf.nn.softmax(Z)

##############################################
# BACK PROP
##############################################

E = tf.subtract(A, Y)

DW = tf.matmul(tf.transpose(X), E)
DB = tf.reduce_sum(E, axis=0)

weights = weights.assign(tf.subtract(weights, tf.scalar_mul(0.01, DW)))
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
        sess.run([weights, bias], feed_dict={X: xs, Y: ys})

    acc = sess.run(total_correct, feed_dict={X: x_test, Y: y_test})
    print (acc)

##############################################












