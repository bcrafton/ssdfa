
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="cifar10_conv")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import time
import tensorflow as tf
import keras
import math
import numpy as np
from whiten import whiten

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.Activation import Relu
from lib.Activation import Tanh
from lib.Activation import Softmax
from lib.Activation import LeakyRelu
from lib.Activation import Linear

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

train_fc=True
weights_fc=None

train_conv=True
weights_conv='filters.npy'

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)
Y = tf.placeholder(tf.float32, [None, 10])

l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[6, 6, 3, 96], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='conv1', load=weights_conv, train=train_conv)
l1 = MaxPool(size=[batch_size, 16, 16, 96], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

##############################################

model = Model(layers=[l0, l1])
predict = model.predict(X=X)
weights = model.get_weights()

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

y_train = keras.utils.to_categorical(y_train, 10)
if np.shape(x_train) != (TRAIN_EXAMPLES, 32, 32, 3):
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    
y_test = keras.utils.to_categorical(y_test, 10)
if np.shape(x_test) != (TEST_EXAMPLES, 32, 32, 3):
    x_test = np.transpose(x_test, (0, 2, 3, 1))

'''
x_train = whiten(X=x_train, method='zca')
x_train = np.reshape(x_train, (TRAIN_EXAMPLES, 32, 32, 3))

x_test = whiten(X=x_test, method='zca')
x_test = np.reshape(x_test, (TEST_EXAMPLES, 32, 32, 3))
'''
##############################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

train_features = np.zeros(shape=(50000, 16, 16, 96), dtype=np.float32)
test_features = np.zeros(shape=(10000, 16, 16, 96), dtype=np.float32)
    
#############################
    
for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
    xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
    ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
    _predict = sess.run(predict, feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
    train_features[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE] = _predict
    
#############################

for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
    xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
    ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
    _predict = sess.run(predict, feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
    test_features[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE] = _predict

#############################

np.save('x_train', train_features)
np.save('x_test', test_features)





