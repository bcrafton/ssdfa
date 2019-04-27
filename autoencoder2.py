
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1e-5)
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
parser.add_argument('--name', type=str, default="autoencoder")
parser.add_argument('--custom', type=int, default=0)
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

from lib.ModelMSE import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.CustomConvolution import Convolution
from lib.MaxPool import MaxPool
from lib.UpSample import UpSample
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
train_conv=True

weights_fc=None
weights_conv=None

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)

l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, name='conv1', load=weights_conv, train=train_conv, custom=args.custom)
l1 = MaxPool(size=[batch_size, 32, 32, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

l2 = Convolution(input_sizes=[batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, name='conv2', load=weights_conv, train=train_conv, custom=args.custom)
l3 = MaxPool(size=[batch_size, 16, 16, 128], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

l4 = Convolution(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name='conv3', load=weights_conv, train=train_conv, custom=args.custom)
l5 = UpSample(size=[batch_size, 8, 8, 128], ksize=2)

# NOT SURE IF WE SHUD END ON A UPSAMPLE OR A CONV.

l6 = Convolution(input_sizes=[batch_size, 16, 16, 128], filter_sizes=[5, 5, 128, 96], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name='conv4', load=weights_conv, train=train_conv, custom=args.custom)
l7 = UpSample(size=[batch_size, 16, 16, 96], ksize=2)

l8 = Convolution(input_sizes=[batch_size, 32, 32, 96], filter_sizes=[5, 5, 96, 3], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name='conv5', load=weights_conv, train=train_conv, custom=args.custom)

##############################################

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8])
predict = model.predict(X=X)
weights = model.get_weights()

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay":
    if args.dfa:
        grads_and_vars = model.dfa_gvs(X=X, Y=X)
    else:
        grads_and_vars, loss = model.gvs(X=X, Y=X)
        
    if args.opt == "adam":
        train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "rms":
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "decay":
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
    else:
        assert(False)

else:
    if args.dfa:
        train = model.dfa(X=X, Y=X)
    else:
        train = model.train(X=X, Y=X)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
mean = np.mean(x_train, axis=(1, 2, 3), keepdims=True)
std = np.std(x_train, axis=(1, 2, 3), ddof=1, keepdims=True)
scale = std + 1.
x_train = x_train - mean
x_train = x_train / scale

##############################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

for ii in range(EPOCHS):
    if args.opt == 'decay' or args.opt == 'gd':
        decay = np.power(args.decay, ii)
        lr = args.alpha * decay
    else:
        lr = args.alpha
        
    print (ii)
    
    #############################
    
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        [_, _gvs] = sess.run([train, grads_and_vars], feed_dict={batch_size: BATCH_SIZE, dropout_rate: args.dropout, learning_rate: lr, X: xs})
    
        #for gv in _gvs:
        #    print (np.shape(gv[0]), np.std(gv[0]), np.std(gv[1]))

    #############################

    if args.save:
        [w] = sess.run([weights], feed_dict={})
        np.save(args.name, w)

