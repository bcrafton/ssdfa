
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='tanh')
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

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution2D import Convolution2D
from lib.Convolution3D import Convolution3D
from lib.ConvolutionDW import ConvolutionDW
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv
from lib.BatchNorm import BatchNorm

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

train_conv = True
weights_conv = None

train_conv_dw = True
weights_conv_dw = None

train_conv_pw = True
weights_conv_pw = None

train_fc = True
weights_fc = None

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)
Y = tf.placeholder(tf.float32, [None, 10])

l0_1 = Convolution2D(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[3, 3, 3, 32], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv1", load=weights_conv, train=train_conv)
l0_2 = BatchNorm(input_size=[batch_size, 32, 32, 32], name='conv1_bn', load=weights_conv, train=train_conv)
l0_3 = Relu()

l1_1 = ConvolutionDW(input_sizes=[batch_size, 32, 32, 32], filter_sizes=[3, 3, 32, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_1", load=weights_conv_dw, train=train_conv_dw)
l1_2 = BatchNorm(input_size=[batch_size, 32, 32, 32], name='conv_dw_1_bn', load=weights_conv_dw, train=train_conv_dw)
l1_3 = Relu()
l1_4 = Convolution2D(input_sizes=[batch_size, 32, 32, 32], filter_sizes=[1, 1, 32, 32], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_1", load=weights_conv_pw, train=train_conv_pw)
l1_5 = BatchNorm(input_size=[batch_size, 32, 32, 32], name='conv_pw_1_bn', load=weights_conv_pw, train=train_conv_pw)
l1_6 = Relu()

l2_1 = ConvolutionDW(input_sizes=[batch_size, 32, 32, 32], filter_sizes=[3, 3, 32, 1], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_2", load=weights_conv_dw, train=train_conv_dw)
l2_2 = BatchNorm(input_size=[batch_size, 16, 16, 32], name='conv_dw_2_bn', load=weights_conv_dw, train=train_conv_dw)
l2_3 = Relu()
l2_4 = Convolution2D(input_sizes=[batch_size, 16, 16, 32], filter_sizes=[1, 1, 32, 64], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_2", load=weights_conv_pw, train=train_conv_pw)
l2_5 = BatchNorm(input_size=[batch_size, 16, 16, 64], name='conv_pw_2_bn', load=weights_conv_pw, train=train_conv_pw)
l2_6 = Relu()

l3_1 = ConvolutionDW(input_sizes=[batch_size, 16, 16, 64], filter_sizes=[3, 3, 64, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_3", load=weights_conv_dw, train=train_conv_dw)
l3_2 = BatchNorm(input_size=[batch_size, 16, 16, 64], name='conv_dw_3_bn', load=weights_conv_dw, train=train_conv_dw)
l3_3 = Relu()
l3_4 = Convolution2D(input_sizes=[batch_size, 16, 16, 64], filter_sizes=[1, 1, 64, 64], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_3", load=weights_conv_pw, train=train_conv_pw)
l3_5 = BatchNorm(input_size=[batch_size, 16, 16, 64], name='conv_pw_3_bn', load=weights_conv_pw, train=train_conv_pw)
l3_6 = Relu()

l4_1 = ConvolutionDW(input_sizes=[batch_size, 16, 16, 64], filter_sizes=[3, 3, 64, 1], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_4", load=weights_conv_dw, train=train_conv_dw)
l4_2 = BatchNorm(input_size=[batch_size, 8, 8, 64], name='conv_dw_4_bn', load=weights_conv_dw, train=train_conv_dw)
l4_3 = Relu()
l4_4 = Convolution2D(input_sizes=[batch_size, 8, 8, 64], filter_sizes=[1, 1, 64, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_4", load=weights_conv_pw, train=train_conv_pw)
l4_5 = BatchNorm(input_size=[batch_size, 8, 8, 128], name='conv_pw_4_bn', load=weights_conv_pw, train=train_conv_pw)
l4_6 = Relu()

##############################################

l5_1 = ConvolutionDW(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[3, 3, 128, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_5", load=weights_conv_dw, train=train_conv_dw)
l5_2 = BatchNorm(input_size=[batch_size, 8, 8, 128], name='conv_dw_5_bn', load=weights_conv_dw, train=train_conv_dw)
l5_3 = Relu()
l5_4 = Convolution2D(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[1, 1, 128, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_5", load=weights_conv_pw, train=train_conv_pw)
l5_5 = BatchNorm(input_size=[batch_size, 8, 8, 128], name='conv_pw_5_bn', load=weights_conv_pw, train=train_conv_pw)
l5_6 = Relu()

l6_1 = ConvolutionDW(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[3, 3, 128, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_6", load=weights_conv_dw, train=train_conv_dw)
l6_2 = BatchNorm(input_size=[batch_size, 8, 8, 128], name='conv_dw_6_bn', load=weights_conv_dw, train=train_conv_dw)
l6_3 = Relu()
l6_4 = Convolution2D(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[1, 1, 128, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_6", load=weights_conv_pw, train=train_conv_pw)
l6_5 = BatchNorm(input_size=[batch_size, 8, 8, 128], name='conv_pw_6_bn', load=weights_conv_pw, train=train_conv_pw)
l6_6 = Relu()

l7_1 = ConvolutionDW(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[3, 3, 128, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_7", load=weights_conv_dw, train=train_conv_dw)
l7_2 = BatchNorm(input_size=[batch_size, 8, 8, 128], name='conv_dw_7_bn', load=weights_conv_dw, train=train_conv_dw)
l7_3 = Relu()
l7_4 = Convolution2D(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[1, 1, 128, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_7", load=weights_conv_pw, train=train_conv_pw)
l7_5 = BatchNorm(input_size=[batch_size, 8, 8, 128], name='conv_pw_7_bn', load=weights_conv_pw, train=train_conv_pw)
l7_6 = Relu()

##############################################

l8_1 = ConvolutionDW(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[3, 3, 128, 1], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_8", load=weights_conv_dw, train=train_conv_dw)
l8_2 = BatchNorm(input_size=[batch_size, 4, 4, 128], name='conv_dw_8_bn', load=weights_conv_dw, train=train_conv_dw)
l8_3 = Relu()
l8_4 = Convolution2D(input_sizes=[batch_size, 4, 4, 128], filter_sizes=[1, 1, 128, 256], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_8", load=weights_conv_pw, train=train_conv_pw)
l8_5 = BatchNorm(input_size=[batch_size, 4, 4, 256], name='conv_pw_8_bn', load=weights_conv_pw, train=train_conv_pw)
l8_6 = Relu()

l9_1 = ConvolutionDW(input_sizes=[batch_size, 4, 4, 256], filter_sizes=[3, 3, 256, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_dw_9", load=weights_conv_dw, train=train_conv_dw)
l9_2 = BatchNorm(input_size=[batch_size, 4, 4, 256], name='conv_dw_9_bn', load=weights_conv_dw, train=train_conv_dw)
l9_3 = Relu()
l9_4 = Convolution2D(input_sizes=[batch_size, 4, 4, 256], filter_sizes=[1, 1, 256, 256], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Linear(), bias=args.bias, name="conv_pw_9", load=weights_conv_pw, train=train_conv_pw)
l9_5 = BatchNorm(input_size=[batch_size, 4, 4, 256], name='conv_pw_9_bn', load=weights_conv_pw, train=train_conv_pw)
l9_6 = Relu()

##############################################

l10 = AvgPool(size=[batch_size, 4, 4, 256], ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

##############################################

l11 = ConvToFullyConnected(shape=[1, 1, 256])

l12 = FullyConnected(size=[256, 10], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=args.bias, last_layer=True, name="fc1", load=weights_fc, train=train_fc)

##############################################

model = Model(layers=[l0_1, l0_2, l0_3,                               \
                      l1_1, l1_2, l1_3, l1_4, l1_5, l1_6,             \
                      l2_1, l2_2, l2_3, l2_4, l2_5, l2_6,             \
                      l3_1, l3_2, l3_3, l3_4, l3_5, l3_6,             \
                      l4_1, l4_2, l4_3, l4_4, l4_5, l4_6,             \
                      l5_1, l5_2, l5_3, l5_4, l5_5, l5_6,             \
                      l6_1, l6_2, l6_3, l6_4, l6_5, l6_6,             \
                      l7_1, l7_2, l7_3, l7_4, l7_5, l7_6,             \
                      l8_1, l8_2, l8_3, l8_4, l8_5, l8_6,             \
                      l9_1, l9_2, l9_3, l9_4, l8_5, l8_6,             \
                      l10,                                            \
                      l11,                                            \
                      l12])

predict = model.predict(X=X)

weights = model.get_weights()

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay":
    if args.dfa:
        grads_and_vars = model.dfa_gvs(X=X, Y=Y)
    else:
        grads_and_vars = model.gvs(X=X, Y=Y)
        
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
        train = model.dfa(X=X, Y=Y)
    else:
        train = model.train(X=X, Y=Y)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

assert(np.shape(x_train) == (50000, 32, 32, 3))
assert(np.shape(x_test) == (10000, 32, 32, 3))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

##############################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

train_accs = []
test_accs = []

for ii in range(EPOCHS):
    if args.opt == 'decay' or args.opt == 'gd':
        decay = np.power(args.decay, ii)
        lr = args.alpha * decay
    else:
        lr = args.alpha
        
    print (ii)
    
    #############################
    
    _count = 0
    _total_correct = 0
    
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct, _ = sess.run([total_correct, train], feed_dict={batch_size: BATCH_SIZE, dropout_rate: args.dropout, learning_rate: lr, X: xs, Y: ys})
        
        _total_correct += _correct
        _count += BATCH_SIZE

    train_acc = 1.0 * _total_correct / _count
    train_accs.append(train_acc)

    #############################

    _count = 0
    _total_correct = 0

    for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
        xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct = sess.run(total_correct, feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
        
        _total_correct += _correct
        _count += BATCH_SIZE
        
    test_acc = 1.0 * _total_correct / _count
    test_accs.append(test_acc)
    
    #############################
            
    print ("train acc: %f test acc: %f" % (train_acc, test_acc))
    
    f = open(filename, "a")
    f.write("train acc: %f test acc: %f\n" % (train_acc, test_acc))
    f.close()

##############################################

if args.save:
    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_accs
    w['test_acc'] = test_accs
    np.save(args.name, w)
    
##############################################


