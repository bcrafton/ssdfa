
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--alg', type=str, default='bp')
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="gd")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="cifar10_fc_weights")
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
from tensorflow.examples.tutorials.mnist import input_data

from Model import Model

from Layer import Layer 
from ConvToFullyConnected import ConvToFullyConnected
from FullyConnected import FullyConnected
from Convolution import Convolution
from MaxPool import MaxPool
from Dropout import Dropout
from FeedbackFC import FeedbackFC
from FeedbackConv import FeedbackConv

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu
from Activation import Tanh
from Activation import Softmax
from Activation import LeakyRelu
from Activation import Linear

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size

bias = 0.1

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

Y = tf.placeholder(tf.float32, [None, 10])
X = tf.placeholder(tf.float32, [None, 3072])

l0 = Dropout(rate=dropout_rate/5.)

l1 = FullyConnected(size=[3072, 1000], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Relu(), bias=bias, l2=0.1, last_layer=False, name="fc1")
l2 = Dropout(rate=dropout_rate)
l3 = FeedbackFC(size=[3072, 1000], num_classes=10, sparse=args.sparse, rank=args.rank, name="fc1_fb")

l4 = FullyConnected(size=[1000, 1000], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Relu(), bias=bias, l2=0.1, last_layer=False, name="fc2")
l5 = Dropout(rate=dropout_rate)
l6 = FeedbackFC(size=[1000, 1000], num_classes=10, sparse=args.sparse, rank=args.rank, name="fc2_fb")

l7 = FullyConnected(size=[1000, 1000], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Relu(), bias=bias, l2=0.1, last_layer=False, name="fc3")
l8 = Dropout(rate=dropout_rate)
l9 = FeedbackFC(size=[1000, 1000], num_classes=10, sparse=args.sparse, rank=args.rank, name="fc3_fb")

l10 = FullyConnected(size=[1000, 10], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=bias, l2=0.1, last_layer=True, name="fc4")

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])

##############################################

predict = model.predict(X=X)

weights = model.get_weights()

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay":
    if args.alg == 'dfa':
        grads_and_vars = model.dfa_gvs(X=X, Y=Y)
    elif args.alg == 'lel':
        grads_and_vars = model.lel_gvs(X=X, Y=Y)
    elif args.alg == 'bp':
        grads_and_vars = model.gvs(X=X, Y=Y)
    else:
        assert(False)
        
    if args.opt == "adam":
        train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "rms":
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=1.0).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "decay":
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
    else:
        assert(False)

else:
    if args.alg == 'dfa':
        train = model.dfa(X=X, Y=Y)
    elif args.alg == 'lel':
        train = model.lel(X=X, Y=Y)
    elif args.alg == 'bp':
        train = model.train(X=X, Y=Y)
    else:
        assert(False)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

# x_train = x_train / 255.
x_train = (x_train - mean) / std
x_train = x_train.reshape(TRAIN_EXAMPLES, 3072)
y_train = keras.utils.to_categorical(y_train, 10)

# x_test = x_test / 255.
x_test = (x_test - mean) / std
x_test = x_test.reshape(TEST_EXAMPLES, 3072)
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
        _correct, _ = sess.run([total_correct, train], feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.5, learning_rate: lr, X: xs, Y: ys})
        
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
    f.write(str(test_acc) + "\n")
    f.close()

##############################################

if args.save:
    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_accs
    w['test_acc'] = test_accs
    np.save(args.name, w)
    
##############################################

