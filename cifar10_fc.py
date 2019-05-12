
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--time_steps', type=int, default=32)
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
parser.add_argument('--name', type=str, default="cifar10_fc")
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
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.SpikingFC import SpikingFC
from lib.SpikingTimeConv import SpikingTimeConv
from lib.SpikingSum import SpikingSum

from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.Activation import Relu
from lib.Activation import Tanh
from lib.Activation import Softmax
from lib.Activation import LeakyRelu
from lib.Activation import Linear

##############################################

def to_spike_train(mat):
    shape = np.shape(mat)
    assert(len(shape) == 2)
    N, O = shape

    mat = mat / 8.
    mat = np.floor(mat).astype(int)
    mat = np.reshape(mat, N*O)
    mat = keras.utils.to_categorical(mat, 32)
    mat = np.reshape(mat, (N, O, 32))
    mat = np.transpose(mat, (0, 2, 1))
    
    assert(np.count_nonzero(mat) == N * O)
    
    return mat

'''
cmp_arr = np.random.uniform(low=0., high=1., size=(1, args.time_steps, 1))
def to_spike_train(mat):
    shape = np.shape(mat)
    assert(len(shape) == 2)
    N, O = shape
    mat = np.reshape(mat, (N, 1, O))
    
    out_shape = N, args.time_steps, O
    train = cmp_arr < mat
    
    return train
'''
    
##############################################

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

X = tf.placeholder(tf.float32, [args.batch_size, args.time_steps, 3072])
Y = tf.placeholder(tf.float32, [args.batch_size, 10])

l0 = SpikingFC(input_shape=[args.batch_size, args.time_steps, 3072], size=128, init=args.init, activation=Linear(), name="sfc1")
l1 = SpikingTimeConv(input_shape=[args.batch_size, args.time_steps, 128], filter_size=5, init=args.init, activation=Linear(), name="stc1", train=False)
l2 = Relu()

l3 = SpikingFC(input_shape=[args.batch_size, args.time_steps, 128], size=128, init=args.init, activation=Linear(), name="sfc2")
l4 = SpikingTimeConv(input_shape=[args.batch_size, args.time_steps, 128], filter_size=5, init=args.init, activation=Linear(), name="stc2", train=False)
l5 = Relu()

l6 = SpikingSum(input_shape=[args.batch_size, args.time_steps, 128], init=args.init, activation=Relu(), name="ss1", train=False)
l7 = FullyConnected(input_shape=128, size=10, init=args.init, activation=Linear(), name='fc1')

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7])

##############################################

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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert(np.shape(x_train) == (TRAIN_EXAMPLES, 32, 32, 3))
assert(np.shape(x_test) == (TEST_EXAMPLES, 32, 32, 3))

x_train = np.reshape(x_train, (TRAIN_EXAMPLES, 32*32*3))
#x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, 10)

x_test = np.reshape(x_test, (TEST_EXAMPLES, 32*32*3))
#x_test = x_test / 255.
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

for ii in range(args.epochs):
    if args.opt == 'decay' or args.opt == 'gd':
        decay = np.power(args.decay, ii)
        lr = args.alpha * decay
    else:
        lr = args.alpha
        
    print (ii)
    
    #############################
    
    _count = 0
    _total_correct = 0
    
    for jj in range(0, TRAIN_EXAMPLES, args.batch_size):
        start = jj
        end = jj + args.batch_size
        assert(end <= TRAIN_EXAMPLES)
        
        xs = x_train[start:end]
        xs = to_spike_train(xs)
        
        # print (xs[0], np.shape(xs[0]))
        # print (np.sum(xs[0], axis=0), np.shape(xs[0]))
        # print (xs[0].T)
        
        # xs = xs * 1.0 / 64.
        ys = y_train[start:end]
        
        _correct, _ = sess.run([total_correct, train], feed_dict={dropout_rate: args.dropout, learning_rate: lr, X: xs, Y: ys})
        
        _total_correct += _correct 
        _count += args.batch_size

    train_acc = 1.0 * _total_correct / _count
    train_accs.append(train_acc)

    #############################

    _count = 0
    _total_correct = 0

    for jj in range(0, TEST_EXAMPLES, args.batch_size):
        start = jj
        end = jj + args.batch_size
        assert(end <= TEST_EXAMPLES)
        
        xs = x_test[start:end]
        xs = to_spike_train(xs)
        # xs = xs * 1.0 / 64.
        ys = y_test[start:end]
        
        _correct = sess.run(total_correct, feed_dict={dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
        
        _total_correct += _correct
        _count += args.batch_size
        
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


