
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="glorot_uniform")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="cifar10_conv")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import tensorflow as tf
import keras
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

from lib.Activation import Relu
from lib.Activation import Tanh

##############################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_examples = 50000
test_examples = 10000

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train = np.concatenate((x_train, -1. * x_train), axis=3)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test = np.concatenate((x_test, -1. * x_test), axis=3)
y_test = keras.utils.to_categorical(y_test, 10)

##############################################

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
lr = tf.placeholder(tf.float32, shape=())

X = tf.placeholder(tf.float32, [None, 32, 32, 6])
Y = tf.placeholder(tf.float32, [None, 10])

l0 = Convolution(input_shape=[batch_size, 32, 32, 6], filter_sizes=[5, 5, 6, 128], init=args.init, activation=act, bias=args.bias, name='conv1')
l1 = MaxPool(size=[batch_size, 32, 32, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2 = Convolution(input_shape=[batch_size, 16, 16, 128], filter_sizes=[5, 5, 128, 192], init=args.init, activation=act, bias=args.bias, name='conv2')
l3 = MaxPool(size=[batch_size, 16, 16, 192], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l4 = Convolution(input_shape=[batch_size, 8, 8, 192], filter_sizes=[5, 5, 192, 256], init=args.init, activation=act, bias=args.bias, name='conv3')
l5 = MaxPool(size=[batch_size, 8, 8, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l6 = ConvToFullyConnected(input_shape=[4, 4, 256])

l7 = FullyConnected(input_shape=4*4*256, size=10, init=args.init, bias=args.bias, name='fc1')

##############################################

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7])
predict = model.predict(X=X)
weights = model.get_weights()

if args.dfa:
    grads_and_vars = model.dfa_gvs(X=X, Y=Y)
else:
    grads_and_vars = model.gvs(X=X, Y=Y)

[conv3, conv2, conv1, fc1] = grads_and_vars
        
train1 = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=[fc1, conv1])
train2 = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=[fc1, conv2])
train3 = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=[fc1, conv3])

correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

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

    #############################
    
    _total_correct = 0
    for jj in range(0, train_examples, args.batch_size):
        s = jj
        e = min(jj + args.batch_size, train_examples)
        b = e - s
        
        xs = x_train[s:e]
        ys = y_train[s:e]
        
        if ii < 5:
            _correct, _ = sess.run([total_correct, train1], feed_dict={batch_size: b, dropout_rate: args.dropout, lr: args.lr, X: xs, Y: ys})
        elif ii < 10:
            _correct, _ = sess.run([total_correct, train2], feed_dict={batch_size: b, dropout_rate: args.dropout, lr: args.lr, X: xs, Y: ys})
        elif ii < 15:
            _correct, _ = sess.run([total_correct, train3], feed_dict={batch_size: b, dropout_rate: args.dropout, lr: args.lr, X: xs, Y: ys})
            
        _total_correct += _correct

    train_acc = 1.0 * _total_correct / (train_examples - (train_examples % args.batch_size))
    train_accs.append(train_acc)

    #############################

    _total_correct = 0
    for jj in range(0, test_examples, args.batch_size):
        s = jj
        e = min(jj + args.batch_size, test_examples)
        b = e - s
        
        xs = x_test[s:e]
        ys = y_test[s:e]
        
        _correct = sess.run(total_correct, feed_dict={batch_size: b, dropout_rate: 0.0, lr: 0.0, X: xs, Y: ys})
        _total_correct += _correct
        
    test_acc = 1.0 * _total_correct / (test_examples - (test_examples % args.batch_size))
    test_accs.append(test_acc)
    
    #############################
            
    p = "%d | train acc: %f | test acc: %f" % (ii, train_acc, test_acc)
    print (p)
    f = open(filename, "a")
    f.write(p + "\n")
    f.close()

##############################################

if args.save:
    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_accs
    w['test_acc'] = test_accs
    np.save(args.name, w)
    
##############################################


