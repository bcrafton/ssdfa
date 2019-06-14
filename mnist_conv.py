
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="mnist_conv")
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
from lib.AvgPool import AvgPool
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

mnist = tf.keras.datasets.mnist.load_data()

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

train_fc=True
if args.load:
    train_conv=False
else:
    train_conv=True

weights_fc=None
weights_conv=args.load

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)
Y = tf.placeholder(tf.float32, [None, 10])

'''
l0 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='conv1', load=weights_conv, train=train_conv)
l1 = FeedbackConv(size=[batch_size, 28, 28, 32], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv1_fb')

l2 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='conv2', load=weights_conv, train=train_conv)
l3 = MaxPool(size=[batch_size, 28, 28, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
l4 = Dropout(rate=dropout_rate/2.)
l5 = FeedbackConv(size=[batch_size, 14, 14, 64], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv2_fb')

l6 = ConvToFullyConnected(shape=[14, 14, 64])
l7 = FullyConnected(size=[14*14*64, 128], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, name='fc1', load=weights_fc, train=train_fc)
l8 = Dropout(rate=dropout_rate)
l9 = FeedbackFC(size=[14*14*64, 128], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc1_fb')

l10 = FullyConnected(size=[128, 10], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=args.bias, last_layer=True, name='fc2', load=weights_fc, train=train_fc)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])
'''

l1_1 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 64], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
l1_2 = Relu()
l1_3 = Convolution(input_sizes=[batch_size, 28, 28, 64], filter_sizes=[3, 3, 64, 64], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
l1_4 = Relu()
l1_5 = AvgPool(size=[batch_size, 28, 28, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2_1 = Convolution(input_sizes=[batch_size, 14, 14, 64], filter_sizes=[3, 3, 64, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
l2_2 = Relu()
l2_3 = Convolution(input_sizes=[batch_size, 14, 14, 128], filter_sizes=[3, 3, 128, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
l2_4 = Relu()
l2_5 = AvgPool(size=[batch_size, 14, 14, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l3_1 = ConvToFullyConnected(input_shape=[7, 7, 128])
l3_2 = FullyConnected(input_shape=7*7*128, size=10, init=args.init, name="fc1")

layers = [
l1_1, l1_2, l1_3, l1_4, l1_5,
l2_1, l2_2, l2_3, l2_4, l2_5,
l3_1, l3_2
]
model = Model(layers=layers)

##############################################

predict = model.predict(X=X)
upto = model.up_to(X=X, N=9)
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

(x_train, y_train), (x_test, y_test) = mnist

x_train = x_train.reshape(TRAIN_EXAMPLES, 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TEST_EXAMPLES, 28, 28, 1)
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
    
    feature_maps = []
    labels = []
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct, _, _upto = sess.run([total_correct, train, upto], feed_dict={batch_size: BATCH_SIZE, dropout_rate: args.dropout, learning_rate: lr, X: xs, Y: ys})
        
        # print (np.shape(_upto))
        
        _total_correct += _correct
        _count += BATCH_SIZE
        
        feature_maps.append(_upto)
        labels.append(ys)

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

###########################

feature_maps = np.concatenate(feature_maps, axis=0)
labels = np.concatenate(labels, axis=0)

ones = np.where(labels==1)
twos = np.where(labels==2)

ones_img = feature_maps[ones]
twos_img = feature_maps[twos]

np.save('conv_ones', ones_img)
np.save('conv_twos', twos_img)









