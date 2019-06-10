
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
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="autoencoder")
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
from lib.Convolution import Convolution
from lib.AvgPool import AvgPool
from lib.UpSample import UpSample
from lib.Dropout import Dropout

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

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)

l1_1 = Convolution(input_sizes=[args.batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], init=args.init, strides=[1,1,1,1], padding="SAME", name='conv1')
l1_2 = Relu()
l1_3 = AvgPool(size=[args.batch_size, 32, 32, 96], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2_1 = Convolution(input_sizes=[args.batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name='conv2')
l2_2 = Relu()
l2_3 = AvgPool(size=[args.batch_size, 16, 16, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l3_1 = Convolution(input_sizes=[args.batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name='conv3')
l3_2 = UpSample(input_shape=[args.batch_size, 8, 8, 128], ksize=2)

l4_1 = Convolution(input_sizes=[args.batch_size, 16, 16, 128], filter_sizes=[5, 5, 128, 96], init=args.init, strides=[1,1,1,1], padding="SAME", name='conv4')
l4_2 = UpSample(input_shape=[args.batch_size, 16, 16, 96], ksize=2)

l5 = Convolution(input_sizes=[args.batch_size, 32, 32, 96], filter_sizes=[5, 5, 96, 3], init=args.init, strides=[1,1,1,1], padding="SAME", name='conv5')

##############################################

layers = [
l1_1, l1_2, l1_3, 
l2_1, l2_2, l2_3, 
l3_1, l3_2,
l4_1, l4_2,
l5
]
          
model = Model(layers=layers, shape_y=[args.batch_size, 32, 32, 3])
                     
grads_and_vars, loss = model.gvs(X=X, Y=X)
train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)

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

for ii in range(args.epochs):
    if args.opt == 'decay' or args.opt == 'gd':
        decay = np.power(args.decay, ii)
        lr = args.alpha * decay
    else:
        lr = args.alpha
        
    print (ii)
    
    #############################
    
    for jj in range(int(TRAIN_EXAMPLES / args.batch_size)):
        xs = x_train[jj*args.batch_size:(jj+1)*args.batch_size]
        ys = y_train[jj*args.batch_size:(jj+1)*args.batch_size]
        [_, _loss] = sess.run([train, loss], feed_dict={dropout_rate: args.dropout, learning_rate: lr, X: xs})
        print (_loss)
    
    #############################

